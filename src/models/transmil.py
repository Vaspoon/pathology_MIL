"""
src/models/transmil.py — TransMIL (Shao et al., NeurIPS 2021)
https://arxiv.org/abs/2106.00908

Key components:
  NystromAttention  : efficient O(N) approximation of self-attention (Xiong et al., 2021)
  PPEG              : Pyramid Position Encoding Generator (depthwise multi-scale convs)
  TransLayer        : pre-norm Transformer block with NystromAttention
  TransMIL          : full model — returns (logits, attn) compatible with TrainerABMIL
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NystromAttention(nn.Module):
    """
    Nystrom-based self-attention (Xiong et al., 2021) without external dependencies.

    When N <= num_landmarks the module falls back to standard full attention.
    Includes a depth-wise residual convolution on V (sequence dimension) as in the
    original nystrom-attention package used by TransMIL.
    """

    def __init__(
        self,
        dim:              int,
        num_heads:        int   = 8,
        num_landmarks:    int   = 256,
        pinv_iterations:  int   = 6,
        dropout:          float = 0.0,
        residual_kernel:  int   = 33,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads       = num_heads
        self.num_landmarks   = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.head_dim        = dim // num_heads
        self.scale           = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

        # Depth-wise conv residual applied along sequence dimension of V
        padding = residual_kernel // 2
        self.res_conv = nn.Conv2d(
            num_heads, num_heads,
            kernel_size=(residual_kernel, 1),
            padding=(padding, 0),
            groups=num_heads,
            bias=False,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
        Returns:
            [B, N, dim]
        """
        B, N, _ = x.shape
        H, D, M = self.num_heads, self.head_dim, self.num_landmarks

        # Pad sequence so N is divisible by M (padding at front → original tokens at tail)
        remainder = N % M
        pad_len   = (M - remainder) if remainder else 0
        if pad_len:
            x = F.pad(x, (0, 0, pad_len, 0))
        Np = x.shape[1]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # [B, H, Np, D]
        q, k, v = (t.reshape(B, Np, H, D).permute(0, 2, 1, 3) for t in (q, k, v))
        q = q * self.scale

        if Np == M:
            # Standard full attention (sequence fits in one landmark segment)
            attn = torch.softmax(q @ k.transpose(-2, -1), dim=-1)  # [B, H, Np, Np]
            out  = attn @ v                                          # [B, H, Np, D]
        else:
            # Nystrom approximation: segment mean landmarks from Q and K
            seg = Np // M
            q_land = q.reshape(B, H, M, seg, D).mean(dim=-2)        # [B, H, M, D]
            k_land = k.reshape(B, H, M, seg, D).mean(dim=-2)        # [B, H, M, D]

            # Three kernel matrices
            sim1 = q @ q_land.transpose(-2, -1)                     # [B, H, Np, M]
            sim2 = q_land @ k_land.transpose(-2, -1)                # [B, H, M,  M]
            sim3 = q_land @ k.transpose(-2, -1)                     # [B, H, M,  Np]

            a1 = torch.softmax(sim1, dim=-1)                        # [B, H, Np, M]
            a2 = torch.softmax(sim2, dim=-1)                        # [B, H, M,  M]
            a3 = torch.softmax(sim3, dim=-1)                        # [B, H, M,  Np]

            a2_inv = self._iterative_pinv(a2, self.pinv_iterations) # [B, H, M,  M]
            out = a1 @ (a2_inv @ (a3 @ v))                          # [B, H, Np, D]

        # Depth-wise conv residual on V  (v treated as [B, H, Np, D] image)
        out = out + self.res_conv(v)                                 # [B, H, Np, D]

        # Remove front-padding and reshape
        out = out[:, :, pad_len:, :]                                 # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * D)          # [B, N, dim]
        return self.to_out(out)

    # ------------------------------------------------------------------
    @staticmethod
    def _iterative_pinv(A: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
        """
        Iterative Moore-Penrose pseudoinverse via Schulz–Hotelling iterations
        (cubic variant for faster convergence).

        A: [B, H, M, M]  (softmax attention matrix, non-negative)
        """
        abs_A    = A.abs()
        # Safe initialisation: V_0 = A^T / (max_col_sum * max_row_sum)
        col_max  = abs_A.sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)  # [B,H,1,1]
        row_max  = abs_A.sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)  # [B,H,1,1]
        V        = A.transpose(-2, -1) / (col_max * row_max + 1e-8)

        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)  # broadcast [M,M]
        for _ in range(n_iter):
            AV = A @ V
            V  = 0.25 * V @ (13 * I - AV @ (15 * I - AV @ (7 * I - AV)))
        return V


# ---------------------------------------------------------------------------

class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator (TransMIL paper, Fig. 2).

    Reshapes the flat patch sequence into a 2-D spatial grid, applies
    depth-wise convolutions at three kernel scales, and folds back.
    The class token is passed through unchanged.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x:    [B, H*W+1, C]  — class token at position 0
            H, W: spatial grid dims (H*W == number of patch tokens)
        Returns:
            [B, H*W+1, C]
        """
        B, _, C   = x.shape
        cls_token = x[:, :1, :]                                    # [B, 1, C]
        feat      = x[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)   # [B, C, H, W]
        pe = self.proj(feat) + feat + self.proj1(feat) + self.proj2(feat)
        pe = pe.flatten(2).transpose(1, 2)                         # [B, H*W, C]
        return torch.cat([cls_token, pe], dim=1)                   # [B, H*W+1, C]


# ---------------------------------------------------------------------------

class TransLayer(nn.Module):
    """Pre-norm Transformer block: x = x + NystromAttention(LN(x))."""

    def __init__(
        self,
        dim:           int   = 512,
        num_heads:     int   = 8,
        num_landmarks: int   = 256,
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            num_heads=num_heads,
            num_landmarks=num_landmarks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


# ---------------------------------------------------------------------------

class TransMIL(nn.Module):
    """
    TransMIL — Transformer-based Correlated MIL (Shao et al., NeurIPS 2021).

    Architecture:
        1. Linear projection  [N, input_dim] → [N, hidden_dim]
        2. Pad to nearest perfect square and add class token
        3. TransLayer-1 (NystromAttention)
        4. PPEG (spatial position encoding via multi-scale depthwise convs)
        5. TransLayer-2 (NystromAttention)
        6. LayerNorm → extract class token → Linear → logits

    Interface matches TrainerABMIL: forward returns (logits [n_classes], attn [N]).
    Attention weights are uniform (TransMIL does not produce per-patch scores).

    Args:
        input_dim     : patch embedding dimension (1536 for UNI2/GigaPath)
        hidden_dim    : internal transformer dimension (must be divisible by num_heads)
        n_classes     : number of output classes
        num_heads     : attention heads in NystromAttention
        num_landmarks : Nystrom landmark count (falls back to full attention when N ≤ this)
        dropout       : dropout applied after each attention projection
    """

    def __init__(
        self,
        input_dim:     int   = 1536,
        hidden_dim:    int   = 512,
        n_classes:     int   = 2,
        num_heads:     int   = 8,
        num_landmarks: int   = 256,
        dropout:       float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.layer1    = TransLayer(hidden_dim, num_heads, num_landmarks, dropout)
        self.pos_layer = PPEG(hidden_dim)
        self.layer2    = TransLayer(hidden_dim, num_heads, num_landmarks, dropout)

        self.norm       = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    # ------------------------------------------------------------------
    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bag: [N, input_dim]
        Returns:
            logits: [n_classes]
            attn:   [N]  — uniform placeholder (TransMIL has no per-patch attention)
        """
        N = bag.shape[0]

        h = self.fc(bag)  # [N, hidden_dim]

        # Pad sequence to nearest perfect square so PPEG can reshape to [H, W]
        H = math.ceil(math.sqrt(N))
        W = H
        pad_n = H * W - N
        if pad_n > 0:
            # Repeat leading patches rather than zero-padding to avoid conv artefacts
            h = torch.cat([h, h[:pad_n]], dim=0)   # [H*W, hidden_dim]

        h = h.unsqueeze(0)                                         # [1, H*W, hidden_dim]

        cls = self.cls_token.expand(1, -1, -1)                     # [1, 1, hidden_dim]
        h   = torch.cat([cls, h], dim=1)                           # [1, H*W+1, hidden_dim]

        h = self.layer1(h)           # [1, H*W+1, hidden_dim]
        h = self.pos_layer(h, H, W)  # [1, H*W+1, hidden_dim]
        h = self.layer2(h)           # [1, H*W+1, hidden_dim]

        h      = self.norm(h)
        cls_out = h[:, 0]            # [1, hidden_dim]

        logits = self.classifier(cls_out).squeeze(0)               # [n_classes]
        attn   = torch.ones(N, device=bag.device) / N             # [N] uniform

        return logits, attn
