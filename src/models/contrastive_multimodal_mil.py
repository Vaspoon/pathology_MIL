"""
src/models/contrastive_multimodal_mil.py

Multimodal MIL with Simultaneous Modality Dropout (SMD) and Contrastive Fusion.
Adapted from: "Learning Contrastive Multimodal Fusion with Improved Modality Dropout
for Disease Detection and Prediction" (https://arxiv.org/abs/2509.18284)

Inputs: HE patch embeddings [N_he, input_dim], IHC patch embeddings [N_ihc, input_dim]

Key innovations from the paper:
  1. Learnable tokens E_he / E_ihc replacing a missing modality at fusion input
     (replaces zero-padding used in earlier work)
  2. Simultaneous Modality Dropout: jointly supervises full, HE-only, IHC-only
     predictions in a single forward pass (no random sampling needed)
  3. Three-way sigmoid contrastive loss aligning z_he, z_ihc, z_fused

Return contract:
  - eval mode  → (logits [n_classes], attn_dict)           compatible with TrainerMultiModalABMIL.eval_epoch
  - train mode → (logits_full, logits_he, logits_ihc,
                  z_he, z_ihc, z_fused, attn_dict)          handled by TrainerContrastiveMultiModalABMIL
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from .abmil import GatedAttention


class ContrastiveMultiModalMIL(nn.Module):
    """
    Args:
        input_dim  : patch embedding dimension (1536 for UNI2 / GigaPath)
        hidden_dim : internal dimension for attention and fusion
        n_classes  : number of output classes
        proj_dim   : projection head output dimension for contrastive loss
        dropout    : dropout rate inside GatedAttention and classifier
    """

    def __init__(
        self,
        input_dim:  int   = 1536,
        hidden_dim: int   = 256,
        n_classes:  int   = 2,
        proj_dim:   int   = 128,
        dropout:    float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Independent MIL aggregator per modality
        self.he_attn  = GatedAttention(input_dim, hidden_dim, dropout)
        self.ihc_attn = GatedAttention(input_dim, hidden_dim, dropout)

        # Learnable missing-modality tokens in aggregated space [1, hidden_dim].
        # Small random init prevents vanishing gradients at the start of training.
        self.he_token  = nn.Parameter(torch.empty(1, hidden_dim))
        self.ihc_token = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.he_token,  std=0.02)
        nn.init.normal_(self.ihc_token, std=0.02)

        # Single-layer MLP fusion (paper: "lightweight single-layer MLP")
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared classifier — used for all three SMD predictions
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # Per-modality projection heads for contrastive alignment
        self.proj_he    = nn.Linear(hidden_dim, proj_dim)
        self.proj_ihc   = nn.Linear(hidden_dim, proj_dim)
        self.proj_fused = nn.Linear(hidden_dim, proj_dim)

        # Learnable temperature (log scale) and bias for sigmoid contrastive loss (Eq. 2)
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.bias     = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------

    def _fuse_and_classify(
        self,
        z_he:  torch.Tensor,  # [1, hidden_dim]
        z_ihc: torch.Tensor,  # [1, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate two aggregated representations, fuse, and classify."""
        z_fused = self.fusion(torch.cat([z_he, z_ihc], dim=-1))  # [1, hidden_dim]
        logits  = self.classifier(z_fused).squeeze(0)            # [n_classes]
        return z_fused, logits

    # ------------------------------------------------------------------

    def forward(
        self,
        hes: torch.Tensor,
        ihc: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Dict],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor, Dict],
    ]:
        """
        Args:
            hes : [N_he,  input_dim]  HE patch embeddings
            ihc : [N_ihc, input_dim]  IHC patch embeddings, or None for HE-only inference

        Returns (eval):
            logits  : [n_classes]
            attn    : {"hes": [N_he], "ihc": [N_ihc] or None}

        Returns (train):
            logits_full  : [n_classes]  multimodal prediction
            logits_he    : [n_classes]  HE-only prediction (IHC replaced by learned token)
            logits_ihc   : [n_classes]  IHC-only prediction (HE replaced by learned token)
            z_he         : [1, hidden_dim]
            z_ihc        : [1, hidden_dim]
            z_fused      : [1, hidden_dim]
            attn         : {"hes": [N_he], "ihc": [N_ihc] or None}
        """
        # ── Aggregate patches per modality ────────────────────────────────────
        he_attn_w, z_he = self.he_attn(hes)          # [N,1], [1, hidden]

        if ihc is not None:
            ihc_attn_w, z_ihc = self.ihc_attn(ihc)   # [N,1], [1, hidden]
        else:
            # Inference with HE only: replace IHC with its learnable token
            z_ihc      = self.ihc_token
            ihc_attn_w = None

        # ── Full multimodal prediction ────────────────────────────────────────
        z_fused, logits_full = self._fuse_and_classify(z_he, z_ihc)

        attn = {
            "hes": he_attn_w.squeeze(-1),
            "ihc": ihc_attn_w.squeeze(-1) if ihc_attn_w is not None else None,
        }

        if not self.training:
            return logits_full, attn

        # ── Simultaneous Modality Dropout — unimodal branch predictions ───────
        # HE-only: real z_he + learned IHC token (simulates missing IHC)
        _, logits_he  = self._fuse_and_classify(z_he,         self.ihc_token)
        # IHC-only: learned HE token + real z_ihc (simulates missing HE)
        _, logits_ihc = self._fuse_and_classify(self.he_token, z_ihc)

        return logits_full, logits_he, logits_ihc, z_he, z_ihc, z_fused, attn
