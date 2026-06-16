"""
src/models/multimodal_porpoise.py — Late-fusion multimodal MIL (PORPOISE-style)
Chen et al., "Multimodal Co-Attention Transformer for Survival Prediction
in Gigapixel Whole Slide Images", ICCV 2021 / PORPOISE framework.
https://github.com/mahmoodlab/PORPOISE

Architecture (late fusion):
    HES patches [N_hes, input_dim] ──► GatedAttention (ABMIL) ──► z_hes [hidden_dim]  ─┐
                                                                                          ├─ cat ► MLP ► logits
    IHC patches [N_ihc, input_dim] ──► GatedAttention (ABMIL) ──► z_ihc [hidden_dim]  ─┘

Each modality is aggregated independently (no cross-modal attention), then their
bag-level embeddings are concatenated and passed to a shared classifier.
This "late fusion" strategy is the baseline proposed in PORPOISE before adding
co-attention, and is appropriate when both modalities carry complementary signal
but should not influence each other's patch weighting.

Inference with HES only: pass ihc=None — the IHC branch is zero-filled so the
model degrades gracefully without requiring a separate checkpoint.

Interface: forward returns (logits [n_classes], attn_dict) to be compatible
with TrainerMultiModalABMIL (and heatmap utilities).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .abmil import GatedAttention


class MultiModalMILPorpoise(nn.Module):
    """
    Late-fusion multimodal MIL following the PORPOISE framework.

    Args:
        input_dim  : patch embedding dimension (1536 for UNI2/GigaPath)
        hidden_dim : internal dimension of each attention branch
        n_classes  : number of output classes (2 for binary)
        dropout    : dropout applied inside GatedAttention and the classifier
    """

    def __init__(
        self,
        input_dim:  int   = 1536,
        hidden_dim: int   = 256,
        n_classes:  int   = 2,
        dropout:    float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Independent ABMIL (gated attention) branch per modality
        self.hes_attention = GatedAttention(input_dim, hidden_dim, dropout)
        self.ihc_attention = GatedAttention(input_dim, hidden_dim, dropout)

        # Fusion of the two bag embeddings -> hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Harmonized classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)

    # ------------------------------------------------------------------
    def forward(
        self,
        hes: torch.Tensor,
        ihc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hes : [N_hes, input_dim]  — HES patch embeddings
            ihc : [N_ihc, input_dim]  — IHC patch embeddings, or None for
                                        HES-only inference (IHC branch zeroed)
        Returns:
            logits   : [n_classes]
            attn     : {"hes": [N_hes], "ihc": [N_ihc] or None}
        """
        # ── HES branch ──────────────────────────────────────────────────
        hes_attn_w, z_hes = self.hes_attention(hes)  # [N,1], [1, hidden_dim]

        # ── IHC branch (or zero-fill for HES-only inference) ────────────
        if ihc is not None:
            ihc_attn_w, z_ihc = self.ihc_attention(ihc)  # [N,1], [1, hidden_dim]
        else:
            z_ihc      = torch.zeros(1, self.hidden_dim, device=hes.device)
            ihc_attn_w = None

        # ── Late fusion: concatenate bag-level embeddings ────────────────
        cat    = torch.cat([z_hes, z_ihc], dim=-1)    # [1, 2*hidden_dim]
        fused  = self.fusion(cat)                      # [1, hidden_dim]
        logits = self.classifier(fused).squeeze(0)     # [n_classes]

        attn = {
            "hes": hes_attn_w.squeeze(-1),                                    # [N_hes]
            "ihc": ihc_attn_w.squeeze(-1) if ihc_attn_w is not None else None,
        }
        return logits, attn
