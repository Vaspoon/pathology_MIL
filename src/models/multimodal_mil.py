"""
src/models/multimodal_mil.py

MultiModalMIL: late fusion par concatenation.
MultiModalMILGated : fusion par cross-attention HES ↔ IHC.

Cas d'usage visé :
    - Entraînement : HES + IHC disponibles
    - Inférence    : HES seule (ihc=None → zéro-filling automatique)
"""

import torch
import torch.nn as nn
from .attention_mil import AttentionMIL
from .abmil import GatedAttention


# ─────────────────────────────────────────────────────────────
# Modèle existant — inchangé
# ─────────────────────────────────────────────────────────────

class MultiModalMIL(nn.Module):
    """Fusion simple par concaténation (existant — non modifié)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hes_mil = AttentionMIL(input_dim, hidden_dim)
        self.ihc_mil = AttentionMIL(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hes, ihc):
        h1, _ = self.hes_mil(hes)
        h2, _ = self.ihc_mil(ihc)
        fused  = torch.cat([h1, h2], dim=0)
        return self.classifier(fused).squeeze()


# ─────────────────────────────────────────────────────────────
# Nouveau modèle — fusion par cross-attention
# ─────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Bloc de cross-attention entre deux modalités.

    Direction A → B :
        Q = embedding_A  (1, hidden_dim)
        K = V = patches_B  (N_B, hidden_dim)

    L'embedding A s'enrichit en interrogeant les patches B.
    Compatible avec IHC manquante à l'inférence (ihc=None).
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim    = hidden_dim,
            num_heads    = n_heads,
            dropout      = dropout,
            batch_first  = True,
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_emb:   torch.Tensor,   # [1, hidden_dim]  — embedding agrégé
        key_patches: torch.Tensor,   # [N, hidden_dim]  — patches de l'autre modalité
    ) -> torch.Tensor:
        """Retourne [1, hidden_dim] enrichi par cross-attention."""
        q = query_emb.unsqueeze(0)      # [1, 1, hidden_dim]  batch=1
        k = key_patches.unsqueeze(0)    # [1, N, hidden_dim]

        out, _ = self.cross_attn(q, k, k)   # [1, 1, hidden_dim]
        out    = out.squeeze(0)              # [1, hidden_dim]

        # Connexion résiduelle + LayerNorm
        return self.norm(query_emb + self.dropout(out))


class MultiModalMILGated(nn.Module):
    """
    MIL multimodal HES + IHC avec fusion par cross-attention.

    Architecture :
        HES patches → GatedAttention → z_hes  ─┐
                                                 ├─ CrossAttention ─┐
        IHC patches → GatedAttention → z_ihc  ─┘                   ├─ MLP → logit
                                                                     │
        z_hes enrichi + z_ihc enrichi ─────────────────────────────┘

    Inférence HES seule :
        Passez ihc=None → la branche IHC est remplacée par des zéros.
        Le modèle dégrade gracieusement : il perd la supervision IHC
        mais l'encodeur HES a appris des features enrichies pendant l'entraînement.

    Args:
        input_dim   : dimension des embeddings (1536 pour UNI2)
        hidden_dim  : dimension interne partagée
        n_classes   : 2 pour binaire OvR
        n_heads     : têtes de cross-attention (doit diviser hidden_dim)
        dropout     : dropout global
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 256,
        n_classes:  int   = 2,
        n_heads:    int   = 4,
        dropout:    float = 0.25,
    ):
        super().__init__()

        # Encodeurs par modalité (Gated Attention)
        self.hes_attention = GatedAttention(input_dim, hidden_dim, dropout)
        self.ihc_attention = GatedAttention(input_dim, hidden_dim, dropout)

        # Cross-attention bidirectionnelle
        self.hes_reads_ihc = CrossAttentionFusion(hidden_dim, n_heads, dropout)
        self.ihc_reads_hes = CrossAttentionFusion(hidden_dim, n_heads, dropout)

        # Projection intermédiaire des patches pour la cross-attention
        # (les patches bruts → même espace hidden_dim que les embeddings agrégés)
        self.hes_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.ihc_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Tête de classification fusionnée
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        hes: torch.Tensor,                    # [N_hes, input_dim]
        ihc: torch.Tensor | None = None,      # [N_ihc, input_dim] | None
    ):
        """
        Args:
            hes : patches HES  [N_hes, input_dim]
            ihc : patches IHC  [N_ihc, input_dim] ou None (inférence HES seule)

        Returns:
            logits : [n_classes]
            attn   : dict avec 'hes' [N_hes] et 'ihc' [N_ihc] ou None
        """
        # ── Branche HES ──────────────────────────────────────
        hes_attn, z_hes = self.hes_attention(hes)    # [N,1], [1, hidden_dim]
        h_hes = self.hes_proj(hes)                   # [N, hidden_dim]

        # ── Branche IHC (ou zéro-filling si absente) ─────────
        if ihc is not None:
            ihc_attn, z_ihc = self.ihc_attention(ihc)
            h_ihc = self.ihc_proj(ihc)
        else:
            # Inférence HES seule : IHC remplacée par vecteur zéro
            z_ihc    = torch.zeros(1, self.hidden_dim, device=hes.device)
            h_ihc    = torch.zeros(1, self.hidden_dim, device=hes.device)
            ihc_attn = None

        # ── Cross-attention bidirectionnelle ──────────────────
        z_hes_enriched = self.hes_reads_ihc(z_hes, h_ihc)  # HES lit IHC
        z_ihc_enriched = self.ihc_reads_hes(z_ihc, h_hes)  # IHC lit HES

        # ── Fusion + classification ───────────────────────────
        fused  = torch.cat([z_hes_enriched, z_ihc_enriched], dim=-1)  # [1, 2*hidden]
        logits = self.classifier(fused).squeeze(0)                     # [n_classes]

        attn = {
            "hes": hes_attn.squeeze(-1),                  # [N_hes]
            "ihc": ihc_attn.squeeze(-1) if ihc_attn is not None else None,
        }
        return logits, attn
