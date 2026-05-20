"""
src/models/abmil.py — ABMIL avec Gated Attention (Ilse et al., 2018)

Complète attention_mil.py existant en ajoutant :
    - GatedAttention  : gate sigmoid + tanh (Eq. 9 du papier)
    - ABMIL           : modèle complet avec tête de classification
"""

import torch
import torch.nn as nn
from typing import Tuple


class GatedAttention(nn.Module):
    """
    Mécanisme d'attention gated (Ilse et al., 2018 — Eq. 9).

        a_i = softmax( W^T · (tanh(V·hᵢ) ⊙ σ(U·hᵢ)) )

    Différence avec AttentionMIL existant :
        - AttentionMIL    : tanh uniquement  → pas de gate
        - GatedAttention  : tanh ⊙ sigmoid  → gate apprise qui filtre le signal
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()

        # Projection commune input → hidden
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Branche valeur (tanh)
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Branche gate (sigmoid)
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Score scalaire par patch
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : [N, input_dim]  — N patches du bag
        Returns:
            A_w : [N, 1]          — poids d'attention (somme = 1)
            z   : [1, hidden_dim] — embedding agrégé du bag
        """
        h   = self.feature_proj(x)                          # [N, hidden_dim]
        A   = self.attention_w(self.attention_V(h) * self.attention_U(h))  # [N, 1]
        A_w = torch.softmax(A, dim=0)                       # [N, 1]
        z   = torch.sum(A_w * h, dim=0, keepdim=True)       # [1, hidden_dim]
        return A_w, z


class ABMIL(nn.Module):
    """
    ABMIL complet — drop-in compatible avec AttentionMIL_Papagoras existant.

    Args:
        input_dim   : dimension des embeddings (1536 pour UNI2/GigaPath)
        hidden_dim  : dimension interne de l'attention
        n_classes   : nombre de classes (2 pour binaire OvR, >2 pour multi-classes)
        dropout     : dropout dans la projection
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 256,
        n_classes:  int   = 2,
        dropout:    float = 0.25,
    ):
        super().__init__()
        self.attention  = GatedAttention(input_dim, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(
        self,
        bag:              torch.Tensor,        # [N, input_dim]
        return_attention: bool = False,
    ):
        """
        Retourne (logits, attn) — même signature que AttentionMIL_Papagoras.

        Args:
            bag              : [N, input_dim]
            return_attention : si True, retourne aussi les poids d'attention

        Returns:
            logits : [n_classes]
            attn   : [N]  (poids softmax, somme = 1)
        """
        A_w, z   = self.attention(bag)           # [N,1], [1, hidden_dim]
        logits   = self.classifier(z).squeeze(0) # [n_classes]
        attn     = A_w.squeeze(-1)               # [N]
        return logits, attn
