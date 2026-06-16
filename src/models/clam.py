"""
src/models/clam.py — CLAM (Lu et al., 2021, Nature Biomedical Engineering)
"Data-efficient and weakly supervised computational pathology on whole-slide images"
https://github.com/mahmoodlab/CLAM

Implémente :
    - Attn_Net_Gated     : attention gated (identique à GatedAttention d'ABMIL,
                            mais avec n_classes branches de score possibles)
    - CLAM_SB            : Single-Branch — une seule branche d'attention,
                            classifieur de bag unique
    - CLAM_MB            : Multi-Branch — une branche d'attention par classe,
                            un classifieur de bag par classe

Les deux variantes ajoutent une perte de clustering au niveau instance
(instance-level clustering loss) qui supervise les patches avec la plus forte
et la plus faible attention via des classifieurs binaires "in-class / out-of-class".

forward(bag, label=None, instance_eval=False, return_attention=False)
    Retourne (logits, attn, instance_loss) :
        logits        : [n_classes]
        attn          : [N]            (CLAM_SB) ou [n_classes, N] (CLAM_MB)
        instance_loss : scalaire (0.0 si instance_eval=False ou label=None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Attn_Net_Gated(nn.Module):
    """
    Attention gated (Ilse et al., 2018), avec n_classes branches de score.

        a_i = W^T · (tanh(V·hᵢ) ⊙ σ(U·hᵢ))   pour chaque classe

    n_classes=1  → CLAM_SB (une seule branche d'attention)
    n_classes=C  → CLAM_MB (une branche d'attention par classe)
    """

    def __init__(self, hidden_dim: int = 256, attn_dim: int = 128, dropout: float = 0.25, n_classes: int = 1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(attn_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [N, hidden_dim]
        Returns:
            A : [N, n_classes] — scores d'attention bruts (avant softmax)
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A


def _create_targets(length: int, value: int, device) -> torch.Tensor:
    return torch.full((length,), value, dtype=torch.long, device=device)


class CLAM_SB(nn.Module):
    """
    CLAM Single-Branch.

    Args:
        input_dim     : dimension des embeddings (1536 pour UNI2/GigaPath)
        hidden_dim    : dimension interne après projection
        attn_dim      : dimension interne de l'attention gated
        n_classes     : nombre de classes (2 pour binaire)
        dropout       : dropout dans la projection / attention
        k_sample      : nombre de patches top/bottom utilisés pour la
                        perte de clustering au niveau instance
        subtyping     : si True, applique aussi la perte "out-of-class"
                        aux instances des classes négatives (utile en
                        multi-classes / subtyping)
        instance_loss : "ce" (CrossEntropy) — perte des classifieurs d'instance
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 256,
        attn_dim:   int   = 128,
        n_classes:  int   = 2,
        dropout:    float = 0.25,
        k_sample:   int   = 8,
        subtyping:  bool  = False,
        instance_loss: str = "ce",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample  = k_sample
        self.subtyping = subtyping

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_net = Attn_Net_Gated(hidden_dim, attn_dim, dropout, n_classes=1)
        self.classifiers   = nn.Linear(hidden_dim, n_classes)

        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
        )

        if instance_loss == "ce":
            self.instance_loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown instance_loss '{instance_loss}'")

    # ------------------------------------------------------------------
    # Instance-level clustering (Eq. 2-3 du papier)
    # ------------------------------------------------------------------
    def _inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module):
        """Patches d'une classe présente dans le slide (in-class)."""
        device   = h.device
        k_sample = min(self.k_sample, h.shape[0])

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p     = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n     = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = _create_targets(k_sample, 1, device)
        n_targets = _create_targets(k_sample, 0, device)

        all_targets   = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)

        logits = classifier(all_instances)
        return self.instance_loss_fn(logits, all_targets)

    def _inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module):
        """Patches d'une classe absente du slide (out-of-class, subtyping)."""
        device   = h.device
        k_sample = min(self.k_sample, h.shape[0])

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p     = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = _create_targets(k_sample, 0, device)

        logits = classifier(top_p)
        return self.instance_loss_fn(logits, p_targets)

    def _instance_eval(self, A: torch.Tensor, h: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """A: [1, N] (softmax'd attention pour la branche unique)."""
        device      = h.device
        inst_labels = F.one_hot(label.long().view(-1), num_classes=self.n_classes).squeeze(0)

        total_loss = h.new_zeros(())
        n_terms    = 0
        for i, classifier in enumerate(self.instance_classifiers):
            if inst_labels[i].item() == 1:
                total_loss = total_loss + self._inst_eval(A, h, classifier)
                n_terms += 1
            elif self.subtyping:
                total_loss = total_loss + self._inst_eval_out(A, h, classifier)
                n_terms += 1

        if n_terms > 0 and self.subtyping:
            total_loss = total_loss / n_terms
        return total_loss

    # ------------------------------------------------------------------
    def forward(
        self,
        bag:              torch.Tensor,          # [N, input_dim]
        label:            torch.Tensor = None,   # scalaire (0..n_classes-1), requis si instance_eval=True
        instance_eval:    bool = False,
        return_attention: bool = False,
    ):
        h = self.fc(bag)                          # [N, hidden_dim]

        A = self.attention_net(h)                 # [N, 1]
        A = torch.transpose(A, 1, 0)               # [1, N]
        A = F.softmax(A, dim=1)                    # [1, N]

        instance_loss = h.new_zeros(())
        if instance_eval and label is not None:
            instance_loss = self._instance_eval(A, h, label)

        M      = torch.mm(A, h)                    # [1, hidden_dim]
        logits = self.classifiers(M).squeeze(0)    # [n_classes]
        attn   = A.squeeze(0)                      # [N]

        return logits, attn, instance_loss


class CLAM_MB(CLAM_SB):
    """
    CLAM Multi-Branch — une branche d'attention et un classifieur de bag par classe.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 256,
        attn_dim:   int   = 128,
        n_classes:  int   = 2,
        dropout:    float = 0.25,
        k_sample:   int   = 8,
        subtyping:  bool  = False,
        instance_loss: str = "ce",
    ):
        super().__init__(
            input_dim, hidden_dim, attn_dim, n_classes, dropout, k_sample, subtyping, instance_loss
        )
        # Une branche d'attention par classe
        self.attention_net = Attn_Net_Gated(hidden_dim, attn_dim, dropout, n_classes=n_classes)
        # Un classifieur de bag par classe (sortie scalaire)
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_classes)]
        )

    def _instance_eval(self, A: torch.Tensor, h: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """A: [n_classes, N] (softmax'd attention, une ligne par classe)."""
        inst_labels = F.one_hot(label.long().view(-1), num_classes=self.n_classes).squeeze(0)

        total_loss = h.new_zeros(())
        n_terms    = 0
        for i, classifier in enumerate(self.instance_classifiers):
            A_i = A[i].unsqueeze(0)  # [1, N]
            if inst_labels[i].item() == 1:
                total_loss = total_loss + self._inst_eval(A_i, h, classifier)
                n_terms += 1
            elif self.subtyping:
                total_loss = total_loss + self._inst_eval_out(A_i, h, classifier)
                n_terms += 1

        if n_terms > 0 and self.subtyping:
            total_loss = total_loss / n_terms
        return total_loss

    def forward(
        self,
        bag:              torch.Tensor,
        label:            torch.Tensor = None,
        instance_eval:    bool = False,
        return_attention: bool = False,
    ):
        h = self.fc(bag)                          # [N, hidden_dim]

        A = self.attention_net(h)                 # [N, n_classes]
        A = torch.transpose(A, 1, 0)               # [n_classes, N]
        A = F.softmax(A, dim=1)                    # [n_classes, N]

        instance_loss = h.new_zeros(())
        if instance_eval and label is not None:
            instance_loss = self._instance_eval(A, h, label)

        M = torch.mm(A, h)                         # [n_classes, hidden_dim]
        logits = torch.cat(
            [self.classifiers[c](M[c].unsqueeze(0)) for c in range(self.n_classes)], dim=1
        ).squeeze(0)                               # [n_classes]

        attn = A                                   # [n_classes, N]

        return logits, attn, instance_loss
