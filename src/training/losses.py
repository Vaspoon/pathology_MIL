"""
src/training/losses.py

Focal Loss (Lin et al., ICCV 2017) and criterion factory shared by all trainers.

Reference: "Focal Loss for Dense Object Detection"
           T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár — ICCV 2017
           https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the predicted probability for the true class.

    Args:
        gamma: focusing parameter >= 0. gamma=0 reduces to standard cross-entropy.
        alpha: per-class weight tensor of shape [n_classes], or None.
               When provided, each sample loss is scaled by alpha[target_class].
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : [B, C]   targets : [B] long
        log_p  = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.view(-1, 1)).squeeze(1)   # [B]
        pt     = log_pt.exp()                                       # [B]  == p_t

        loss = -(1 - pt).pow(self.gamma) * log_pt                  # [B]

        if self.alpha is not None:
            loss = self.alpha[targets] * loss

        return loss.mean()


def build_criterion(
    loss_type: str  = "cross_entropy",
    class_weights   = None,
    focal_gamma: float = 2.0,
    focal_alpha     = None,
    device          = "cpu",
) -> nn.Module:
    """
    Return the loss criterion corresponding to the requested configuration.

    Args:
        loss_type    : "cross_entropy" (default) | "focal"
        class_weights: per-class weights for cross-entropy (list[float] | None).
                       Ignored when loss_type == "focal".
        focal_gamma  : gamma focusing parameter for focal loss (default 2.0).
                       Ignored when loss_type == "cross_entropy".
        focal_alpha  : per-class alpha for focal loss (list[float] | None).
                       Ignored when loss_type == "cross_entropy".
        device       : torch device — ensures weight tensors are on the right device.
    """
    if loss_type == "focal":
        alpha = (
            torch.tensor(focal_alpha, dtype=torch.float32)
            if focal_alpha is not None else None
        )
        return FocalLoss(gamma=focal_gamma, alpha=alpha).to(device)

    # default: standard (optionally weighted) cross-entropy
    weight = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None else None
    )
    return nn.CrossEntropyLoss(weight=weight)
