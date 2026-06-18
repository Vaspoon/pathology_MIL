"""
src/training/trainer_contrastive.py

TrainerContrastiveMultiModalABMIL: extends TrainerMultiModalABMIL with
  1. Simultaneous Modality Dropout (SMD) loss — Eq. 1 of the paper
  2. Sigmoid-based three-way contrastive fusion loss — Eq. 2 of the paper

Reference: "Learning Contrastive Multimodal Fusion with Improved Modality Dropout
for Disease Detection and Prediction" (https://arxiv.org/abs/2509.18284)

Batch-size challenge: MIL uses batch_size=1 (one slide per step). Contrastive
losses require multiple samples to be meaningful. This is handled via a circular
memory bank that accumulates detached z representations across `contrastive_batch_size`
consecutive slides. Gradients from the contrastive loss flow only through the
projection heads (proj_he, proj_ihc, proj_fused, log_temp, bias). The attention
aggregators receive gradients exclusively from the SMD classification loss.

Total loss: L = L_smd + λ_con * L_con
  L_smd = CE(y, logits_full) + λ_smd * [CE(y, logits_he) + CE(y, logits_ihc)]
  L_con = [L^con(z_he, z_ihc) + L^con(z_he, z_fused) + L^con(z_ihc, z_fused)] / 3
"""

import torch
import torch.nn.functional as F

from .trainer import TrainerMultiModalABMIL


class TrainerContrastiveMultiModalABMIL(TrainerMultiModalABMIL):
    """
    Args:
        lambda_smd            : weight for unimodal terms in SMD loss (default 1.0)
        lambda_con            : weight balancing contrastive vs SMD loss (default 0.1)
        contrastive_batch_size: slides to accumulate before computing contrastive
                                loss (default 16; must be ≥ 2)
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        writer=None,
        early_stopping_patience: int   = 10,
        monitor:                 str   = "val_loss",
        early_stopping:          bool  = True,
        lambda_smd:              float = 1.0,
        lambda_con:              float = 0.1,
        contrastive_batch_size:  int   = 16,
        class_weights                  = None,
    ):
        super().__init__(
            model, optimizer, device, writer,
            early_stopping_patience=early_stopping_patience,
            monitor=monitor,
            early_stopping=early_stopping,
            class_weights=class_weights,
        )
        self.lambda_smd             = lambda_smd
        self.lambda_con             = lambda_con
        self.contrastive_batch_size = max(contrastive_batch_size, 2)
        self._reset_memory()

    # ------------------------------------------------------------------

    def _reset_memory(self):
        self._memory: dict = {"z_he": [], "z_ihc": [], "z_fused": [], "y": []}

    @staticmethod
    def _sigmoid_contrastive_loss(
        z_i:      torch.Tensor,   # [B, proj_dim] — L2-normalised
        z_j:      torch.Tensor,   # [B, proj_dim] — L2-normalised
        labels:   torch.Tensor,   # [B] integer class labels
        log_temp: torch.Tensor,   # scalar — learnable
        bias:     torch.Tensor,   # scalar — learnable
    ) -> torch.Tensor:
        """
        Sigmoid-based supervised contrastive loss, Eq. 2 of the paper.

            L = -mean_{u,v} log σ( a(u,v) * (t * z_i_u · z_j_v - b) )

        where a(u,v) = +1 for same-class pairs, -1 for different-class pairs,
        and t = exp(log_temp) is a learnable temperature.

        Property: L(z_i, z_j) == L(z_j, z_i) — no need to compute both directions.
        """
        t   = log_temp.exp().clamp(min=1e-2, max=100.0)
        sim = z_i @ z_j.T                                            # [B, B]

        same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))    # [B, B] bool
        a = same_label.float() * 2.0 - 1.0                          # +1 / -1

        logits = a * (t * sim - bias)                                # [B, B]
        return -F.logsigmoid(logits).mean()

    def _contrastive_loss_from_memory(self) -> torch.Tensor:
        """
        Project and L2-normalise the banked representations, then compute the
        three pairwise contrastive losses: (he,ihc), (he,fused), (ihc,fused).
        Gradients flow through the projectors but NOT through the banked z's.
        """
        Z_he    = torch.cat(self._memory["z_he"],    dim=0)          # [B, hidden]
        Z_ihc   = torch.cat(self._memory["z_ihc"],   dim=0)
        Z_fused = torch.cat(self._memory["z_fused"], dim=0)
        Y       = torch.stack(self._memory["y"]).long()              # [B]

        P_he    = F.normalize(self.model.proj_he(Z_he),      dim=-1) # [B, proj]
        P_ihc   = F.normalize(self.model.proj_ihc(Z_ihc),    dim=-1)
        P_fused = F.normalize(self.model.proj_fused(Z_fused), dim=-1)

        L = (
            self._sigmoid_contrastive_loss(P_he, P_ihc,   Y, self.model.log_temp, self.model.bias)
            + self._sigmoid_contrastive_loss(P_he, P_fused, Y, self.model.log_temp, self.model.bias)
            + self._sigmoid_contrastive_loss(P_ihc, P_fused, Y, self.model.log_temp, self.model.bias)
        ) / 3.0
        return L

    # ------------------------------------------------------------------

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_smd = 0.0
        total_con = 0.0
        n_con     = 0

        for step, (hes, ihc, y) in enumerate(loader):
            hes = hes[0].to(self.device)
            ihc = ihc[0].to(self.device)
            y   = y.to(self.device).squeeze(0)

            self.optimizer.zero_grad()

            # ── Forward (training mode returns 7-tuple) ───────────────────────
            logits_full, logits_he, logits_ihc, z_he, z_ihc, z_fused, _ = (
                self.model(hes, ihc)
            )

            # ── SMD loss (Eq. 1) ──────────────────────────────────────────────
            y_long   = y.long().unsqueeze(0)
            loss_smd = (
                self.criterion(logits_full.unsqueeze(0), y_long)
                + self.lambda_smd * (
                    self.criterion(logits_he.unsqueeze(0),  y_long)
                    + self.criterion(logits_ihc.unsqueeze(0), y_long)
                )
            )

            # ── Fill memory bank (detach to avoid storing the full graph) ─────
            self._memory["z_he"].append(z_he.detach())
            self._memory["z_ihc"].append(z_ihc.detach())
            self._memory["z_fused"].append(z_fused.detach())
            self._memory["y"].append(y.detach())

            # Circular buffer: keep the most recent slides
            if len(self._memory["y"]) > self.contrastive_batch_size:
                for k in self._memory:
                    self._memory[k].pop(0)

            # ── Contrastive loss once the bank is full (Eq. 2) ───────────────
            loss_con = torch.tensor(0.0, device=self.device)
            if len(self._memory["y"]) >= self.contrastive_batch_size:
                loss_con = self._contrastive_loss_from_memory()
                n_con    += 1
                total_con += loss_con.item()

            # ── Backward + update ─────────────────────────────────────────────
            loss = loss_smd + self.lambda_con * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_smd += loss_smd.item()

            if self.writer:
                gs = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_smd", loss_smd.item(), gs)
                self.writer.add_scalar("train/loss_con",
                                       loss_con.item() if isinstance(loss_con, torch.Tensor) else 0.0,
                                       gs)

        self._reset_memory()   # clear between epochs

        avg_smd = total_smd / len(loader)
        if self.writer:
            self.writer.add_scalar("train/loss_epoch", avg_smd, epoch)
            if n_con:
                self.writer.add_scalar("train/loss_con_epoch", total_con / n_con, epoch)

        return avg_smd
