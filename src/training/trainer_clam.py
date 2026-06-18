"""
src/training/trainer_clam.py

TrainerCLAM : entraînement de CLAM_SB / CLAM_MB (Lu et al., 2021).

Différences avec TrainerABMIL :
    - La loss combine la perte de classification au niveau bag (cross-entropy)
      et la perte de clustering au niveau instance (renvoyée par le modèle) :

          loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss

    - L'entraînement appelle le modèle avec instance_eval=True et le label
      du bag, pour calculer la perte de clustering instance.
    - L'évaluation appelle le modèle sans instance_eval (comme ABMIL),
      retour (logits, attn, instance_loss=0).

Hérite de TrainerABMIL pour l'early stopping, le checkpointing et fit().
"""

import torch

from training.trainer import TrainerABMIL


class TrainerCLAM(TrainerABMIL):
    """
    Trainer pour CLAM_SB et CLAM_MB.

    Args additionnels par rapport à TrainerABMIL :
        bag_weight : poids de la bag-loss dans la loss totale
                     (1 - bag_weight) est appliqué à la instance-loss.
        loss / focal_gamma / focal_alpha / class_weights : voir TrainerABMIL.
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
        bag_weight:              float = 0.7,
        loss:                    str   = "cross_entropy",
        class_weights                  = None,
        focal_gamma:             float = 2.0,
        focal_alpha                    = None,
    ):
        super().__init__(
            model, optimizer, device, writer,
            early_stopping_patience=early_stopping_patience,
            monitor=monitor,
            early_stopping=early_stopping,
            class_weights=class_weights,
            loss=loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        self.bag_weight = bag_weight

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            hes = batch[0][0].to(self.device)
            y   = batch[-2 if len(batch) == 4 else -1].to(self.device).squeeze(0)

            self.optimizer.zero_grad()

            logits, _, instance_loss = self.model(hes, label=y, instance_eval=True)

            bag_loss = self.criterion(logits.unsqueeze(0), y.long().unsqueeze(0))
            loss = self.bag_weight * bag_loss + (1 - self.bag_weight) * instance_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if self.writer:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_step", loss.item(), global_step)
                self.writer.add_scalar("train/bag_loss_step", bag_loss.item(), global_step)
                self.writer.add_scalar("train/instance_loss_step", instance_loss.item(), global_step)

        avg_loss = total_loss / len(loader)
        if self.writer:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, loader, epoch: int) -> dict:
        self.model.eval()
        total_loss  = 0.0
        all_probs   = []
        all_labels  = []

        for batch in loader:
            hes = batch[0][0].to(self.device)
            y   = batch[-2 if len(batch) == 4 else -1].to(self.device).squeeze(0)

            logits, _, _ = self.model(hes)

            loss  = self.criterion(logits.unsqueeze(0), y.long().unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)[1].item()

            total_loss += loss.item()
            all_probs.append(probs)
            all_labels.append(int(y.item()))

        avg_loss = total_loss / len(loader)

        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            preds = [1 if p > 0.5 else 0 for p in all_probs]
            auc   = roc_auc_score(all_labels, all_probs)
            acc   = accuracy_score(all_labels, preds)
        except Exception:
            auc = float("nan")
            acc = float("nan")

        metrics = {"val_loss": avg_loss, "val_auc": auc, "val_acc": acc}

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k.replace('val_', '')}", v, epoch)

        return metrics
