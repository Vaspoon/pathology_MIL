"""
src/training/trainer.py

Trainer
TrainerMultiModal
TrainerABMIL : metrics AUC, early stopping, sauvegarde best checkpoint.
"""

import copy
import os
import torch
import torch.nn.functional as F
from tqdm import trange


class Trainer:
    """Trainer pour mil_max / mil_mean (pooling simple, cross-entropy)."""

    def __init__(self, model, optimizer, device, writer=None):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.writer    = writer

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for step, (hes, y) in enumerate(loader):
            hes = hes[0].to(self.device)
            y   = y.to(self.device).squeeze(0)

            self.optimizer.zero_grad()
            logits = self.model(hes)  # [n_classes]
            loss = F.cross_entropy(logits.unsqueeze(0), y.long().unsqueeze(0))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.writer:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_step", loss.item(), global_step)

        avg_loss = total_loss / len(loader)
        if self.writer:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        return avg_loss

    def eval_epoch(self, loader, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for hes, y in loader:
                hes    = hes[0].to(self.device)
                y      = y.to(self.device).squeeze(0)
                logits = self.model(hes)  # [n_classes]
                loss   = F.cross_entropy(logits.unsqueeze(0), y.long().unsqueeze(0))
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
        return avg_loss

    def fit(self, train_loader, val_loader, cfg, run_dir: str,
            save_checkpoints: bool = True, desc: str = "epochs") -> dict:
        """
        Boucle d'entraînement avec early stopping et checkpointing optionnel.
        Interface identique à TrainerABMIL.fit() pour permettre un appel uniforme.
        """
        early_stopping = bool(cfg.training.get("early_stopping", True))
        patience       = int(cfg.training.get("early_stopping_patience", 10))
        early_stopper  = EarlyStopping(patience=patience, mode="min") if early_stopping else None

        self.monitor        = "val_loss"
        self.mode           = "min"
        self.best_metric    = float("inf")
        self.best_epoch     = None
        self.best_state_dict = None

        history = {"train_loss": [], "val_loss": []}
        pbar = trange(cfg.training.epochs, desc=desc, leave=False)
        for epoch in pbar:
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss   = self.eval_epoch(val_loader, epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

            if val_loss < self.best_metric:
                self.best_metric     = val_loss
                self.best_epoch      = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                if save_checkpoints:
                    os.makedirs(run_dir, exist_ok=True)
                    path = os.path.join(run_dir, "best.pt")
                    torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                                "val_loss": val_loss}, path)
                    pbar.write(f"  Best model saved -> {path}  (val_loss={val_loss:.4f})")

            if save_checkpoints and (epoch + 1) % cfg.training.get("save_every", 5) == 0:
                os.makedirs(run_dir, exist_ok=True)
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()},
                           os.path.join(run_dir, f"epoch_{epoch+1}.pt"))

            if early_stopper is not None and early_stopper.step(val_loss):
                pbar.write(
                    f"Early stopping déclenché à l'epoch {epoch} "
                    f"(patience={patience}, best val_loss={early_stopper.best:.4f})"
                )
                break

        if save_checkpoints:
            os.makedirs(run_dir, exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()},
                       os.path.join(run_dir, "last.pt"))

        return history


class TrainerMultiModal:

    def __init__(self, model, optimizer, device, writer=None):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.writer    = writer

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for step, (hes, ihc, y) in enumerate(loader):
            hes = hes[0].to(self.device)
            ihc = ihc[0].to(self.device)
            y   = y.to(self.device)

            self.optimizer.zero_grad()
            pred, _ = self.model(hes, ihc)   # ihc fourni à l'entraînement
            loss     = F.cross_entropy(pred.unsqueeze(0), y.long())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.writer is not None:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_step", loss.item(), global_step)

        avg_loss = total_loss / len(loader)
        if self.writer is not None:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        return avg_loss

    def eval_epoch(self, loader, epoch):
        """
        Évaluation HES seule (ihc=None).
        FIX : l'original appelait self.model(hes) sans ihc → AttributeError.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for hes, ihc, y in loader:
                hes  = hes[0].to(self.device)
                y    = y.to(self.device)
                # ihc=None → inférence HES seule (zero-filling dans le modèle)
                pred, _ = self.model(hes, ihc=None)
                loss     = F.cross_entropy(pred.unsqueeze(0), y.long())
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
        return avg_loss



class EarlyStopping:
    """Arrête l'entraînement si la métrique surveillée ne s'améliore plus."""

    def __init__(self, patience: int = 10, mode: str = "min", delta: float = 1e-5):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.triggered = False

    def step(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.delta if self.mode == "min"
            else metric > self.best + self.delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class TrainerABMIL:
    """
    Trainer pour ABMIL (et AttentionMIL_Papagoras).

    Nouveautés par rapport à Trainer existant :
        - Métriques : AUC + accuracy en plus de la loss
        - Early stopping sur val_loss ou val_auc
        - Sauvegarde automatique du meilleur modèle (best.pt)
        - Gradient clipping pour la stabilité
        - TensorBoard : loss + AUC + accuracy

    Compatible avec tout modèle retournant (logits, attn)
    comme ABMIL et AttentionMIL_Papagoras.
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        writer=None,
        early_stopping_patience: int  = 10,
        monitor:                 str  = "val_loss",   # "val_loss" | "val_auc"
        early_stopping:          bool = True,         # désactive l'early stopping si False
    ):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.writer    = writer
        self.monitor   = monitor

        mode = "min" if monitor == "val_loss" else "max"
        self.early_stopping_enabled = early_stopping
        self.early_stopper = EarlyStopping(patience=early_stopping_patience, mode=mode)

        self.best_metric    = float("inf") if mode == "min" else float("-inf")
        self.mode           = mode
        self.best_state_dict = None
        self.best_epoch      = None

    def _is_better(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            # Support HER2Dataset (hes, label) et HER2DatasetWithCoords (hes, coords, label, id)
            hes = batch[0][0].to(self.device)
            y   = batch[-2 if len(batch) == 4 else -1].to(self.device).squeeze(0)

            self.optimizer.zero_grad()

            logits, _ = self.model(hes)

            # Gestion binaire (y scalaire) vs multi-classes
            if logits.shape[-1] == 2:
                loss = F.cross_entropy(logits.unsqueeze(0), y.long().unsqueeze(0))
            else:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if self.writer:
                self.writer.add_scalar(
                    "train/loss_step", loss.item(), epoch * len(loader) + step
                )

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

            logits, _ = self.model(hes)

            if logits.shape[-1] == 2:
                loss  = F.cross_entropy(logits.unsqueeze(0), y.long().unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[1].item()   # prob classe 1
            else:
                loss  = F.binary_cross_entropy_with_logits(logits.squeeze(), y)
                probs = torch.sigmoid(logits).squeeze().item()

            total_loss += loss.item()
            all_probs.append(probs)
            all_labels.append(int(y.item()))

        avg_loss = total_loss / len(loader)

        # AUC (sklearn requis — dégradé gracieusement si absent)
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            preds   = [1 if p > 0.5 else 0 for p in all_probs]
            auc     = roc_auc_score(all_labels, all_probs)
            acc     = accuracy_score(all_labels, preds)
        except Exception:
            auc = float("nan")
            acc = float("nan")

        metrics = {"val_loss": avg_loss, "val_auc": auc, "val_acc": acc}

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k.replace('val_', '')}", v, epoch)

        return metrics

    def save_checkpoint(self, run_dir: str, tag: str, epoch: int, metrics: dict):
        """Sauvegarde un checkpoint avec la config du modèle."""
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"{tag}.pt")
        torch.save({
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "metrics":          metrics,
        }, path)
        return path

    def fit(self, train_loader, val_loader, cfg, run_dir: str, save_checkpoints: bool = True, desc: str = "epochs") -> dict:
        """
        Boucle d'entraînement complète avec early stopping et checkpointing.

        Args:
            save_checkpoints: si False, n'écrit aucun fichier .pt sur disque
                (utile pour les boucles cross-validation / multi-seeds).
            desc: préfixe affiché sur la barre de progression tqdm (ex: nom du
                fold/seed en cours).

        Retourne l'historique des métriques.
        Usage :
            trainer = TrainerABMIL(model, optimizer, device, writer)
            history = trainer.fit(train_loader, val_loader, cfg, run_dir)
        """
        history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

        pbar = trange(cfg.training.epochs, desc=desc, leave=False)
        for epoch in pbar:
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.eval_epoch(val_loader, epoch)

            history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                history[k].append(v)

            monitor_val = val_metrics[self.monitor]

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_metrics['val_loss']:.4f}",
                val_auc=f"{val_metrics['val_auc']:.4f}",
                val_acc=f"{val_metrics['val_acc']:.4f}",
            )

            # Sauvegarde best model
            if self._is_better(monitor_val):
                self.best_metric     = monitor_val
                self.best_epoch      = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                if save_checkpoints:
                    path = self.save_checkpoint(run_dir, "best", epoch, val_metrics)
                    pbar.write(f"  Best model sauvegardé -> {path}  ({self.monitor}={monitor_val:.4f})")

            # Checkpoint périodique
            if save_checkpoints and (epoch + 1) % cfg.training.get("save_every", 5) == 0:
                self.save_checkpoint(run_dir, f"epoch_{epoch+1}", epoch, val_metrics)

            # Early stopping
            if self.early_stopping_enabled and self.early_stopper.step(monitor_val):
                pbar.write(
                    f"Early stopping déclenché à l'epoch {epoch} "
                    f"(patience={self.early_stopper.patience}, "
                    f"best {self.monitor}={self.early_stopper.best:.4f})"
                )
                break

        # Checkpoint final
        if save_checkpoints:
            self.save_checkpoint(run_dir, "last", epoch, val_metrics)
        return history


class TrainerMultiModalABMIL(TrainerABMIL):
    """
    Trainer pour les modèles multimodaux retournant (logits, attn_dict)
    avec forward(hes, ihc) — e.g. MultiModalMILPorpoise, MultiModalMILGated.

    Hérite de TrainerABMIL (early stopping, AUC, checkpointing, fit()) et
    surcharge uniquement les boucles d'epoch pour gérer le batch (hes, ihc, y).
    L'évaluation utilise les deux modalités (IHC disponible en validation).
    """

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, (hes, ihc, y) in enumerate(loader):
            hes = hes[0].to(self.device)
            ihc = ihc[0].to(self.device)
            y   = y.to(self.device).squeeze(0)

            self.optimizer.zero_grad()
            logits, _ = self.model(hes, ihc)

            if logits.shape[-1] == 2:
                loss = F.cross_entropy(logits.unsqueeze(0), y.long().unsqueeze(0))
            else:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            if self.writer:
                self.writer.add_scalar(
                    "train/loss_step", loss.item(), epoch * len(loader) + step
                )

        avg_loss = total_loss / len(loader)
        if self.writer:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, loader, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_probs  = []
        all_labels = []

        for hes, ihc, y in loader:
            hes = hes[0].to(self.device)
            ihc = ihc[0].to(self.device)
            y   = y.to(self.device).squeeze(0)

            logits, _ = self.model(hes, ihc)

            if logits.shape[-1] == 2:
                loss  = F.cross_entropy(logits.unsqueeze(0), y.long().unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[1].item()
            else:
                loss  = F.binary_cross_entropy_with_logits(logits.squeeze(), y)
                probs = torch.sigmoid(logits).squeeze().item()

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
