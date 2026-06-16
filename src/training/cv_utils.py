"""
src/training/cv_utils.py

Helpers pour l'entraînement multi-seeds avec k-fold ou Leave-One-Out
cross-validation (utilisés par train_cv.py) :

    set_seed              : fixe les graines aléatoires (torch / numpy / random)
    get_cv_splits         : génère les indices (train, val) pour kfold ou loo
    collect_predictions   : fait l'inférence sur un loader et renvoie (probs, labels)
    train_one_fold        : entraîne un modèle pour un fold (avec/sans early stopping)
"""

import copy
import random

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from tqdm import trange

from training.trainer import EarlyStopping
from utils.build import ABMIL_FAMILY


def set_seed(seed: int) -> None:
    """Fixe les graines aléatoires pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cv_splits(labels, method: str, k: int, seed: int):
    """
    Génère une liste de (train_idx, val_idx) (np.ndarray d'indices positionnels).

    Args:
        labels : array-like des labels (utilisé pour la stratification kfold)
        method : "kfold" | "loo"
        k      : nombre de folds (ignoré si method == "loo")
        seed   : graine pour le shuffle du kfold
    """
    n = len(labels)
    indices = np.arange(n)

    if method == "loo":
        splitter = LeaveOneOut()
        return list(splitter.split(indices))

    if method == "kfold":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(splitter.split(indices, labels))

    raise ValueError(f"Unknown cv method '{method}'. Expected 'kfold' or 'loo'.")


@torch.no_grad()
def collect_predictions(model, loader, multimodal: bool, device):
    """
    Inférence sur `loader` (batch_size=1, shuffle=False).

    Retourne deux listes alignées avec l'ordre d'itération du loader :
        probs  : probabilité prédite pour la classe positive (1)
        labels : label réel (0/1)
    """
    model.eval()
    probs, labels = [], []

    for batch in loader:
        if multimodal:
            hes, ihc, y = batch
            hes = hes[0].to(device)
            ihc = ihc[0].to(device)
            out = model(hes, ihc)
        else:
            hes = batch[0][0].to(device)
            y   = batch[-1]
            out = model(hes)

        logits = out[0] if isinstance(out, tuple) else out

        if logits.dim() == 0:
            p = torch.sigmoid(logits).item()
        elif logits.shape[-1] == 2:
            p = torch.softmax(logits, dim=-1)[1].item()
        else:
            p = torch.sigmoid(logits.reshape(-1)[0]).item()

        probs.append(p)
        labels.append(int(y.reshape(-1)[0].item()))

    return probs, labels


def train_one_fold(trainer, TrainerClass, train_loader, val_loader, cfg, run_dir, desc: str = "epochs"):
    """
    Entraîne `trainer` pour un fold/seed donné.

    - Pour les trainers de la famille ABMIL (TrainerABMIL, TrainerCLAM,
      TrainerMultiModalABMIL, TrainerContrastiveMultiModalABMIL) : utilise
      fit() avec early stopping paramétrable (cfg.training.early_stopping)
      et sans écriture de checkpoints (cfg.cv.save_checkpoints).
    - Pour le Trainer basique (mil_max / mil_mean) : boucle manuelle avec
      early stopping optionnel sur val_loss.

    `desc` est affiché sur la barre de progression tqdm des epochs (identifie
    le fold/seed en cours).
    """
    if issubclass(TrainerClass, ABMIL_FAMILY):
        save_checkpoints = bool(cfg.cv.get("save_checkpoints", False))
        return trainer.fit(train_loader, val_loader, cfg, run_dir, save_checkpoints=save_checkpoints, desc=desc)

    # Basic Trainer (mil_max / mil_mean) — boucle manuelle
    early_stopping = bool(cfg.training.get("early_stopping", True))
    patience        = int(cfg.training.get("early_stopping_patience", 10))
    early_stopper   = EarlyStopping(patience=patience, mode="min") if early_stopping else None

    # Attributs ajoutés dynamiquement pour exposer une interface uniforme avec
    # TrainerABMIL.fit() (utilisée par train_cv.py pour sauvegarder le meilleur
    # modèle par seed, peu importe la classe de trainer).
    trainer.monitor          = "val_loss"
    trainer.mode             = "min"
    trainer.best_metric      = float("inf")
    trainer.best_epoch       = None
    trainer.best_state_dict  = None

    history = {"train_loss": [], "val_loss": []}
    pbar = trange(cfg.training.epochs, desc=desc, leave=False)
    for epoch in pbar:
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss   = trainer.eval_epoch(val_loader, epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < trainer.best_metric:
            trainer.best_metric     = val_loss
            trainer.best_epoch      = epoch
            trainer.best_state_dict = copy.deepcopy(trainer.model.state_dict())

        if early_stopper is not None and early_stopper.step(val_loss):
            break

    return history
