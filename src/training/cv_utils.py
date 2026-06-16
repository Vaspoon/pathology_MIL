"""
src/training/cv_utils.py

Helpers pour l'entraînement multi-seeds avec k-fold ou Leave-One-Out
cross-validation (utilisés par train_cv.py) :

    set_seed              : fixe les graines aléatoires (torch / numpy / random)
    get_cv_splits         : génère les indices (train, val) pour kfold ou loo
    collect_predictions   : fait l'inférence sur un loader et renvoie (probs, labels)
    train_one_fold        : entraîne un modèle pour un fold (avec/sans early stopping)
"""

import random

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold


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

    Tous les trainers exposent désormais fit() avec la même interface
    (early stopping, best_state_dict, best_epoch, best_metric, monitor, mode).
    """
    save_checkpoints = bool(cfg.cv.get("save_checkpoints", False))
    return trainer.fit(train_loader, val_loader, cfg, run_dir,
                       save_checkpoints=save_checkpoints, desc=desc)
