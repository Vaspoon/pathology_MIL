"""
train_distillation.py — Knowledge Distillation from multimodal teacher to unimodal student.

The teacher (multimodal PORPOISE, HE+IHC) is frozen. The student (any unimodal
architecture from the `model` config group) learns from HE-only inputs,
supervised by a mix of:
    - Hard label loss   : cross-entropy / focal loss on ground truth (cfg.training.loss)
    - Soft logit loss   : KL divergence on teacher soft logits (Hinton et al., 2015)
    - Feature loss      : MSE between student and teacher fused bag embedding (z_fused)

Runs all requested distillation modes (logits, features, both) in a single execution.

Driven by Hydra like train_cv.py: data paths, seeds/folds and training hyperparameters
are read from configs/config.yaml (cfg.data / cfg.cv / cfg.training). The student
architecture is picked via the `model` config group. Distillation-specific
hyperparameters live under cfg.distillation.

Usage:
    python src/train_distillation.py                                  # student = current `model` default (config.yaml)
    python src/train_distillation.py model=attention_mil
    python src/train_distillation.py model=abmil distillation.alpha=0.7 distillation.temperature=3.0
    python src/train_distillation.py model=abmil distillation.mode=[logits]
    # multirun over several student architectures:
    python src/train_distillation.py -m model=abmil,mil_mean,mil_max,clam_sb
    # point to a teacher trained on a different dataset (e.g. her2contest):
    python src/train_distillation.py distillation.teacher_path=outputs/cv_results_her2contest_he/multimodal_porpoise/seed0_best.pt
"""

import copy
import os
import sys

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets.her2_datasets import DatasetMultiModal
from models.multimodal_porpoise import MultiModalMILPorpoise
from training.losses import build_criterion
from utils.build import MODEL_REGISTRY, build_model

# Only unimodal architectures (forward(hes)) can be distillation students.
STUDENT_REGISTRY = {k: v for k, v in MODEL_REGISTRY.items() if not v["multimodal"]}


def _seeds(cfg):
    """Same convention as train_cv.py: cfg.cv.seeds overrides seed_start/n_seeds."""
    if "seeds" in cfg.cv and cfg.cv.seeds:
        return list(cfg.cv.seeds)
    seed_start = int(cfg.cv.get("seed_start", 0))
    n_seeds    = int(cfg.cv.get("n_seeds", 1))
    return list(range(seed_start, seed_start + n_seeds))


def load_teacher(teacher_path, device, input_dim, hidden_dim):
    model = MultiModalMILPorpoise(input_dim=input_dim, hidden_dim=hidden_dim,
                                   n_classes=2, dropout=0.25)
    ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_teacher_bag_embedding(teacher, hes, ihc):
    """Extract the fused (HE+IHC) bag-level embedding from the teacher."""
    _, z_hes = teacher.hes_attention(hes)
    if ihc is not None:
        _, z_ihc = teacher.ihc_attention(ihc)
    else:
        z_ihc = torch.zeros(1, teacher.hidden_dim, device=hes.device)
    cat = torch.cat([z_hes, z_ihc], dim=-1)
    return teacher.fusion(cat)  # [1, hidden_dim]


def get_student_bag_embedding(student, student_name, hes):
    """Extract the bag-level embedding from the student (before classifier)."""
    if student_name == "abmil":
        _, z = student.attention(hes)
        return z
    elif student_name == "attention_mil":
        H = student.feature_extractor(hes)
        A = torch.softmax(student.attention(H), dim=0)
        return torch.sum(A * H, dim=0, keepdim=True)
    elif student_name == "clam_sb":
        h = student.fc(hes)
        A = torch.transpose(student.attention_net(h), 1, 0)
        A = torch.softmax(A, dim=1)
        return torch.mm(A, h)
    elif student_name == "mil_mean":
        h = student.encoder(hes)
        return h.mean(dim=0, keepdim=True)
    elif student_name == "mil_max":
        h = student.encoder(hes)
        return h.max(dim=0, keepdim=True).values
    return None  # TransMIL: no clean bag embedding


def train_one_fold(teacher, student, student_name, train_loader, val_loader,
                    test_loader, mode, seed, fold, *, device, epochs, lr, patience,
                    hidden_dim, alpha, temperature, feature_weight, ce_criterion):
    params = list(student.parameters())
    feature_proj = None
    if mode in ("features", "both"):
        feature_proj = nn.Linear(hidden_dim, hidden_dim).to(device)
        params = params + list(feature_proj.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        student.train()
        if feature_proj:
            feature_proj.train()

        for hes, ihc, y in train_loader:
            hes = hes[0].to(device)
            ihc = ihc[0].to(device)
            y = y.to(device).squeeze(0)

            optimizer.zero_grad()

            student_out = student(hes)
            student_logits = student_out[0] if isinstance(student_out, tuple) else student_out
            hard_loss = ce_criterion(student_logits.unsqueeze(0), y.long().unsqueeze(0))

            with torch.no_grad():
                teacher_logits, _ = teacher(hes, ihc)

            distill_loss = torch.tensor(0.0, device=device)

            if mode in ("logits", "both"):
                T = temperature
                student_log_soft = F.log_softmax(student_logits / T, dim=-1)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)
                kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)
                distill_loss = distill_loss + kl_loss

            if mode in ("features", "both"):
                with torch.no_grad():
                    teacher_feat = get_teacher_bag_embedding(teacher, hes, ihc)
                student_feat = get_student_bag_embedding(student, student_name, hes)
                if student_feat is not None and teacher_feat is not None:
                    projected = feature_proj(student_feat)
                    feat_loss = F.mse_loss(projected, teacher_feat.detach())
                    weight = feature_weight if mode == "both" else 1.0
                    distill_loss = distill_loss + weight * feat_loss

            loss = (1 - alpha) * hard_loss + alpha * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hes, ihc, y in val_loader:
                hes = hes[0].to(device)
                y = y.to(device).squeeze(0)
                out = student(hes)
                logits = out[0] if isinstance(out, tuple) else out
                val_loss += ce_criterion(logits.unsqueeze(0), y.long().unsqueeze(0)).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    # Collect predictions on val and test
    results = []
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        student.eval()
        with torch.no_grad():
            for hes, ihc, y in loader:
                hes = hes[0].to(device)
                y = y.to(device).squeeze(0)
                out = student(hes)
                logits = out[0] if isinstance(out, tuple) else out
                prob = torch.softmax(logits, dim=-1)[1].item()
                pred = 1 if prob >= 0.5 else 0
                results.append({
                    "seed": seed, "fold": fold, "split": split_name,
                    "y_true": int(y.item()), "y_prob": prob, "y_pred": pred,
                })

    return results


def compute_and_print_metrics(df, seeds, label):
    test_df = df[df["split"] == "test"]
    aucs, accs, f1s, mccs = [], [], [], []
    for seed in seeds:
        ds = test_df[test_df["seed"] == seed]
        if len(ds) == 0:
            continue
        aucs.append(roc_auc_score(ds["y_true"], ds["y_prob"]))
        accs.append(accuracy_score(ds["y_true"], ds["y_pred"]))
        f1s.append(f1_score(ds["y_true"], ds["y_pred"]))
        mccs.append(matthews_corrcoef(ds["y_true"], ds["y_pred"]))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Metric':<12} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*35}")
    for name, vals in [("AUC", aucs), ("Accuracy", accs), ("F1", f1s), ("MCC", mccs)]:
        print(f"  {name:<12} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")

    return {"auc": (np.mean(aucs), np.std(aucs)),
            "acc": (np.mean(accs), np.std(accs)),
            "f1": (np.mean(f1s), np.std(f1s)),
            "mcc": (np.mean(mccs), np.std(mccs))}


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    student_name = cfg.model.name
    if student_name not in STUDENT_REGISTRY:
        raise ValueError(
            f"model={student_name} is multimodal, it cannot be a distillation student "
            f"(needs forward(hes) only). Available students: {list(STUDENT_REGISTRY)}"
        )

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    train_csv = pd.read_csv(cfg.data.train_csv)
    val_csv   = pd.read_csv(cfg.data.val_csv)
    test_csv  = pd.read_csv(cfg.data.test_csv)
    full_csv  = pd.concat([train_csv, val_csv], ignore_index=True)

    he_dirs = [cfg.data.embeddings_dir_train, cfg.data.embeddings_dir_val, cfg.data.embeddings_dir_test]
    ihc_dirs = [cfg.data.embeddings_dir_ihc_train, cfg.data.embeddings_dir_ihc_val, cfg.data.embeddings_dir_ihc_test]

    test_ds = DatasetMultiModal(test_csv, he_dirs, ihc_dirs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    dcfg = cfg.distillation
    hidden_dim = cfg.model.hidden_dim
    input_dim  = cfg.model.input_dim

    print(f"Loading teacher from {dcfg.teacher_path}")
    teacher = load_teacher(dcfg.teacher_path, device, input_dim, hidden_dim)
    print("  Teacher loaded and frozen.")

    out_dir = Path(dcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds  = _seeds(cfg)
    k_folds = int(cfg.cv.get("k", 4))
    modes  = list(dcfg.mode) if not isinstance(dcfg.mode, str) else [dcfg.mode]
    all_metrics = {}

    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"  DISTILLATION MODE: {mode.upper()}  —  student: {student_name}")
        print(f"{'#'*70}")

        all_results = []

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")

            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            labels = full_csv["label"].values

            for fold, (train_idx, val_idx) in enumerate(skf.split(full_csv, labels)):
                print(f"    Fold {fold}", end=" ", flush=True)

                train_df = full_csv.iloc[train_idx].reset_index(drop=True)
                val_df = full_csv.iloc[val_idx].reset_index(drop=True)

                train_ds = DatasetMultiModal(train_df, he_dirs, ihc_dirs)
                val_ds = DatasetMultiModal(val_df, he_dirs, ihc_dirs)

                train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

                torch.manual_seed(seed)
                np.random.seed(seed)
                student, _, is_multimodal = build_model(cfg.model, device)
                assert not is_multimodal

                ce_criterion = build_criterion(
                    cfg.training.loss, cfg.training.class_weights,
                    cfg.training.focal_gamma, cfg.training.focal_alpha, device,
                )

                fold_results = train_one_fold(
                    teacher, student, student_name,
                    train_loader, val_loader, test_loader,
                    mode, seed, fold,
                    device=device, epochs=cfg.training.epochs, lr=cfg.training.lr,
                    patience=cfg.training.early_stopping_patience, hidden_dim=hidden_dim,
                    alpha=dcfg.alpha, temperature=dcfg.temperature,
                    feature_weight=dcfg.feature_weight, ce_criterion=ce_criterion,
                )
                all_results.extend(fold_results)
                print("done")

        df = pd.DataFrame(all_results)
        df["model"] = f"distill_{student_name}_{mode}"

        csv_path = out_dir / f"distill_{student_name}_{mode}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Predictions saved to {csv_path}")

        metrics = compute_and_print_metrics(df, seeds, f"KD {mode.upper()} — {student_name} from teacher")
        all_metrics[mode] = metrics

    # Final summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: {student_name} — all distillation modes")
    print(f"{'='*80}")
    fmt = "{:<25} {:>18} {:>18} {:>18} {:>18}"
    print(fmt.format("Mode", "AUC", "Accuracy", "F1", "MCC"))
    print("-" * 98)
    for mode in modes:
        m = all_metrics[mode]
        print(fmt.format(
            f"KD {mode}",
            f"{m['auc'][0]:.3f} +/- {m['auc'][1]:.3f}",
            f"{m['acc'][0]:.3f} +/- {m['acc'][1]:.3f}",
            f"{m['f1'][0]:.3f} +/- {m['f1'][1]:.3f}",
            f"{m['mcc'][0]:.3f} +/- {m['mcc'][1]:.3f}",
        ))


if __name__ == "__main__":
    main()
