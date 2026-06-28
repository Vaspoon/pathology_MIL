"""
train_distillation.py — Knowledge Distillation from multimodal PORPOISE to unimodal student.

The teacher (PORPOISE, HE+IHC) is frozen. The student (any unimodal arch) learns
from HE-only inputs, supervised by a mix of:
    - Hard label loss   : standard cross-entropy on ground truth
    - Soft logit loss   : KL divergence on teacher soft logits (Hinton et al., 2015)
    - Feature loss      : MSE between student and teacher fused bag embedding (z_fused)

Runs all 3 distillation modes (logits, features, both) in a single execution.

Usage:
    python src/train_distillation.py --student abmil
    python src/train_distillation.py --student abmil --seeds 0 1 2 --epochs 10
    python src/train_distillation.py --student attention_mil --alpha 0.7 --temperature 3.0
"""

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets.her2_datasets import H5Dataset, DatasetMultiModal
from models.multimodal_porpoise import MultiModalMILPorpoise
from models.abmil import ABMIL
from models.attention_mil import AttentionMIL_Papagoras
from models.transmil import TransMIL
from models.clam import CLAM_SB

STUDENT_REGISTRY = {
    "abmil":         ABMIL,
    "attention_mil": AttentionMIL_Papagoras,
    "transmil":      TransMIL,
    "clam_sb":       CLAM_SB,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", type=str, default="abmil",
                   choices=list(STUDENT_REGISTRY.keys()))
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight of distillation loss vs hard label loss")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Temperature for soft logits")
    p.add_argument("--feature_weight", type=float, default=1.0,
                   help="Weight of feature loss relative to logit loss (both mode)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--k_folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--input_dim", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--class_weights", type=float, nargs="+", default=[2.367, 0.634],
                   help="Weights for weighted cross-entropy loss")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--teacher_path", type=str,
                   default="D:/pathology_MIL/teacher/chu_multimodal_propoise_teacher.pt")
    p.add_argument("--output_dir", type=str, default="outputs/distillation")
    return p.parse_args()


def load_teacher(teacher_path, device, input_dim=1536, hidden_dim=256):
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


def build_student(name, input_dim, hidden_dim, device, dropout=0.25):
    cls = STUDENT_REGISTRY[name]
    if name == "transmil":
        model = cls(input_dim=input_dim, n_classes=2)
    elif name == "attention_mil":
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=2,
                     dropout_rate=dropout)
    elif name == "clam_sb":
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=2,
                     dropout=dropout)
    else:
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=2,
                     dropout=dropout)
    return model.to(device)


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
        _, z = student.attention(hes)
        return z
    return None  # TransMIL: no clean bag embedding


def train_one_fold(teacher, student, student_name, train_loader, val_loader,
                   test_loader, mode, args, seed, fold):
    device = args.device
    weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=weights)

    feature_proj = None
    params = list(student.parameters())
    if mode in ("features", "both"):
        feature_proj = nn.Linear(args.hidden_dim, args.hidden_dim).to(device)
        params = params + list(feature_proj.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        student.train()
        if feature_proj:
            feature_proj.train()
        total_loss = 0.0

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
                T = args.temperature
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
                    weight = args.feature_weight if mode == "both" else 1.0
                    distill_loss = distill_loss + weight * feat_loss

            loss = (1 - args.alpha) * hard_loss + args.alpha * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

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
            if patience_counter >= args.patience:
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    base = Path("D:/pathology_MIL")
    train_csv = pd.read_csv(base / "data/splits_chu_unbalanced/train.csv")
    val_csv = pd.read_csv(base / "data/splits_chu_unbalanced/val.csv")
    test_csv = pd.read_csv(base / "data/splits_chu_unbalanced/test.csv")
    full_csv = pd.concat([train_csv, val_csv], ignore_index=True)

    he_dirs = [
        str(base / "data/CHU_UNI2_embeds_unbalanced/HE/train"),
        str(base / "data/CHU_UNI2_embeds_unbalanced/HE/val"),
        str(base / "data/CHU_UNI2_embeds_unbalanced/HE/test"),
    ]
    ihc_dirs = [
        str(base / "data/CHU_UNI2_embeds_unbalanced/IHC/train"),
        str(base / "data/CHU_UNI2_embeds_unbalanced/IHC/val"),
        str(base / "data/CHU_UNI2_embeds_unbalanced/IHC/test"),
    ]

    test_ds = DatasetMultiModal(test_csv, he_dirs, ihc_dirs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Load teacher once (same for all seeds)
    print(f"Loading teacher from {args.teacher_path}")
    teacher = load_teacher(args.teacher_path, device, args.input_dim, args.hidden_dim)
    print("  Teacher loaded and frozen.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["logits", "features", "both"]
    all_metrics = {}

    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"  DISTILLATION MODE: {mode.upper()}")
        print(f"{'#'*70}")

        all_results = []

        for seed in args.seeds:
            print(f"\n  --- Seed {seed} ---")

            skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
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
                student = build_student(args.student, args.input_dim,
                                         args.hidden_dim, device, args.dropout)

                fold_results = train_one_fold(
                    teacher, student, args.student,
                    train_loader, val_loader, test_loader,
                    mode, args, seed, fold,
                )
                all_results.extend(fold_results)
                print("done")

        df = pd.DataFrame(all_results)
        df["model"] = f"distill_{args.student}_{mode}"

        csv_path = out_dir / f"distill_{args.student}_{mode}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Predictions saved to {csv_path}")

        metrics = compute_and_print_metrics(df, args.seeds,
                                             f"KD {mode.upper()} — {args.student} from PORPOISE")
        all_metrics[mode] = metrics

    # Final summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: {args.student} — all distillation modes")
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
