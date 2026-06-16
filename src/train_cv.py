"""
src/train_cv.py

Entraînement multi-seeds avec k-fold ou Leave-One-Out cross-validation.

Pour chaque seed (cfg.cv.seed_start .. seed_start + n_seeds - 1) et chaque fold
(cfg.cv.method == "kfold" -> StratifiedKFold(k) | "loo" -> LeaveOneOut) :
    1. fusionne train_csv + val_csv en un pool, et génère les folds sur ce pool
    2. (re)construit le modèle (poids aléatoires, dépendants de la seed)
    3. entraîne avec ou sans early stopping (cfg.training.early_stopping)
    4. collecte les prédictions sur le fold de validation, et (si fourni)
       sur un jeu de test indépendant (cfg.data.test_csv)

Toutes les prédictions sont écrites dans :
    - <run_dir>/predictions.csv                       (sortie hydra de ce run)
    - <cfg.cv.results_dir>/<model_name>_predictions.csv  (chemin stable, utilisé
      par les notebooks notebooks/eval_cv_<model>.ipynb)
"""

import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.her2_datasets import H5Dataset, DatasetMultiModal
from training.cv_utils import collect_predictions, get_cv_splits, set_seed, train_one_fold
from utils.build import MODEL_REGISTRY, build_model, build_trainer


def _build_datasets(cfg, is_multimodal, has_test):
    train_df = pd.read_csv(cfg.data.train_csv)
    val_df   = pd.read_csv(cfg.data.val_csv)
    pool_df  = pd.concat([train_df, val_df], ignore_index=True)
    test_df  = pd.read_csv(cfg.data.test_csv) if has_test else None

    hes_dirs = [cfg.data.embeddings_dir_train, cfg.data.embeddings_dir_val]
    if has_test:
        hes_dirs.append(cfg.data.embeddings_dir_test)

    if is_multimodal:
        ihc_dirs = [cfg.data.embeddings_dir_ihc_train, cfg.data.embeddings_dir_ihc_val]
        if has_test:
            ihc_dirs.append(cfg.data.embeddings_dir_ihc_test)

        pool_dataset = DatasetMultiModal(pool_df, embeddings_dir_hes=hes_dirs, embeddings_dir_ihc=ihc_dirs)
        test_dataset = (
            DatasetMultiModal(test_df, embeddings_dir_hes=hes_dirs, embeddings_dir_ihc=ihc_dirs)
            if has_test else None
        )
    else:
        pool_dataset = H5Dataset(pool_df, embeddings_dir=hes_dirs)
        test_dataset = H5Dataset(test_df, embeddings_dir=hes_dirs) if has_test else None

    return pool_dataset, test_dataset


def _seeds(cfg):
    if "seeds" in cfg.cv and cfg.cv.seeds:
        return list(cfg.cv.seeds)
    seed_start = int(cfg.cv.get("seed_start", 0))
    n_seeds    = int(cfg.cv.get("n_seeds", 1))
    return list(range(seed_start, seed_start + n_seeds))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device  = torch.device(cfg.training.device)
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    model_name    = cfg.model.name
    entry         = MODEL_REGISTRY[model_name]
    is_multimodal = entry["multimodal"]

    has_test = bool(cfg.data.get("test_csv"))

    pool_dataset, test_dataset = _build_datasets(cfg, is_multimodal, has_test)
    pool_df = pool_dataset.df.reset_index(drop=True)

    seeds  = _seeds(cfg)
    method = cfg.cv.get("method", "kfold")
    k      = int(cfg.cv.get("k", 5))

    results_dir       = cfg.cv.get("results_dir", "outputs/cv_results")
    save_best_per_seed = bool(cfg.cv.get("save_best_per_seed", True))

    records = []

    seed_bar = tqdm(seeds, desc=f"{model_name} | seeds", unit="seed")
    for seed in seed_bar:
        seed_bar.set_postfix(seed=seed)
        set_seed(seed)
        splits = get_cv_splits(pool_df["label"].values, method, k, seed)
        seed_best = None

        fold_bar = tqdm(list(enumerate(splits)), desc=f"seed {seed} | folds", unit="fold", leave=False)
        for fold_idx, (train_idx, val_idx) in fold_bar:
            fold_bar.set_postfix(fold=f"{fold_idx + 1}/{len(splits)}")

            model, TrainerClass, _ = build_model(cfg.model, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
            trainer   = build_trainer(TrainerClass, model, optimizer, device, writer=None, cfg_training=cfg.training)

            train_subset = Subset(pool_dataset, train_idx)
            val_subset   = Subset(pool_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=cfg.training.batch_size, shuffle=True)
            val_loader   = DataLoader(val_subset, batch_size=cfg.training.batch_size, shuffle=False)

            fold_run_dir = os.path.join(run_dir, f"seed{seed}", f"fold{fold_idx}")
            epoch_desc = f"seed {seed} | fold {fold_idx + 1}/{len(splits)} | epochs"
            train_one_fold(trainer, TrainerClass, train_loader, val_loader, cfg, fold_run_dir, desc=epoch_desc)

            # ── Prédictions sur le fold de validation ──────────────────────
            eval_val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            val_probs, val_labels = collect_predictions(model, eval_val_loader, is_multimodal, device)
            val_slide_ids = pool_df.iloc[val_idx]["slide_id"].tolist()

            for sid, label, prob in zip(val_slide_ids, val_labels, val_probs):
                records.append({
                    "model": model_name, "seed": seed, "fold": fold_idx, "split": "val",
                    "slide_id": sid, "y_true": label, "y_prob": prob, "y_pred": int(prob > 0.5),
                })

            # ── Prédictions sur le jeu de test indépendant (optionnel) ─────
            if has_test:
                eval_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                test_probs, test_labels = collect_predictions(model, eval_test_loader, is_multimodal, device)
                test_slide_ids = test_dataset.df["slide_id"].tolist()

                for sid, label, prob in zip(test_slide_ids, test_labels, test_probs):
                    records.append({
                        "model": model_name, "seed": seed, "fold": fold_idx, "split": "test",
                        "slide_id": sid, "y_true": label, "y_prob": prob, "y_pred": int(prob > 0.5),
                    })

            # ── Suivi du meilleur modèle pour cette seed (toutes folds) ─────
            if save_best_per_seed and trainer.best_state_dict is not None:
                metric_val, mode = trainer.best_metric, trainer.mode
                is_better = (
                    seed_best is None
                    or (mode == "min" and metric_val < seed_best["metric"])
                    or (mode == "max" and metric_val > seed_best["metric"])
                )
                if is_better:
                    seed_best = {
                        "metric":     metric_val,
                        "monitor":    trainer.monitor,
                        "mode":       mode,
                        "fold":       fold_idx,
                        "epoch":      trainer.best_epoch,
                        "state_dict": trainer.best_state_dict,
                    }

        # ── Sauvegarde du meilleur modèle de la seed (toutes folds confondues) ─
        if save_best_per_seed and seed_best is not None:
            checkpoint = {
                "model_state_dict": seed_best["state_dict"],
                "model_cfg":        OmegaConf.to_container(cfg.model, resolve=True),
                "model_name":       model_name,
                "seed":             seed,
                "fold":             seed_best["fold"],
                "epoch":            seed_best["epoch"],
                "monitor":          seed_best["monitor"],
                "metric_value":     seed_best["metric"],
            }

            seed_dir = os.path.join(run_dir, f"seed{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(seed_dir, "best_model.pt"))

            stable_dir = os.path.join(results_dir, model_name)
            os.makedirs(stable_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(stable_dir, f"seed{seed}_best.pt"))

    predictions_df = pd.DataFrame.from_records(records)

    out_path = os.path.join(run_dir, "predictions.csv")
    predictions_df.to_csv(out_path, index=False)
    print(f"\nPrédictions sauvegardées -> {out_path}")

    os.makedirs(results_dir, exist_ok=True)
    stable_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
    predictions_df.to_csv(stable_path, index=False)
    print(f"Prédictions sauvegardées -> {stable_path}")


if __name__ == "__main__":
    main()
