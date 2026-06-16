"""
src/eval/cv_metrics.py

Helpers partagés par les notebooks notebooks/eval_cv_<model>.ipynb pour
analyser les prédictions produites par src/train_cv.py
(outputs/cv_results/<model>_predictions.csv).

Fonctions :
    load_predictions          : charge le CSV de prédictions d'un modèle
    per_group_metrics          : AUC / accuracy / ROC par (seed, fold, split)
    aggregate_metrics          : moyenne ± std des métriques, agrégées par seed
    mean_roc_curve              : courbe ROC moyenne (interpolée) + bande de variance
    wilcoxon_seed_stability     : test de Wilcoxon signé (AUC par fold) entre deux seeds
    wilcoxon_class_separation   : test de Wilcoxon rank-sum, scores classe 0 vs classe 1
    wilcoxon_compare_models     : test de Wilcoxon signé sur AUC (seed, fold) entre 2 modèles
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.stats import ranksums, wilcoxon

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "cv_results"


def load_predictions(model_name: str, results_dir=DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Charge outputs/cv_results/<model_name>_predictions.csv."""
    path = Path(results_dir) / f"{model_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancez d'abord :\n"
            f"  python src/train_cv.py model={model_name} ..."
        )
    return pd.read_csv(path)


def per_group_metrics(df: pd.DataFrame, group_cols=("model", "seed", "fold", "split")) -> pd.DataFrame:
    """
    Calcule AUC et accuracy pour chaque groupe (par défaut : model, seed, fold, split).

    Les groupes à une seule classe (ex: fold LOO avec un seul échantillon) ont AUC = NaN.
    """
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        y_true = g["y_true"].values
        y_prob = g["y_prob"].values
        y_pred = g["y_pred"].values

        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        except ValueError:
            auc = np.nan

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["auc"] = auc
        row["accuracy"] = acc
        row["n"] = len(g)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame, group_cols=("model", "seed", "split")) -> pd.DataFrame:
    """
    Agrège les métriques par-fold en moyenne ± std par groupe (par défaut: par seed).

    Pour les folds LOO (n=1, auc=NaN), on agrège plutôt les prédictions concaténées
    via `pooled_metrics` — cette fonction est surtout utile pour le k-fold.
    """
    return (
        metrics_df.groupby(list(group_cols))[["auc", "accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )


def pooled_metrics(df: pd.DataFrame, group_cols=("model", "seed", "split")) -> pd.DataFrame:
    """
    Calcule AUC/accuracy sur les prédictions poolées de tous les folds d'une même seed.
    Recommandé pour LOO (chaque fold n'a qu'un seul point, AUC indéfinie par fold).
    """
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        y_true = g["y_true"].values
        y_prob = g["y_prob"].values
        y_pred = g["y_pred"].values

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        row["accuracy"] = accuracy_score(y_true, y_pred)
        row["n"] = len(g)
        rows.append(row)

    return pd.DataFrame(rows)


def mean_roc_curve(df: pd.DataFrame, n_points: int = 100):
    """
    Courbe ROC moyenne (interpolée sur une grille commune de FPR) à travers
    tous les groupes (seed, fold) présents dans `df`, + écart-type du TPR.

    Retourne (mean_fpr, mean_tpr, std_tpr).
    """
    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []

    for (_, _), g in df.groupby(["seed", "fold"]):
        y_true = g["y_true"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, g["y_prob"].values)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    tprs = np.array(tprs)
    return mean_fpr, tprs.mean(axis=0), tprs.std(axis=0)


def wilcoxon_seed_stability(metrics_df: pd.DataFrame, seed_a: int, seed_b: int, metric: str = "auc"):
    """
    Test de Wilcoxon signé sur les valeurs de `metric` par fold,
    entre deux seeds (paires = mêmes indices de fold).

    Retourne (statistic, p_value, n_pairs).
    """
    a = metrics_df.loc[metrics_df["seed"] == seed_a].sort_values("fold")[metric].values
    b = metrics_df.loc[metrics_df["seed"] == seed_b].sort_values("fold")[metric].values

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 1 or np.allclose(a, b):
        return np.nan, np.nan, len(a)

    stat, p = wilcoxon(a, b)
    return stat, p, len(a)


def wilcoxon_class_separation(df: pd.DataFrame, split: str = "val"):
    """
    Test de Wilcoxon rank-sum (ranksums) comparant les probabilités prédites
    pour les échantillons de label 0 vs label 1 — vérifie que le modèle
    sépare les deux classes (probas plus hautes pour la classe 1).

    Retourne (statistic, p_value).
    """
    sub = df[df["split"] == split]
    probs_0 = sub.loc[sub["y_true"] == 0, "y_prob"].values
    probs_1 = sub.loc[sub["y_true"] == 1, "y_prob"].values
    return ranksums(probs_1, probs_0)


def wilcoxon_compare_models(metrics_a: pd.DataFrame, metrics_b: pd.DataFrame, metric: str = "auc"):
    """
    Test de Wilcoxon signé comparant `metric` entre deux modèles, appariés
    par (seed, fold). Les deux DataFrames doivent provenir de `per_group_metrics`
    et partager la même grille (seed, fold).

    Retourne (statistic, p_value, n_pairs).
    """
    merged = metrics_a.merge(metrics_b, on=["seed", "fold", "split"], suffixes=("_a", "_b"))
    a = merged[f"{metric}_a"].values
    b = merged[f"{metric}_b"].values

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 1 or np.allclose(a, b):
        return np.nan, np.nan, len(a)

    stat, p = wilcoxon(a, b)
    return stat, p, len(a)


def list_available_models(results_dir=DEFAULT_RESULTS_DIR):
    """Liste les modèles pour lesquels un fichier <model>_predictions.csv existe."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(p.stem.replace("_predictions", "") for p in results_dir.glob("*_predictions.csv"))
