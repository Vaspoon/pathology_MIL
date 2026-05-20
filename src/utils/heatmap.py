"""
src/utils/heatmap.py — Génération de heatmaps d'attention pour WSI

Usage minimal :
    from utils.heatmap import generate_heatmap
    generate_heatmap(model, features, coords, slide_id="slide_001",
                     save_path="outputs/heatmap.png")
"""

import os
import logging
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@torch.no_grad()
def get_attention_scores(
    model:    nn.Module,
    features: torch.Tensor,   # [N, D]
    device:   torch.device,
) -> Tuple[np.ndarray, float]:
    """
    Extrait les scores d'attention normalisés depuis un modèle ABMIL.

    Compatible avec :
        - ABMIL              → retourne (logits, attn)
        - AttentionMIL_Papagoras → retourne (logits, attn)
        - MultiModalMILGated → retourne (logits, {'hes': ..., 'ihc': ...})

    Retourne :
        attn_norm : [N]   scores normalisés [0, 1]
        pred_prob : float probabilité de la classe positive
    """
    model.eval()
    features = features.to(device)

    out = model(features)

    # Normalisation selon le type de sortie
    if isinstance(out, tuple) and len(out) == 2:
        logits, attn = out
        # MultiModalMILGated retourne un dict pour attn
        if isinstance(attn, dict):
            attn = attn["hes"]
    else:
        raise ValueError(
            "Le modèle doit retourner (logits, attn). "
            "Vérifiez que vous utilisez ABMIL ou AttentionMIL_Papagoras."
        )

    # Normalisation min-max
    attn = attn.cpu().float()
    a_min, a_max = attn.min(), attn.max()
    attn_norm = ((attn - a_min) / (a_max - a_min + 1e-8)).numpy()

    # Probabilité prédite
    logits = logits.cpu()
    if logits.shape[-1] == 2:
        pred_prob = torch.softmax(logits, dim=-1)[1].item()
    else:
        pred_prob = torch.sigmoid(logits).squeeze().item()

    return attn_norm, pred_prob


def build_heatmap_grid(
    attn_scores: np.ndarray,    # [N]
    coords:      np.ndarray,    # [N, 2] (x, y) en pixels
    patch_size:  int = 256,
    downsample:  int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit la grille 2D des scores d'attention.

    Retourne :
        heatmap  : [H, W] float32 — valeurs entre 0 et 1
        coverage : [H, W] bool    — True là où il y a des patches
    """
    wsi_w = int(coords[:, 0].max()) + patch_size
    wsi_h = int(coords[:, 1].max()) + patch_size
    h_ds  = wsi_h // downsample
    w_ds  = wsi_w // downsample
    ps_ds = patch_size // downsample

    heatmap = np.zeros((h_ds, w_ds), dtype=np.float32)
    count   = np.zeros((h_ds, w_ds), dtype=np.float32)

    for i, (x, y) in enumerate(zip(coords[:, 0], coords[:, 1])):
        x_ds  = int(x) // downsample
        y_ds  = int(y) // downsample
        x_end = min(x_ds + ps_ds, w_ds)
        y_end = min(y_ds + ps_ds, h_ds)
        heatmap[y_ds:y_end, x_ds:x_end] += attn_scores[i]
        count  [y_ds:y_end, x_ds:x_end] += 1.0

    valid = count > 0
    heatmap[valid] /= count[valid]
    return heatmap, valid


def generate_heatmap(
    model:      nn.Module,
    features:   torch.Tensor,               # [N, D]
    coords:     torch.Tensor,               # [N, 2]
    device:     torch.device,
    slide_id:   str  = "slide",
    wsi_thumb:  Optional[np.ndarray] = None,  # [H, W, 3] thumbnail optionnel
    save_path:  Optional[str] = None,
    patch_size: int   = 256,
    downsample: int   = 4,
    colormap:   str   = "jet",
    alpha:      float = 0.5,
    top_k:      int   = 10,
) -> Dict:
    """
    Pipeline complet : attention → heatmap → figure.

    Args:
        model      : ABMIL ou AttentionMIL_Papagoras
        features   : [N, D] embeddings du bag
        coords     : [N, 2] coordonnées (x, y) en pixels
        device     : torch.device
        slide_id   : identifiant de la lame (titre de la figure)
        wsi_thumb  : thumbnail WSI [H, W, 3] pour l'overlay (optionnel)
        save_path  : chemin de sauvegarde (None = pas de sauvegarde)
        patch_size : taille des patches en pixels
        downsample : facteur de réduction pour la heatmap
        colormap   : colormap matplotlib ("jet", "hot", "viridis"...)
        alpha      : transparence de l'overlay sur le thumbnail
        top_k      : nombre de top patches à afficher dans les logs

    Retourne dict avec :
        'attn_norm'   : [N] scores normalisés
        'pred_prob'   : float probabilité de la classe positive
        'top_patches' : dict {'coords': ..., 'scores': ...}
        'fig'         : Figure matplotlib
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image

    coords_np = coords.numpy() if isinstance(coords, torch.Tensor) else coords

    # ── 1. Scores d'attention ────────────────────────────────
    attn_norm, pred_prob = get_attention_scores(model, features, device)

    # ── 2. Grille 2D ────────────────────────────────────────
    heatmap, coverage = build_heatmap_grid(attn_norm, coords_np, patch_size, downsample)

    # ── 3. Colormap → RGBA ──────────────────────────────────
    cmap      = cm.get_cmap(colormap)
    rgba      = cmap(heatmap)
    rgba[~coverage, 3] = 0.0
    heatmap_rgba = (rgba * 255).astype(np.uint8)

    # ── 4. Figure ────────────────────────────────────────────
    ncols = 3 if wsi_thumb is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    fig.suptitle(f"{slide_id}  |  P(positif) = {pred_prob:.3f}", fontsize=13)

    if wsi_thumb is not None:
        axes[0].imshow(wsi_thumb)
        axes[0].set_title("Thumbnail WSI")
        axes[0].axis("off")
        ax_heat, ax_merge = axes[1], axes[2]
    else:
        ax_heat, ax_merge = axes[0], axes[1]

    # Heatmap seule
    ax_heat.imshow(heatmap_rgba)
    ax_heat.set_title("Heatmap d'attention")
    ax_heat.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap(colormap), norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_heat, fraction=0.046, pad=0.04, label="Attention normalisée")

    # Overlay ou distribution
    if wsi_thumb is not None:
        th, tw = wsi_thumb.shape[:2]
        overlay = np.array(Image.fromarray(heatmap_rgba).resize((tw, th), Image.BILINEAR))
        ax_merge.imshow(wsi_thumb)
        ax_merge.imshow(overlay, alpha=alpha)
        ax_merge.set_title(f"Overlay (α={alpha})")
        ax_merge.axis("off")
    else:
        scores_sorted = np.sort(attn_norm)[::-1]
        ax_merge.bar(range(len(scores_sorted)), scores_sorted,
                     color=cmap(scores_sorted), width=1.0)
        ax_merge.set_title("Distribution des scores (triés)")
        ax_merge.set_xlabel("Rang du patch")
        ax_merge.set_ylabel("Score normalisé")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[Heatmap] Sauvegardée → {save_path}")

    # ── 5. Top-K patches ─────────────────────────────────────
    topk_idx    = np.argsort(attn_norm)[::-1][:top_k]
    top_patches = {
        "indices": topk_idx,
        "coords":  coords_np[topk_idx],
        "scores":  attn_norm[topk_idx],
    }

    log.info(f"[{slide_id}] P(positif)={pred_prob:.3f} | Top-{top_k} patches :")
    for rank, i in enumerate(topk_idx):
        log.info(
            f"  #{rank+1} patch={i:5d} | "
            f"coord=({coords_np[i,0]:5d},{coords_np[i,1]:5d}) | "
            f"score={attn_norm[i]:.4f}"
        )

    return {
        "attn_norm":   attn_norm,
        "pred_prob":   pred_prob,
        "top_patches": top_patches,
        "fig":         fig,
    }
