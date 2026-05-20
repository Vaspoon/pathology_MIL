import inspect
import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from datasets.her2_datasets import H5Dataset, DatasetMultiModal
from models.abmil import ABMIL
from models.attention_mil import AttentionMIL_Papagoras
from models.mil_max_pooling import MILMaxPooling
from models.mil_mean_pooling import MILMeanPooling
from models.multimodal_porpoise import MultiModalMILPorpoise
from models.multimodal_mil import MultiModalMILGated
from models.transmil import TransMIL
from training.trainer import Trainer, TrainerABMIL, TrainerMultiModalABMIL

# ---------------------------------------------------------------------------
# Registry: model name → {cls, trainer, multimodal}
#
#   multimodal=False  →  H5Dataset (hes only),       forward(hes)
#   multimodal=True   →  DatasetMultiModal (hes+ihc), forward(hes, ihc)
#
# Trainer classes:
#   Trainer                  — returns scalar logit (mil_max / mil_mean)
#   TrainerABMIL             — returns (logits, attn), full metrics
#   TrainerMultiModalABMIL   — returns (logits, attn), full metrics, multimodal batches
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "mil_max":              {"cls": MILMaxPooling,          "trainer": Trainer,                "multimodal": False},
    "mil_mean":             {"cls": MILMeanPooling,         "trainer": Trainer,                "multimodal": False},
    "attention_mil":        {"cls": AttentionMIL_Papagoras, "trainer": TrainerABMIL,           "multimodal": False},
    "abmil":                {"cls": ABMIL,                  "trainer": TrainerABMIL,           "multimodal": False},
    "transmil":             {"cls": TransMIL,               "trainer": TrainerABMIL,           "multimodal": False},
    "multimodal_porpoise":  {"cls": MultiModalMILPorpoise,  "trainer": TrainerMultiModalABMIL, "multimodal": True},
    "multimodal_gated":     {"cls": MultiModalMILGated,     "trainer": TrainerMultiModalABMIL, "multimodal": True},
}


def _build_model(cfg_model, device):
    """Instantiate model from registry, passing only params accepted by __init__."""
    name = cfg_model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    valid  = set(inspect.signature(entry["cls"].__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg_model).items() if k in valid}
    return entry["cls"](**kwargs).to(device), entry["trainer"], entry["multimodal"]


# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    device  = torch.device(cfg.training.device)
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer  = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    # ── Datasets ────────────────────────────────────────────────────────────
    train_df = pd.read_csv(cfg.data.train_csv)
    val_df   = pd.read_csv(cfg.data.val_csv)

    model, TrainerClass, is_multimodal = _build_model(cfg.model, device)

    if is_multimodal:
        # Expects CSV columns: slide_id, ihc_id, label
        train_dataset = DatasetMultiModal(
            train_df,
            embeddings_dir_hes=cfg.data.embeddings_dir_train,
            embeddings_dir_ihc=cfg.data.embeddings_dir_ihc_train,
        )
        val_dataset = DatasetMultiModal(
            val_df,
            embeddings_dir_hes=cfg.data.embeddings_dir_val,
            embeddings_dir_ihc=cfg.data.embeddings_dir_ihc_val,
        )
    else:
        train_dataset = H5Dataset(train_df, embeddings_dir=cfg.data.embeddings_dir_train)
        val_dataset   = H5Dataset(val_df,   embeddings_dir=cfg.data.embeddings_dir_val)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.training.batch_size, shuffle=False)

    # ── Training ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    if TrainerClass in (TrainerABMIL, TrainerMultiModalABMIL):
        trainer = TrainerClass(
            model, optimizer, device, writer,
            early_stopping_patience=cfg.training.early_stopping_patience,
            monitor=cfg.training.monitor,
        )
        trainer.fit(train_loader, val_loader, cfg, run_dir)

    else:
        # Basic trainer (mil_max / mil_mean) — preserves original manual loop
        trainer = TrainerClass(model, optimizer, device, writer)
        for epoch in trange(cfg.training.epochs, desc="Training Epochs"):
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_loss   = trainer.eval_epoch(val_loader, epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if (epoch + 1) % cfg.training.save_every == 0:
                ckpt = os.path.join(run_dir, f"model_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"Saved checkpoint: {ckpt}")
        final = os.path.join(run_dir, f"model_final_{cfg.training.batch_size}_{cfg.training.lr}.pt")
        torch.save(model, final)
        print(f"Saved final model: {final}")

    writer.close()


if __name__ == "__main__":
    main()
