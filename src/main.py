import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from datasets.her2_datasets import H5Dataset, DatasetMultiModal
from training.trainer import TrainerABMIL, TrainerMultiModalABMIL
from training.trainer_clam import TrainerCLAM
from training.trainer_contrastive import TrainerContrastiveMultiModalABMIL
from utils.build import build_model as _build_model, build_trainer as _build_trainer

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

    if TrainerClass in (TrainerABMIL, TrainerCLAM, TrainerMultiModalABMIL, TrainerContrastiveMultiModalABMIL):
        trainer = _build_trainer(TrainerClass, model, optimizer, device, writer, cfg.training)
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
