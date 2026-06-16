import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.her2_datasets import H5Dataset, DatasetMultiModal
from training.cv_utils import set_seed
from utils.build import build_model as _build_model, build_trainer as _build_trainer

# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    seed = cfg.training.get("seed", None)
    if seed is not None:
        set_seed(int(seed))

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
    trainer   = _build_trainer(TrainerClass, model, optimizer, device, writer, cfg.training)
    trainer.fit(train_loader, val_loader, cfg, run_dir)

    writer.close()


if __name__ == "__main__":
    main()
