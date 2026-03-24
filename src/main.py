import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets.her2_datasets import HER2Dataset
from models.mil_max_pooling import MILMaxPooling
from training.trainer import Trainer
from tqdm import trange


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg)) 

    device = torch.device(cfg.training.device)

    # hydra management
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tb_dir = os.path.join(run_dir, "tensorboard")

    writer = SummaryWriter(log_dir=tb_dir) 

    # load datasets
    train_df = pd.read_csv(cfg.data.train_csv)  
    val_df = pd.read_csv(cfg.data.val_csv)


    train_dataset = HER2Dataset(train_df, embeddings_dir=cfg.data.embeddings_dir_train)
    val_dataset = HER2Dataset(val_df, embeddings_dir=cfg.data.embeddings_dir_val)
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)

    # model
    model = MILMaxPooling(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr
    )

    trainer = Trainer(model, optimizer, device, writer)

    # training loop
# training loop
    for epoch in trange(cfg.training.epochs, desc="Training Epochs"):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.eval_epoch(val_loader, epoch)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # save model every cfg.training.save_every epochs
        if (epoch + 1) % cfg.training.save_every == 0:
            model_path = os.path.join(run_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved checkpoint: {model_path}")

    # save final model at the end
    final_model_path = os.path.join(run_dir, f"model_final_{cfg.training.batch_size}_{cfg.training.lr}.pt")
    torch.save(model, final_model_path)
    print(f"Saved final model: {final_model_path}")

    writer.close()


if __name__ == "__main__":
    main()