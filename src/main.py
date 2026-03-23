import torch
import pandas as pd
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.her2_dataset import HER2Dataset
from models.multimodal_mil import MultiModalMIL
from training.trainer import Trainer

def main(config_path):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg["data"]["csv"])

    dataset = HER2Dataset(df, cfg["data"]["embeddings_dir"])

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    model = MultiModalMIL(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    trainer = Trainer(model, optimizer, device)

    for epoch in range(cfg["training"]["epochs"]):
        loss = trainer.train_epoch(loader)
        print(f"Epoch {epoch} | Loss: {loss:.4f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])