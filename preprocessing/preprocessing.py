import os
import yaml
import torch
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn

# --------------------
# Encoders
# --------------------
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --------------------
# Dataset construction
# --------------------
class WSIPatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.paths = sorted([
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path

# Main Script
def main(config_path):

    # loading config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["data"]["root_dir"]
    output_dir = cfg["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # transforms
    transform = T.Compose([
        T.Resize((cfg["transforms"]["resize"], cfg["transforms"]["resize"])),
        T.ToTensor(),
        T.Normalize(
            mean=cfg["transforms"]["mean"],
            std=cfg["transforms"]["std"]
        )
    ])

    # Encoder
    model = SimpleCNN(cfg["model"]["embedding_dim"]).to(device)
    model.eval()

    if cfg["model"]["weights_path"] is not None:
        model.load_state_dict(torch.load(cfg["model"]["weights_path"], map_location=device))

    
    wsi_folders = sorted(os.listdir(root_dir))
    # loop on WSI 
    for wsi_name in tqdm(wsi_folders):
        wsi_path = os.path.join(root_dir, wsi_name)
        h5_path = os.path.join(output_dir, f"{wsi_name}.h5")
        if os.path.isdir(h5_path):
            print(f"File {h5_path} already exists, skipping file")
            continue
        if not os.path.isdir(wsi_path):
            continue

        print(f"\nProcessing WSI: {wsi_name}")

        dataset = WSIPatchDataset(wsi_path, transform)
        loader = DataLoader(
            dataset,
            batch_size=cfg["dataloader"]["batch_size"],
            shuffle=False,
            num_workers=cfg["dataloader"]["num_workers"],
            pin_memory=cfg["dataloader"]["pin_memory"]
        )



        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset(
                "embeddings",
                shape=(len(dataset), cfg["model"]["embedding_dim"]),
                dtype="float32"
            )

            path_ds = f.create_dataset(
                "paths",
                shape=(len(dataset),),
                dtype=h5py.string_dtype()
            )

            idx = 0

            with torch.no_grad():
                for imgs, paths in tqdm(loader):
                    imgs = imgs.to(device)

                    emb = model(imgs).cpu().numpy()

                    batch_size = emb.shape[0]

                    dset[idx:idx + batch_size] = emb
                    path_ds[idx:idx + batch_size] = paths

                    idx += batch_size

        print(f"Saved: {h5_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)