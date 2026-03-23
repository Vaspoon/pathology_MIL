import torch
from torch.utils.data import Dataset
import numpy as np

class HER2Dataset(Dataset):
    def __init__(self, dataframe, embeddings_dir):
        self.df = dataframe
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        hes = np.load(f"{self.embeddings_dir}/{row.hes_id}.npy")
        ihc = np.load(f"{self.embeddings_dir}/{row.ihc_id}.npy")

        hes = torch.tensor(hes, dtype=torch.float32)
        ihc = torch.tensor(ihc, dtype=torch.float32)

        label = torch.tensor(row.label, dtype=torch.float32)

        return hes, ihc, label