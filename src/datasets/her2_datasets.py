"""
src/datasets/her2_datasets.py

H5Dataset: construction du dataset en se basant sur HER2 pour premiers tests.
WSIDatasetWithCoords: construction du dataset pour wsi avec coords pour la heatmap
DatasetMultiModal : dataset général pour les WSI, target entrainement multimodal
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path

class H5Dataset(Dataset):
    def __init__(self, dataframe, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.df = self._filter_existing(dataframe)

    def _filter_existing(self, dataframe):
        mask = dataframe["slide_id"].apply(
            lambda sid: Path(self.embeddings_dir, f"{sid}.h5").exists()
        )
        missing_ids = dataframe.loc[~mask, "slide_id"].tolist()
        if missing_ids:
            print(f"[H5Dataset] {len(missing_ids)} file(s) .h5 cannot be found - ignored :")
            for sid in missing_ids:
                print(f"  - {sid}.h5")
        return dataframe[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with h5py.File(Path(self.embeddings_dir, f"{row.slide_id}.h5"), "r") as f:
            if "features" in f:
                hes = f["features"][:]
            elif "embeddings" in f:
                hes = f["embeddings"][:]
            else:
                raise KeyError("Neither 'features' nor 'embeddings' found in file")

        hes   = torch.tensor(hes, dtype=torch.float32)
        label = torch.tensor(row.label, dtype=torch.float32)

        return hes, label

class WSIDatasetWithCoords(Dataset):
    """
    Étend HER2Dataset avec les coordonnées (x, y) des patches.
    Nécessaire pour générer les heatmaps d'attention.

    Attend dans le .h5 :
        "features"  ou "embeddings" : [N, D]
        "coords"                    : [N, 2]  (x, y en pixels) — optionnel

    Retourne :
        hes     : [N, D]   embeddings
        coords  : [N, 2]   coordonnées (zeros si absentes du .h5)
        label   : scalaire
        slide_id: str
    """

    def __init__(self, dataframe, embeddings_dir):
        self.df             = dataframe
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row.slide_id)

        with h5py.File(f"{self.embeddings_dir}/{slide_id}.h5", "r") as f:
            # Features
            if "features" in f:
                hes = f["features"][:]
            elif "embeddings" in f:
                hes = f["embeddings"][:]
            else:
                raise KeyError(f"[{slide_id}] Ni 'features' ni 'embeddings' dans le .h5")

            if "coords" in f:
                coords = f["coords"][:]
            else:
                # Crée une grille synthétique si coords absentes
                N = hes.shape[0]
                side = int(np.ceil(np.sqrt(N)))
                xs   = (np.arange(N) % side) * 256
                ys   = (np.arange(N) // side) * 256
                coords = np.stack([xs, ys], axis=1)

        hes    = torch.tensor(hes,    dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.int64)
        label  = torch.tensor(row.label, dtype=torch.float32)

        return hes, coords, label, slide_id



class DatasetMultiModal(Dataset):
    """
    Dataset pour l'entraînement multimodal HES + IHC.

    Attend dans le DataFrame :
        slide_id : identifiant de la lame HES
        ihc_id   : identifiant de la lame IHC correspondante
        label    : label de la lame

    Args:
        dataframe        : DataFrame avec colonnes [slide_id, ihc_id, label]
        embeddings_dir_hes : répertoire des embeddings HES (.h5)
        embeddings_dir_ihc : répertoire des embeddings IHC (.h5)
    """

    def __init__(self, dataframe, embeddings_dir_hes, embeddings_dir_ihc):
        self.df                 = dataframe
        self.embeddings_dir_hes = embeddings_dir_hes
        self.embeddings_dir_ihc = embeddings_dir_ihc

    def __len__(self):
        return len(self.df)

    def _load_h5(self, path: str) -> torch.Tensor:
        with h5py.File(path, "r") as f:
            if "features" in f:
                data = f["features"][:]
            elif "embeddings" in f:
                data = f["embeddings"][:]
            else:
                raise KeyError(f"Ni 'features' ni 'embeddings' dans {path}")
        return torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        hes   = self._load_h5(f"{self.embeddings_dir_hes}/{row.slide_id}.h5")
        ihc   = self._load_h5(f"{self.embeddings_dir_ihc}/{row.ihc_id}.h5")
        label = torch.tensor(row.label, dtype=torch.float32)

        return hes, ihc, label
