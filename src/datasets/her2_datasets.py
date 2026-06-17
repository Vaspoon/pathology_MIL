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
    """
    Args:
        embeddings_dir: répertoire des embeddings .h5, ou liste de répertoires
            (utilisé lorsqu'on fusionne plusieurs splits, ex: train+val pour CV).
            Le premier répertoire contenant le fichier est utilisé.
    """

    def __init__(self, dataframe, embeddings_dir):
        if isinstance(embeddings_dir, (str, Path)):
            self.embeddings_dirs = [embeddings_dir]
        else:
            self.embeddings_dirs = list(embeddings_dir)
        self.df = self._filter_existing(dataframe)

    def _find_path(self, slide_id):
        slide_id = str(slide_id).zfill(2)
        for d in self.embeddings_dirs:
            p = Path(d, f"{slide_id}.h5")
            if p.exists():
                return p
        return None

    def _filter_existing(self, dataframe):
        mask = dataframe["slide_id"].apply(lambda sid: self._find_path(sid) is not None)
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
        with h5py.File(self._find_path(row.slide_id), "r") as f:
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
        slide_id = str(row.slide_id).zfill(2)

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
        embeddings_dir_hes : répertoire des embeddings HES (.h5), ou liste de répertoires
        embeddings_dir_ihc : répertoire des embeddings IHC (.h5), ou liste de répertoires
    """

    def __init__(self, dataframe, embeddings_dir_hes, embeddings_dir_ihc):
        self.df = dataframe.reset_index(drop=True)
        self.embeddings_dirs_hes = (
            [embeddings_dir_hes] if isinstance(embeddings_dir_hes, (str, Path)) else list(embeddings_dir_hes)
        )
        self.embeddings_dirs_ihc = (
            [embeddings_dir_ihc] if isinstance(embeddings_dir_ihc, (str, Path)) else list(embeddings_dir_ihc)
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _find_path(dirs, file_id):
        file_id = str(file_id).zfill(2)
        for d in dirs:
            p = Path(d, f"{file_id}.h5")
            if p.exists():
                return p
        raise FileNotFoundError(f"{file_id}.h5 introuvable dans {dirs}")

    def _load_h5(self, path) -> torch.Tensor:
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

        hes   = self._load_h5(self._find_path(self.embeddings_dirs_hes, row.slide_id))
        ihc   = self._load_h5(self._find_path(self.embeddings_dirs_ihc, row.ihc_id))
        label = torch.tensor(row.label, dtype=torch.float32)

        return hes, ihc, label
