# pathology_MIL
Personal Repository to test multiple instance learning models on pathology slides images

## Installation

```bash
# 1. Cloner le repo
git clone <url>
cd pathology_MIL

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Installer PyTorch (selon ta version de CUDA — voir https://pytorch.org)
# CUDA 12.8 :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# CPU uniquement :
# pip install torch torchvision torchaudio

# 4. Installer les autres dépendances
pip install -r requirements.txt

# 5. Installer le package en mode éditable (permet d'importer src/ sans manipuler PYTHONPATH)
pip install -e .
```

> **Python** : 3.9+ recommandé (testé avec 3.10).

## Modèles disponibles

Tous les modèles sont déclarés dans le registre [`src/utils/build.py`](src/utils/build.py) (`MODEL_REGISTRY`) et configurés via [`configs/model/`](configs/model). Sélection avec `model=<name>`.

| `model=<name>`           | Architecture                                         | Trainer                              | Entrée                | Config                                                            |
|---------------------------|-------------------------------------------------------|----------------------------------------|-------------------------|----------------------------------------------------------------------|
| `mil_max`                  | MIL Max Pooling (baseline)                            | `Trainer`                              | HES                     | [`configs/model/mil_max.yaml`](configs/model/mil_max.yaml)             |
| `mil_mean`                 | MIL Mean Pooling (baseline)                           | `Trainer`                              | HES                     | [`configs/model/mil_mean.yaml`](configs/model/mil_mean.yaml)           |
| `attention_mil`            | Attention MIL (Papagoras), attention tanh             | `TrainerABMIL`                         | HES                     | [`configs/model/attention_mil.yaml`](configs/model/attention_mil.yaml) |
| `abmil`                    | ABMIL — Gated Attention (Ilse et al., 2018)           | `TrainerABMIL`                         | HES                     | [`configs/model/abmil.yaml`](configs/model/abmil.yaml)                 |
| `transmil`                 | TransMIL (Shao et al., NeurIPS 2021)                  | `TrainerABMIL`                         | HES                     | [`configs/model/transmil.yaml`](configs/model/transmil.yaml)           |
| `clam_sb`                  | CLAM Single-Branch (Lu et al., 2021)                  | `TrainerCLAM`                          | HES                     | [`configs/model/clam_sb.yaml`](configs/model/clam_sb.yaml)             |
| `clam_mb`                  | CLAM Multi-Branch (Lu et al., 2021)                   | `TrainerCLAM`                          | HES                     | [`configs/model/clam_mb.yaml`](configs/model/clam_mb.yaml)             |
| `multimodal_gated`         | Fusion HES+IHC par cross-attention                    | `TrainerMultiModalABMIL`               | HES + IHC               | [`configs/model/multimodal_gated.yaml`](configs/model/multimodal_gated.yaml) |
| `multimodal_porpoise`      | Fusion tardive HES+IHC (PORPOISE-style, Chen et al.)  | `TrainerMultiModalABMIL`               | HES + IHC               | [`configs/model/multimodal_porpoise.yaml`](configs/model/multimodal_porpoise.yaml) |
| `contrastive_multimodal`   | Fusion contrastive HES+IHC + modality dropout         | `TrainerContrastiveMultiModalABMIL`    | HES + IHC               | [`configs/model/contrastive_multimodal.yaml`](configs/model/contrastive_multimodal.yaml) |

Les modèles "HES + IHC" attendent un CSV avec les colonnes `slide_id, ihc_id, label` et deux répertoires d'embeddings (`embeddings_dir_*` et `embeddings_dir_ihc_*`).

## Entraînement simple (`src/main.py`)

Un run = un split train/val fixe, avec early stopping et checkpointing (`best.pt`, `last.pt`, `epoch_N.pt`).

```bash
# ABMIL
python src/main.py model=abmil

# Modèle original inchangé
python src/main.py model=mil_max

# TransMIL avec override de config
python src/main.py model=transmil training.epochs=50 training.lr=5e-5

# Modèle multimodal (HES + IHC)
python src/main.py model=multimodal_gated

# Heatmaps sur le val set
python src/main.py mode=infer model=abmil checkpoint_path=outputs/.../best.pt

# Grid search (Hydra multirun)
python src/main.py --multirun model=abmil training.lr=1e-4,5e-5
```

Paramètres d'entraînement clés (`configs/training/default.yaml`, `configs/config.yaml`) :

| Paramètre                          | Rôle                                                                 |
|--------------------------------------|------------------------------------------------------------------------|
| `training.epochs`                    | nombre d'epochs maximum                                                 |
| `training.early_stopping`            | active/désactive l'early stopping (`TrainerABMIL` et dérivés)          |
| `training.early_stopping_patience`   | patience de l'early stopping                                            |
| `training.monitor`                   | métrique surveillée : `val_loss` ou `val_auc`                          |
| `training.bag_weight`                | poids bag-loss / instance-loss pour CLAM (`clam_sb`, `clam_mb`)        |

## Cross-validation multi-seeds (`src/train_cv.py`)

Entraîne un modèle sur **plusieurs seeds** et avec **k-fold** ou **Leave-One-Out (LOO)** cross-validation. Les jeux `train.csv` + `val.csv` sont fusionnés en un pool sur lequel les folds sont générés ; si `data.test_csv` est renseigné, chaque modèle entraîné est en plus évalué sur ce jeu de test indépendant.

```bash
# ABMIL — 5-fold CV, 10 seeds, avec early stopping
python src/train_cv.py model=abmil cv.method=kfold cv.k=5 cv.n_seeds=10

# CLAM-SB — Leave-One-Out, 5 seeds, sans early stopping
python src/train_cv.py model=clam_sb cv.method=loo cv.n_seeds=5 training.early_stopping=false

# Modèle multimodal avec jeu de test indépendant
python src/train_cv.py model=multimodal_gated \
  data.train_csv=data/splits_ihc4bc/her2/train.csv \
  data.val_csv=data/splits_ihc4bc/her2/val.csv \
  data.test_csv=data/splits_ihc4bc/her2/test.csv \
  cv.method=kfold cv.k=5 cv.n_seeds=10
```

Paramètres de cross-validation (`cv:` dans `configs/config.yaml`) :

| Paramètre              | Rôle                                                                          |
|--------------------------|----------------------------------------------------------------------------------|
| `cv.method`              | `kfold` (StratifiedKFold) ou `loo` (Leave-One-Out)                               |
| `cv.k`                   | nombre de folds (ignoré si `method=loo`)                                          |
| `cv.n_seeds` / `cv.seed_start` | nombre de seeds testées et seed de départ (`seeds = seed_start..seed_start+n_seeds-1`) |
| `cv.seeds`               | liste explicite de seeds (remplace `n_seeds`/`seed_start` si fournie)            |
| `cv.save_checkpoints`    | sauvegarder les `.pt` de chaque fold/seed (volumineux, surtout en LOO)           |
| `cv.save_best_per_seed`  | sauvegarder un seul modèle (le meilleur, toutes folds confondues) par seed       |
| `cv.results_dir`         | dossier de sortie stable des prédictions (par défaut `outputs/cv_results/`)      |

Chaque run écrit les prédictions par échantillon (`slide_id, seed, fold, split, y_true, y_prob, y_pred`) dans :
- le dossier de run Hydra (`outputs/<run_dir>/predictions.csv`)
- un chemin stable `outputs/cv_results/<model>_predictions.csv`, lu par les notebooks d'évaluation.

Si `cv.save_best_per_seed=true` (par défaut), le meilleur modèle de chaque seed (toutes folds confondues, sélectionné selon `training.monitor`) est sauvegardé dans :
- `outputs/<run_dir>/seed<N>/best_model.pt`
- `outputs/cv_results/<model>/seed<N>_best.pt` (chemin stable)

Chaque checkpoint contient `model_state_dict`, `model_cfg`, `model_name`, `seed`, `fold`, `epoch`, `monitor`, `metric_value`.

## Évaluation (notebooks)

Un notebook d'évaluation par architecture, dans [`notebooks/`](notebooks) :
`eval_cv_mil_max.ipynb`, `eval_cv_mil_mean.ipynb`, `eval_cv_attention_mil.ipynb`, `eval_cv_abmil.ipynb`, `eval_cv_transmil.ipynb`, `eval_cv_clam_sb.ipynb`, `eval_cv_clam_mb.ipynb`, `eval_cv_multimodal_gated.ipynb`, `eval_cv_multimodal_porpoise.ipynb`, `eval_cv_contrastive_multimodal.ipynb`.

Chaque notebook s'appuie sur la librairie partagée [`src/eval/cv_metrics.py`](src/eval/cv_metrics.py) et, à partir de `outputs/cv_results/<model>_predictions.csv`, calcule :

1. **Métriques par (seed, fold)** — AUC, accuracy, sur `val` et `test`, puis agrégées par seed (moyenne ± std) et "poolées" (utile en LOO).
2. **Courbes ROC** moyennes (avec bande de variance inter-folds).
3. **Sauvegarde des scores** de prédiction dans un CSV global `notebooks/eval_outputs/cv/all_models_predictions.csv` (toutes architectures confondues).
4. **Tests de Wilcoxon** :
   - stabilité inter-seeds (AUC par fold, seed la plus ancienne vs la plus récente) ;
   - séparation des classes (probabilités prédites, classe 1 vs classe 0) ;
   - comparaison inter-architectures (AUC appariée par seed/fold avec les autres modèles déjà évalués dans `outputs/cv_results/`).

Workflow type :

```bash
# 1. Entraîner chaque architecture (CV multi-seeds)
python src/train_cv.py model=abmil      cv.method=kfold cv.k=4 cv.n_seeds=6
python src/train_cv.py model=transmil   cv.method=kfold cv.k=4 cv.n_seeds=6
python src/train_cv.py model=clam_sb    cv.method=kfold cv.k=4 cv.n_seeds=6

ou python src/train_cv -m model=abmil,clam_sb, etc. 
# ...

# 2. Ouvrir les notebooks correspondants pour l'évaluation et les comparaisons
jupyter notebook notebooks/eval_cv_abmil.ipynb

# 3. Après avoir comparé les architectures, ré entrainer la meilleure architetcure sur une seed différente (42). Ce sera ce modèle qui sera utilisé pour les benchmarks et pour la KD

python src/train_cv.py model=abmil   

# 4. Pour la distillation, mettre le bon modèle teacher dans config.yaml et lancer 
python src/train_distillation.py -m model=abmil,transmil? ...

```

