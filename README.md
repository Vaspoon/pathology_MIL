# pathology_MIL
Personal Repository to test multiple instance learning models on pathology slides images

## Content
MIL training pipeline for following papers:
* ABMIL

## How to 
### ABMIL
python src/main.py model=abmil

### Modèle original inchangé
python src/main.py model=mil_max

### Heatmaps sur le val set
python src/main.py mode=infer model=abmil checkpoint_path=outputs/.../best.pt

### Grid search
python src/main.py --multirun model=abmil training.lr=1e-4,5e-5




