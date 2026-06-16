"""
src/utils/build.py

Registre central des modèles/trainers + helpers de construction,
partagés entre main.py et train_cv.py.

   multimodal=False  →  H5Dataset (hes only),       forward(hes)
   multimodal=True   →  DatasetMultiModal (hes+ihc), forward(hes, ihc)

Trainer classes:
   Trainer                  — returns scalar logit (mil_max / mil_mean)
   TrainerABMIL             — returns (logits, attn), full metrics
   TrainerCLAM              — returns (logits, attn, instance_loss)
   TrainerMultiModalABMIL   — returns (logits, attn), full metrics, multimodal batches
   TrainerContrastiveMultiModalABMIL — multimodal + contrastive losses
"""

import inspect

from omegaconf import OmegaConf

from models.abmil import ABMIL
from models.attention_mil import AttentionMIL_Papagoras
from models.clam import CLAM_SB, CLAM_MB
from models.contrastive_multimodal_mil import ContrastiveMultiModalMIL
from models.mil_max_pooling import MILMaxPooling
from models.mil_mean_pooling import MILMeanPooling
from models.multimodal_porpoise import MultiModalMILPorpoise
from models.multimodal_mil import MultiModalMILGated
from models.transmil import TransMIL
from training.trainer import Trainer, TrainerABMIL, TrainerMultiModalABMIL
from training.trainer_clam import TrainerCLAM
from training.trainer_contrastive import TrainerContrastiveMultiModalABMIL

MODEL_REGISTRY = {
    "mil_max":              {"cls": MILMaxPooling,          "trainer": Trainer,                "multimodal": False},
    "mil_mean":             {"cls": MILMeanPooling,         "trainer": Trainer,                "multimodal": False},
    "attention_mil":        {"cls": AttentionMIL_Papagoras, "trainer": TrainerABMIL,           "multimodal": False},
    "abmil":                {"cls": ABMIL,                  "trainer": TrainerABMIL,           "multimodal": False},
    "transmil":             {"cls": TransMIL,               "trainer": TrainerABMIL,           "multimodal": False},
    "clam_sb":              {"cls": CLAM_SB,                "trainer": TrainerCLAM,            "multimodal": False},
    "clam_mb":              {"cls": CLAM_MB,                "trainer": TrainerCLAM,            "multimodal": False},
    "multimodal_porpoise":     {"cls": MultiModalMILPorpoise,     "trainer": TrainerMultiModalABMIL,           "multimodal": True},
    "multimodal_gated":        {"cls": MultiModalMILGated,        "trainer": TrainerMultiModalABMIL,           "multimodal": True},
    "contrastive_multimodal":  {"cls": ContrastiveMultiModalMIL,  "trainer": TrainerContrastiveMultiModalABMIL, "multimodal": True},
}

# Trainer classes that expose the full fit() loop (early stopping, AUC, checkpoints)
ABMIL_FAMILY = (TrainerABMIL, TrainerCLAM, TrainerMultiModalABMIL, TrainerContrastiveMultiModalABMIL)


def build_model(cfg_model, device):
    """Instantiate model from registry, passing only params accepted by __init__."""
    name = cfg_model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    valid  = set(inspect.signature(entry["cls"].__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg_model).items() if k in valid}
    return entry["cls"](**kwargs).to(device), entry["trainer"], entry["multimodal"]


def build_trainer(TrainerClass, model, optimizer, device, writer, cfg_training):
    """Instantiate trainer, forwarding only the kwargs it declares in __init__."""
    valid        = set(inspect.signature(TrainerClass.__init__).parameters) - {"self"}
    training_cfg = OmegaConf.to_container(cfg_training, resolve=True)
    kwargs = {"model": model, "optimizer": optimizer, "device": device, "writer": writer}
    # Pass any matching training-config keys accepted by this trainer class
    for key in (
        "early_stopping", "early_stopping_patience", "monitor",
        "lambda_smd", "lambda_con", "contrastive_batch_size",
        "bag_weight",
    ):
        if key in valid and key in training_cfg:
            kwargs[key] = training_cfg[key]
    return TrainerClass(**kwargs)


# Backwards-compatible aliases (used by main.py)
_build_model   = build_model
_build_trainer = build_trainer
