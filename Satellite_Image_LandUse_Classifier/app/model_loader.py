"""import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models import ResNet18Classifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


def load_model():

    model = ResNet18Classifier(len(CLASSES))

    checkpoint = PROJECT_ROOT / "checkpoints" / "resnet18_best.pt"

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch

from src.models import ResNet18Classifier

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


def load_model():

    model = ResNet18Classifier(10)

    checkpoint = PROJECT_ROOT / "checkpoints" / "resnet18_best.pt"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model