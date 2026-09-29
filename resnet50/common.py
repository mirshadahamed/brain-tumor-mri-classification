"""Shared, explicit data and model definitions for the ResNet50 experiment."""

from __future__ import annotations

import random
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

from spec import CLASS_NAMES, CLASS_TO_IDX, DEFAULT_DATA_ROOT, EXPECTED_COUNTS

WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
IMAGE_SIZE = 224


class PadToSquare:
    """Preserve the entire scan instead of center-cropping non-square images."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, side - width - left, side - height - top),
            fill=0,
        )


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_transform() -> transforms.Compose:
    # Mild geometric augmentation only. Do not augment validation or test data.
    return transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=WEIGHTS.transforms().mean, std=WEIGHTS.transforms().std),
        ]
    )


def evaluation_transform():
    # Same normalization as the selected ImageNet weights, with anatomy-preserving
    # padding instead of the weights' default center crop.
    return transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=WEIGHTS.transforms().mean, std=WEIGHTS.transforms().std),
        ]
    )


def image_folder(data_root: Path, split: str, transform):
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    folder = data_root / split
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing {split} folder: {folder}")
    dataset = datasets.ImageFolder(folder, transform=transform)
    if dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Incorrect class mapping in {folder}: {dataset.class_to_idx}; expected {CLASS_TO_IDX}"
        )
    if len(dataset) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"Unexpected {split} size: {len(dataset)}; expected {EXPECTED_COUNTS[split]}"
        )
    return dataset


def make_model(*, pretrained: bool) -> torch.nn.Module:
    model = resnet50(weights=WEIGHTS if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def configure_phase(model: torch.nn.Module, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if phase == "fine_tune":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    elif phase != "head":
        raise ValueError(f"Unknown phase: {phase}")


def set_training_modes(model: torch.nn.Module, phase: str) -> None:
    # Frozen BatchNorm layers must not update running statistics.
    model.eval()
    model.fc.train()
    if phase == "fine_tune":
        model.layer4.train()
