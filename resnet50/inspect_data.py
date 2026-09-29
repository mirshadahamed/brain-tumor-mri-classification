"""Train/validation-only exploratory checks before fitting ResNet50."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "resnet50-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from spec import CLASS_NAMES, DEFAULT_DATA_ROOT, EXPECTED_COUNTS


def split_samples(data_root: Path, split: str):
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    samples = []
    for index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)
        samples.extend((path, index) for path in sorted(class_dir.glob("*.jpg")))
    if len(samples) != EXPECTED_COUNTS[split]:
        raise ValueError(f"Unexpected {split} count: {len(samples)}")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise ValueError("Choose --output-dir outside the Git repository")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    datasets = {split: split_samples(data_root, split) for split in ("train", "validation")}
    summary = {}
    for split, samples in datasets.items():
        label_counts = Counter(label for _, label in samples)
        widths, heights = [], []
        for file_path, _ in samples:
            with Image.open(file_path) as image:
                image.load()
                widths.append(image.width)
                heights.append(image.height)
        summary[split] = {
            "images": len(samples),
            "class_counts": {name: label_counts[index] for index, name in enumerate(CLASS_NAMES)},
            "width_range": [min(widths), max(widths)],
            "height_range": [min(heights), max(heights)],
        }
    (output_dir / "train_validation_eda.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 4, figsize=(13, 7))
    for row, split in enumerate(("train", "validation")):
        samples = datasets[split]
        for column, class_name in enumerate(CLASS_NAMES):
            file_path = next(path for path, label in samples if label == column)
            with Image.open(file_path) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].set_title(f"{split}: {class_name}")
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "train_validation_samples.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    x = range(len(CLASS_NAMES))
    axis.bar([value - 0.2 for value in x], list(summary["train"]["class_counts"].values()), width=0.4, label="train")
    axis.bar([value + 0.2 for value in x], list(summary["validation"]["class_counts"].values()), width=0.4, label="validation")
    axis.set(xticks=list(x), xticklabels=CLASS_NAMES, ylabel="Images", title="Train and validation class distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "train_validation_distribution.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print("The test images and labels were not opened.")


if __name__ == "__main__":
    main()
