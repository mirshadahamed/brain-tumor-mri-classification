"""One final, held-out test evaluation of an already-selected ResNet50 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "resnet50-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from common import CLASS_NAMES, CLASS_TO_IDX, DEFAULT_DATA_ROOT, evaluation_transform, image_folder, make_model
from spec import verify_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory produced by train.py")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def plot_confusion(path: Path, matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted class", ylabel="True class", title="Held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            count = matrix[row, column]
            axis.text(column, row, str(count), ha="center", va="center", color="white" if count > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            f"Final test metrics already exist at {metrics_path}. Do not repeatedly evaluate/tune against the test set."
        )
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No selected best checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError("Checkpoint class mapping does not match this dataset")
    model = make_model(pretrained=False)
    model.load_state_dict(saved["model_state_dict"])
    model = model.to(device)
    model.eval()
    test_manifest_sha256 = verify_manifest(args.data_root.resolve(), "test")
    test_data = image_folder(args.data_root.resolve(), "test", evaluation_transform())
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    true_labels, predicted_labels, probabilities = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            batch_probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(batch_probabilities.argmax(axis=1).tolist())
            probabilities.extend(batch_probabilities.tolist())

    y_true = np.asarray(true_labels)
    y_pred = np.asarray(predicted_labels)
    y_prob = np.asarray(probabilities)
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = None
    metrics = {
        "split": "test", "test_count": len(test_data),
        "selected_epoch": saved["epoch"], "selected_phase": saved["phase"],
        "selection_validation_macro_f1": saved["validation_macro_f1"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_roc_auc_ovr": macro_auc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "test_manifest_sha256": test_manifest_sha256,
        "inference_seconds": round(time.perf_counter() - started, 2),
    }
    plot_confusion(run_dir / "test_confusion_matrix.png", matrix)
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_file", "true_class", "predicted_class", "predicted_confidence", *[f"prob_{name}" for name in CLASS_NAMES]])
        for index, (filename, _) in enumerate(test_data.samples):
            writer.writerow([
                str(Path(filename).relative_to(args.data_root.resolve())),
                CLASS_NAMES[y_true[index]], CLASS_NAMES[y_pred[index]],
                float(y_prob[index, y_pred[index]]), *y_prob[index].tolist(),
            ])
    # The existence of this file means all final artifacts completed successfully.
    temp_path = run_dir / "test_metrics.tmp.json"
    temp_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    os.replace(temp_path, metrics_path)
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"classification_report", "confusion_matrix"}}, indent=2), flush=True)
    print(f"Final test results written to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
