"""Train ResNet50 using train and validation only; never reads the test split."""

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
import sklearn
from PIL import __version__ as pillow_version
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version

from common import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_DATA_ROOT,
    WEIGHTS,
    configure_phase,
    image_folder,
    make_model,
    set_seed,
    set_training_modes,
    train_transform,
    evaluation_transform,
)
from spec import verify_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True, help="New output directory outside the Git repository")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0, help="0 is robust on Windows")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, phase=None):
    training = optimizer is not None
    if training:
        set_training_modes(model, phase)
    else:
        model.eval()
    total_loss = 0.0
    true_labels, predicted_labels = [], []
    amp = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=amp)
        labels = labels.to(device, non_blocking=amp)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * len(labels)
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0),
    }


def save_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_curves(path: Path, history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric in zip(axes, ("loss", "accuracy", "macro_f1")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="train")
        axis.plot(epochs, [row[f"val_{metric}"] for row in history], label="validation")
        axis.set(xlabel="Epoch", ylabel=metric.replace("_", " ").title())
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if args.batch_size < 1 or args.head_epochs < 1 or args.fine_tune_epochs < 0 or args.patience < 1:
        raise ValueError("Batch size and head epochs must be positive; fine-tune epochs cannot be negative")
    data_root = args.data_root.resolve()
    run_dir = args.run_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if run_dir == repo_root or repo_root in run_dir.parents:
        raise ValueError("Choose --run-dir outside the Git repository so checkpoints are not accidentally staged")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new one: {run_dir}")
    set_seed(args.seed)
    print("Verifying training and validation files against the fixed split manifests...", flush=True)
    manifest_hashes = {split: verify_manifest(data_root, split) for split in ("train", "validation")}
    train_data = image_folder(data_root, "train", train_transform())
    val_data = image_folder(data_root, "validation", evaluation_transform())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; train={len(train_data)}; validation={len(val_data)}; classes={CLASS_TO_IDX}", flush=True)
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable. Training on CPU will be much slower.", flush=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Keep downloaded ImageNet weights outside the repository as well.
    weights_cache_dir = run_dir.parent / "torch-hub"
    torch.hub.set_dir(str(weights_cache_dir))
    # This is the only network download the training code may perform: official ImageNet weights.
    model = make_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir.mkdir(parents=True)
    config = {
        "data_root": str(data_root), "seed": args.seed, "batch_size": args.batch_size,
        "head_epochs": args.head_epochs, "fine_tune_epochs": args.fine_tune_epochs,
        "patience": args.patience, "head_lr": args.head_lr,
        "fine_tune_head_lr": args.fine_tune_head_lr, "backbone_lr": args.backbone_lr,
        "optimizer": "AdamW", "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau(val_loss, factor=0.5, patience=2)",
        "checkpoint_selection": "highest validation macro-F1; validation loss breaks ties",
        "training_augmentation": "PadToSquare, Resize(224), RandomRotation(10 degrees)",
        "evaluation_preprocessing": "PadToSquare, Resize(224), ImageNet normalization",
        "num_workers": args.num_workers, "weights": str(WEIGHTS),
        "classes": CLASS_TO_IDX, "train_count": len(train_data), "validation_count": len(val_data),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "weights_cache_dir": str(weights_cache_dir),
        "deterministic_algorithms_requested": True,
        "torch": torch.__version__, "torchvision": torchvision_version,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "pillow": pillow_version, "matplotlib": matplotlib.__version__,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "head_trainable_parameters": sum(parameter.numel() for parameter in model.fc.parameters()),
        "fine_tune_trainable_parameters": sum(parameter.numel() for parameter in model.fc.parameters()) + sum(parameter.numel() for parameter in model.layer4.parameters()),
        "manifest_sha256": manifest_hashes,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_f1, best_loss, best_epoch = -1.0, float("inf"), None
    started = time.perf_counter()
    current_epoch = 0
    phases = [("head", args.head_epochs), ("fine_tune", args.fine_tune_epochs)]
    for phase, max_epochs in phases:
        if max_epochs == 0:
            continue
        if phase == "fine_tune":
            # Start fine-tuning from the best head-only checkpoint, not its last epoch.
            saved = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(saved["model_state_dict"])
        configure_phase(model, phase)
        if phase == "head":
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.head_lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.layer4.parameters(), "lr": args.backbone_lr},
                    {"params": model.fc.parameters(), "lr": args.fine_tune_head_lr},
                ],
                weight_decay=1e-4,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        non_improving = 0
        phase_best_f1, phase_best_loss = -1.0, float("inf")
        for phase_epoch in range(1, max_epochs + 1):
            current_epoch += 1
            epoch_started = time.perf_counter()
            train_scores = run_epoch(model, train_loader, criterion, device, optimizer, scaler, phase)
            with torch.inference_mode():
                val_scores = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_scores["loss"])
            record = {
                "epoch": current_epoch, "phase": phase,
                **{f"train_{key}": value for key, value in train_scores.items()},
                **{f"val_{key}": value for key, value in val_scores.items()},
                "minutes": round((time.perf_counter() - epoch_started) / 60, 3),
                "head_lr": optimizer.param_groups[-1]["lr"],
                "backbone_lr": optimizer.param_groups[0]["lr"] if phase == "fine_tune" else 0.0,
            }
            history.append(record)
            save_history(run_dir / "history.csv", history)
            better = val_scores["macro_f1"] > best_f1 + 1e-6 or (
                abs(val_scores["macro_f1"] - best_f1) <= 1e-6 and val_scores["loss"] < best_loss
            )
            if better:
                best_f1, best_loss, best_epoch = val_scores["macro_f1"], val_scores["loss"], current_epoch
                checkpoint = {
                    "model_state_dict": model.state_dict(), "class_to_idx": CLASS_TO_IDX,
                    "weights": str(WEIGHTS), "epoch": current_epoch, "phase": phase,
                    "validation_macro_f1": best_f1, "validation_loss": best_loss,
                }
                temp_path = run_dir / "best.tmp.pt"
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, run_dir / "best.pt")
            phase_better = val_scores["macro_f1"] > phase_best_f1 + 1e-6 or (
                abs(val_scores["macro_f1"] - phase_best_f1) <= 1e-6 and val_scores["loss"] < phase_best_loss
            )
            if phase_better:
                phase_best_f1, phase_best_loss = val_scores["macro_f1"], val_scores["loss"]
                non_improving = 0
            else:
                non_improving += 1
            print(
                f"Epoch {current_epoch:02d} ({phase} {phase_epoch}/{max_epochs}) "
                f"train loss={train_scores['loss']:.4f} acc={train_scores['accuracy']:.4f} "
                f"val loss={val_scores['loss']:.4f} acc={val_scores['accuracy']:.4f} "
                f"macro-F1={val_scores['macro_f1']:.4f} {'*best*' if better else ''}",
                flush=True,
            )
            if non_improving >= args.patience:
                print(f"Early stopping {phase} after {non_improving} non-improving epochs", flush=True)
                break

    save_curves(run_dir / "learning_curves.png", history)
    summary = {
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "best_validation_loss": best_loss, "epochs_completed": current_epoch,
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 2),
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if device.type == "cuda" else None,
        "checkpoint": str(run_dir / "best.pt"),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("Training finished. The test split has not been read.", flush=True)


if __name__ == "__main__":
    main()
