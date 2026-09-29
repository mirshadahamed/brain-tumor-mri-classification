"""Run a research-only prediction on one image using the selected checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from common import CLASS_NAMES, CLASS_TO_IDX, evaluation_transform, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError("Checkpoint class mapping does not match this project")
    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with Image.open(args.image) as image:
        tensor = evaluation_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
    result = {
        "predicted_class": CLASS_NAMES[max(range(len(CLASS_NAMES)), key=lambda index: probabilities[index])],
        "probabilities": {name: probabilities[index] for index, name in enumerate(CLASS_NAMES)},
        "warning": "Research demonstration only; not for clinical diagnosis.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
