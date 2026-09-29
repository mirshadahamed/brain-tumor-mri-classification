"""Dataset contract shared by the ResNet50 scripts (no ML dependencies)."""

import csv
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "Dataset-Split"
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
EXPECTED_COUNTS = {"train": 4673, "validation": 1003, "test": 1001}


def verify_manifest(data_root: Path, split: str) -> str:
    """Check exact membership and file hashes for one split; no other split is opened."""
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    manifest = REPO_ROOT / "data_preparation" / "manifests" / f"{split}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Required split manifest is missing: {manifest}")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(f"Manifest has {len(rows)} {split} rows, expected {EXPECTED_COUNTS[split]}")
    expected_paths = set()
    for row in rows:
        if row["split"] != split or row["class_id"] != str(CLASS_TO_IDX[row["class"]]):
            raise ValueError(f"Invalid split or class mapping in {manifest}: {row['output_path']}")
        relative_path = Path(row["output_path"])
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 3
            or relative_path.parts[0] != split
            or relative_path.parts[1] != row["class"]
        ):
            raise ValueError(f"Unsafe or unexpected manifest path: {relative_path}")
        expected_paths.add(relative_path.as_posix())
        file_path = data_root / relative_path
        digest = hashlib.sha256()
        with file_path.open("rb") as image_stream:
            for chunk in iter(lambda: image_stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Image hash differs from split manifest: {file_path}")
    if len(expected_paths) != len(rows):
        raise ValueError(f"Duplicate output paths in {split} manifest")
    actual_paths = {
        path.relative_to(data_root).as_posix()
        for class_name in CLASS_NAMES
        for path in (data_root / split / class_name).glob("*.jpg")
    }
    if expected_paths != actual_paths:
        raise ValueError(f"Image membership differs from {split} manifest")
    return manifest_digest
