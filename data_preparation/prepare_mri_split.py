"""Audit a brain MRI dataset and produce reproducible grouped split manifests.

No model is trained. The source dataset is never changed. The test set must be
reserved for final evaluation. Without patient IDs, patient independence cannot
be established. Requirements: numpy, scipy, Pillow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SPLITS = ["train", "validation", "test"]
RATIOS = np.array([0.70, 0.15, 0.15])
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PHASH_DISTANCE = 4
MAX_GRAY_RMSE = 7.65  # 3% of the 8-bit range; conservative similarity grouping


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect_one(item):
    root, path = item
    relative = path.relative_to(root)
    record = {"source_path": relative.as_posix(), "original_split": relative.parts[0],
              "class": path.parent.name, "class_id": CLASSES.index(path.parent.name),
              "marked_augmented": "-aug-" in path.name.lower(), "error": ""}
    try:
        record["sha256"] = digest(path.read_bytes())
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            record.update(width=image.width, height=image.height)
            record["pixel_sha256"] = digest(str(image.size).encode() + b"|" + image.tobytes())
            gray = image.convert("L")
            small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
            coeff = dctn(small, type=2, norm="ortho")[:8, :8]
            bits = (coeff > np.median(coeff)).reshape(-1)
            record["phash"] = int.from_bytes(np.packbits(bits).tobytes(), "big")
            thumb = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.uint8)
            return record, thumb
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, None


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(root, output, seed):
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError("No images found")
    for p in paths:
        if p.parent.name not in CLASSES or len(p.relative_to(root).parts) != 3:
            raise ValueError(f"Unexpected dataset path: {p}")
    records, thumbs = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (record, thumb) in enumerate(pool.map(inspect_one, ((root, p) for p in paths)), 1):
            records.append(record)
            thumbs.append(thumb)
            if i % 1000 == 0 or i == len(paths):
                print(f"Decoded and audited {i}/{len(paths)} images", flush=True)

    exclusions = []
    valid = []
    for i, row in enumerate(records):
        if row["error"]:
            exclusions.append({**row, "reason": "unreadable_image", "retained_source": ""})
        elif row["marked_augmented"]:
            exclusions.append({**row, "reason": "marked_augmented_no_parent_mapping", "retained_source": ""})
        else:
            valid.append(i)
    by_pixels = defaultdict(list)
    for i in valid:
        by_pixels[records[i]["pixel_sha256"]].append(i)
    retained = []
    for indices in by_pixels.values():
        labels = {records[i]["class"] for i in indices}
        if len(labels) > 1:
            for i in indices:
                exclusions.append({**records[i], "reason": "identical_pixels_conflicting_labels", "retained_source": ""})
            continue
        # Keep one deterministic representative; prefer an original Training path.
        ordered = sorted(indices, key=lambda i: (records[i]["original_split"] != "Training", records[i]["source_path"]))
        keep = ordered[0]
        retained.append(keep)
        for i in ordered[1:]:
            exclusions.append({**records[i], "reason": "duplicate_decoded_pixels", "retained_source": records[keep]["source_path"]})
    retained.sort(key=lambda i: records[i]["source_path"])
    selected = [records[i] for i in retained]
    gray = np.stack([thumbs[i] for i in retained])

    # Perceptual hashes identify candidates; RMSE then checks spatial similarity.
    # These are conservative similarity clusters, not inferred patient identities.
    n = len(selected)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    hashes = np.array([r["phash"] for r in selected], dtype=np.uint64)
    bit_counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    near_pairs = []
    for i in range(n):
        xor = np.bitwise_xor(hashes[i], hashes[i + 1:])
        distances = bit_counts[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        candidates = np.flatnonzero(distances <= PHASH_DISTANCE) + i + 1
        for j in candidates:
            rmse = float(np.sqrt(np.mean((gray[i].astype(np.float32) - gray[j].astype(np.float32)) ** 2)))
            if rmse <= MAX_GRAY_RMSE:
                union(i, int(j))
                near_pairs.append({"source_a": selected[i]["source_path"], "source_b": selected[j]["source_path"],
                                   "phash_distance": int(distances[j - i - 1]), "gray_rmse": round(rmse, 6)})
        if (i + 1) % 2000 == 0:
            print(f"Checked perceptual candidates for {i + 1}/{n} retained images", flush=True)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    # Quarantine entire cross-label high-similarity groups; never infer a diagnosis.
    # The two groups detected for this release were also visually inspected.
    conflict_keys = [key for key, indices in groups.items()
                     if len({selected[i]["class"] for i in indices}) > 1]
    for key in conflict_keys:
        for i in groups.pop(key):
            exclusions.append({**selected[i], "reason": "similar_images_conflicting_labels", "retained_source": ""})
    active_indices = {i for indices in groups.values() for i in indices}
    totals = np.bincount([selected[i]["class_id"] for i in active_indices], minlength=len(CLASSES))
    targets = np.zeros((3, len(CLASSES)), dtype=int)
    for c, total in enumerate(totals):
        ideal = RATIOS * total
        targets[:, c] = np.floor(ideal).astype(int)
        for split in np.argsort(-(ideal - targets[:, c]), kind="stable")[:int(total - targets[:, c].sum())]:
            targets[split, c] += 1
    rng = np.random.default_rng(seed)
    group_items = list(groups.values())
    rng.shuffle(group_items)
    group_items.sort(key=len, reverse=True)
    counts = np.zeros_like(targets)
    assignments = {}
    for indices in group_items:
        vector = np.bincount([selected[i]["class_id"] for i in indices], minlength=len(CLASSES))
        eligible = [s for s in range(3) if np.all(counts[s] + vector <= targets[s])]
        if not eligible:
            raise ValueError("Similarity groups cannot fit requested class targets; review groups before changing ratios")
        # Choose using remaining capacity. Singleton groups fill final exact targets.
        capacity = np.array([float((targets[s] - counts[s])[vector > 0].sum()) for s in eligible])
        split = int(rng.choice(eligible, p=capacity / capacity.sum()))
        counts[split] += vector
        group_id = "g_" + digest("\n".join(sorted(selected[i]["pixel_sha256"] for i in indices)).encode())[:20]
        for i in indices:
            assignments[i] = (SPLITS[split], group_id)

    manifests = []
    for i, row in enumerate(selected):
        if i not in assignments:
            continue
        split, group_id = assignments[i]
        filename = row["original_split"].lower() + "__" + Path(row["source_path"]).name
        manifests.append({**row, "split": split, "group_id": group_id,
                          "output_path": f"{split}/{row['class']}/{filename}"})
    manifest_by_source = {r["source_path"]: r for r in manifests}
    for pair in near_pairs:
        if pair["source_a"] in manifest_by_source and pair["source_b"] in manifest_by_source:
            pair["disposition"] = "grouped_same_split"
            assert manifest_by_source[pair["source_a"]]["split"] == manifest_by_source[pair["source_b"]]["split"]
        else:
            pair["disposition"] = "excluded_label_conflict"
    assert np.array_equal(counts, targets)
    assert len({r["pixel_sha256"] for r in manifests}) == len(manifests)
    assert len({r["output_path"] for r in manifests}) == len(manifests)

    fingerprint = digest("\n".join(f"{r['source_path']}|{r.get('sha256','ERROR')}" for r in records).encode())
    summary = {
        "seed": seed, "requested_ratios": dict(zip(SPLITS, RATIOS.tolist())),
        "source_images": len(records), "retained_images": len(manifests), "excluded_images": len(exclusions),
        "exclusion_reasons": dict(Counter(r["reason"] for r in exclusions)),
        "classes": CLASSES, "class_to_idx": dict(zip(CLASSES, range(len(CLASSES)))),
        "split_counts": {s: {c: int(counts[k, j]) for j, c in enumerate(CLASSES)} for k, s in enumerate(SPLITS)},
        "split_totals": {s: int(counts[k].sum()) for k, s in enumerate(SPLITS)},
        "similarity_pairs": len(near_pairs), "similarity_groups": sum(len(g) > 1 for g in groups.values()),
        "excluded_conflicting_similarity_groups": len(conflict_keys),
        "largest_similarity_group": max(map(len, groups.values())),
        "cross_class_similarity_groups": sum(len({selected[i]['class'] for i in g}) > 1 for g in groups.values()),
        "phash_max_hamming_distance": PHASH_DISTANCE, "gray64_max_rmse": MAX_GRAY_RMSE,
        "dataset_fingerprint_sha256": fingerprint,
        "protocol": "Fresh split pooling original Training and Testing; previous official split is not retained.",
        "limitations": ["No patient IDs: patient-disjoint evaluation cannot be verified.",
                        "Perceptual checks do not detect every transformed, re-encoded, or related image.",
                        "A test image is unseen only if excluded from training, tuning, and model selection; prior use cannot be undone.",
                        "Files marked aug are excluded because augmentation parent mappings are absent."]
    }
    output.mkdir(parents=True)
    fields = ["source_path", "output_path", "original_split", "split", "class", "class_id", "group_id", "sha256", "pixel_sha256", "width", "height", "phash"]
    write_csv(output / "all_splits.csv", manifests, fields)
    for split in SPLITS:
        write_csv(output / f"{split}.csv", [r for r in manifests if r["split"] == split], fields)
    write_csv(output / "excluded.csv", exclusions, ["source_path", "class", "original_split", "reason", "retained_source", "sha256", "pixel_sha256", "error"])
    write_csv(output / "similarity_pairs.csv", near_pairs, ["source_a", "source_b", "phash_distance", "gray_rmse", "disposition"])
    write_csv(output / "source_inventory.csv", records, ["source_path", "original_split", "class", "sha256", "pixel_sha256", "width", "height", "phash", "marked_augmented", "error"])
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def materialize(root, audit_dir, destination):
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {destination}")
    with (audit_dir / "all_splits.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty split manifest")
    source_names, target_names = set(), set()
    group_splits, seen_pixels = {}, set()
    for row in rows:
        source_rel, target_rel = Path(row["source_path"]), Path(row["output_path"])
        if source_rel.is_absolute() or target_rel.is_absolute() or ".." in source_rel.parts or ".." in target_rel.parts:
            raise ValueError("Manifest contains an unsafe path")
        if not (root / source_rel).resolve().is_relative_to(root) or not (destination / target_rel).resolve().is_relative_to(destination):
            raise ValueError("Manifest path escapes its root")
        if (row["split"] not in SPLITS or row["class"] not in CLASSES
                or len(target_rel.parts) != 3 or target_rel.parts[:2] != (row["split"], row["class"])
                or len(source_rel.parts) != 3 or source_rel.parts[1] != row["class"]):
            raise ValueError("Manifest path disagrees with split/class")
        if row["source_path"] in source_names or row["output_path"] in target_names or row["pixel_sha256"] in seen_pixels:
            raise ValueError("Duplicate source, destination, or decoded pixels in manifest")
        source_names.add(row["source_path"])
        target_names.add(row["output_path"])
        seen_pixels.add(row["pixel_sha256"])
        if group_splits.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("Similarity group crosses split boundaries")
    destination.mkdir(parents=True)
    for split in SPLITS:
        for label in CLASSES:
            (destination / split / label).mkdir(parents=True)
    for i, row in enumerate(rows, 1):
        source = root / row["source_path"]
        target = destination / row["output_path"]
        if digest(source.read_bytes()) != row["sha256"]:
            raise ValueError(f"Source changed since audit: {source}")
        shutil.copy2(source, target)
        if digest(target.read_bytes()) != row["sha256"]:
            raise ValueError(f"Copy verification failed: {target}")
        if i % 1000 == 0 or i == len(rows):
            print(f"Copied and verified {i}/{len(rows)} images", flush=True)
    shutil.copytree(audit_dir, destination / "manifests", ignore=shutil.ignore_patterns("*.png"))
    observed = Counter(p.relative_to(destination).parts[0] for p in destination.glob("*/*/*")
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
    expected = Counter(r["split"] for r in rows)
    if observed != expected:
        raise ValueError(f"Copied image counts mismatch: {observed} != {expected}")
    verification = {"all_copies_sha256_verified": True, "unique_decoded_pixels": True,
                    "no_similarity_group_crosses_splits": True, "counts": dict(observed), "total": len(rows)}
    (destination / "manifests" / "copy_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize", type=Path)
    args = parser.parse_args()
    if args.materialize:
        materialize(args.source.resolve(), args.audit_dir.resolve(), args.materialize.resolve())
    else:
        audit(args.source.resolve(), args.audit_dir.resolve(), args.seed)


if __name__ == "__main__":
    main()
