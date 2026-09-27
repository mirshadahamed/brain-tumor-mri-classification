# Shared brain MRI dataset split

Prepared on 2026-09-26 with seed 42. All group members and all four models must
use these same split manifests and class mapping.

## Locations

- Original, unchanged data: `../Dataset/`
- Prepared image folders: `../Dataset-Split/train/`, `validation/`, and `test/`
- Exact membership and audit: this directory's `manifests/`

This is a **fresh 70/15/15 split** formed by pooling the original Training and
Testing folders after quality checks. It is not the publisher's original split.
Do not compare scores directly with studies using a different split.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| glioma | 1,244 | 267 | 266 | 1,777 |
| meningioma | 1,039 | 223 | 223 | 1,485 |
| notumor | 1,160 | 249 | 248 | 1,657 |
| pituitary | 1,230 | 264 | 264 | 1,758 |
| **Total** | **4,673** | **1,003** | **1,001** | **6,677** |

The percentages are approximately 70/15/15, with integer rounding within each
class. Class IDs are glioma=0, meningioma=1, notumor=2, pituitary=3.

## Cleaning and leakage checks

All 7,200 original images were decoded successfully. The new dataset excludes:

- 203 files whose names contain `-aug-`, because original-parent mappings are
  unavailable and splitting pre-augmented examples can cause leakage.
- 314 redundant images with identical decoded RGB pixels and dimensions, after
  excluding the marked augmented files. One representative is retained.
- 6 images from two groups of highly similar scans carrying conflicting glioma
  and meningioma labels. These pairs were visually inspected. The labels were
  not guessed or changed; the files remain in the original dataset for review.

Every excluded file and reason is recorded in `manifests/excluded.csv`.
No original file was deleted or modified.

Perceptual candidate pairs use a 64-bit DCT hash with Hamming distance <=4 and
64x64 grayscale RMSE <=7.65 on the 0-255 scale. The resulting connected
components stay together in one split. After excluding the two conflicting
groups, 388 groups contain multiple images. Similarity groups are conservative
grouping decisions, not verified patient IDs. Thresholds were fixed before any
model fitting and did not use validation/test model performance.

Checks confirm unique decoded image content across retained images and zero
detected similarity groups spanning splits. All copied files are checked against
their source SHA256 hashes. See `copy_verification.json` in the prepared dataset's
`manifests/` folder for copy verification.

## Keeping the test set unseen

1. Fit the network and any learned preprocessing statistics using `train` only.
2. Apply random augmentation to `train` only, after the split.
3. Use `validation` for early stopping, learning-rate choices, architecture
   settings, thresholds, and choosing the best checkpoint.
4. Freeze those decisions before evaluating `test` for the final report.
5. Do not choose models/settings from test metrics or regenerate the split after
   seeing test results. Share these fixed manifests with the whole team.
6. Train all compared models afresh with this split. Checkpoints trained on the
   old Training folder may already have seen some images in the new test set.

ImageNet-pretrained weights may initialize the models; they do not replace
training and validation using these new split memberships.

The dataset has no patient identifiers. This process cannot prove that patients
are independent across splits, or detect every modified copy or adjacent slice.
Image-level holdout results must not be described as verified performance on
unseen patients. File organization also cannot undo any prior use of test images
in model development.

## Reproduce

Use Python 3.10+ and the dependencies in `requirements.txt` (the original run
used Python 3.13.5). Run these commands from the repository root:

```powershell
python -m pip install -r data_preparation/requirements.txt
python data_preparation/prepare_mri_split.py --source "../Dataset" --audit-dir "data_preparation/reproduced_manifests" --seed 42
python data_preparation/prepare_mri_split.py --source "../Dataset" --audit-dir "data_preparation/reproduced_manifests" --materialize "../Dataset-Split-Reproduced"
```

Both output locations must be new. The script refuses to overwrite an existing
directory. Reproduction must produce the same split membership; it is not an
opportunity to try a new seed after training.

The input dataset fingerprint (relative filenames plus raw file SHA256 hashes):

`05c99c7f9a588f4441bb567be5330c24bf001a1f27ef55fca9c18d3b96f35e61`

## Files to share through Git

Commit this preparation script, README, dependency information, and manifests.
Keep the image folders and model checkpoints outside the repository. A teammate
with the same original dataset can reproduce the prepared folders using the
script and verify membership against the shared manifests.

Dataset source: [Masoud Nickparvar's Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
These cleaning and split rules should be described in the assignment report.
Preparation code and documentation were created with AI assistance; acknowledge
that assistance as required by the assignment.
