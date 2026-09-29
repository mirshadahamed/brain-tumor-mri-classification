# ResNet50 brain MRI classification

This is the ResNet50 experiment for the four-class SE4050 group comparison. It uses
the fixed `Dataset-Split` folders prepared by the group: 4,673 train, 1,003
validation, and 1,001 test images. Class IDs are `glioma=0`, `meningioma=1`,
`notumor=2`, `pituitary=3`. The split is never regenerated here.

## Method

- Start from Torchvision ResNet50 ImageNet-1K V2 weights, with a new four-output
  classification layer. This is transfer learning, not a model trained from scratch.
- Pad to square (preserving the full MRI), resize to 224x224, and use the
  ImageNet weight's mean/std normalization. Mild rotation augmentation applies
  to **training only**; validation and test transforms are deterministic.
  ImageFolder converts all inputs to RGB.
- Train the head with the backbone frozen, then fine-tune `layer4` and the head at
  lower learning rates. Select the best checkpoint by validation macro-F1 (loss
  breaks ties), use early stopping, and log each epoch.
- `train.py` opens only `train` and `validation`. `evaluate.py` opens `test` only
  after a checkpoint has been selected, and refuses to overwrite existing final
  test metrics. Do not use final test results to revise hyperparameters.
- Report accuracy, balanced accuracy, macro/weighted F1, per-class precision,
  recall and F1, one-vs-rest macro ROC-AUC, confusion matrix, runtime, and curves.

The source is [Masoud Nickparvar's Brain Tumor MRI Dataset, version 2](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset),
listed by Kaggle under CC BY 4.0. The group's cleaning and resplitting decisions
are documented under `data_preparation/`. These data do not provide patient identifiers, so the test
set is image-level holdout, not proven unseen-patient performance. This is a
research/assignment model and must not be used for clinical diagnosis.

The [Aeryes ResNet50 project](https://github.com/Aeryes/BrainTumorClassification)
informed the broad choice of transfer learning and validation monitoring, but
uses a different dataset. Its reported numbers are not comparable to this split.
Pretrained weights come from Torchvision's `ResNet50_Weights.IMAGENET1K_V2`.

## Run on Windows

Use Python 3.12 and a separate environment. Install compatible CUDA-enabled
`torch` and `torchvision` from the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then install the remaining packages in `requirements.txt`. Keep the virtual
environment and model outputs **outside** this repository; model checkpoints
can be large and should not be added accidentally.

From the repository root, after activating that environment:

```powershell
python resnet50/inspect_data.py --output-dir "C:\path\outside\repo\resnet50-eda"
python resnet50/train.py --run-dir "C:\path\outside\repo\resnet50-run-01"
```

This writes `config.json`, `history.csv`, `learning_curves.png`, `best.pt`, and
`training_summary.json`. First review validation results and freeze all model
choices. Then perform the one final test evaluation:

```powershell
python resnet50/evaluate.py --run-dir "C:\path\outside\repo\resnet50-run-01"
```

The final evaluation writes `test_metrics.json`, `test_confusion_matrix.png`, and
`test_predictions.csv`. Compare these results with the group's other architectures
using exactly the same split, label mapping, and metrics.

For a demonstration with an image that is **not** used to tune the model:

```powershell
python resnet50/predict.py --checkpoint "C:\path\outside\repo\resnet50-run-01\best.pt" --image "C:\path\to\scan.jpg"
```

The output is an experimental class prediction, not a medical diagnosis.

The code and documentation were developed with AI assistance. Verify all
claims and acknowledge this support in the assignment report as required.
