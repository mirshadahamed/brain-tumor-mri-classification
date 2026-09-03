# Brain Tumor MRI Classification — Custom CNN

Run the notebooks in VS Code with the repository `.venv` kernel. The CNN has
been checked for inference; full training and final evaluation are not yet implemented.

## Structure

```text
notebooks/                 Dataset inspection, pipeline check, CNN architecture
src/data_pipeline.py       Shared TensorFlow image loader
scripts/                   Shared split verification
data_preparation/         Imported ResNet50 preparation code and documentation
  manifests/               Fixed team split CSVs and cleaning audit
  UPSTREAM.json            Source branch and commit
dataset/                  Original images (unchanged)
models/                   Future saved CNN checkpoints
results/                  Future training curves and metrics
```

## Shared split

The ResNet50 manifests replace the previous local 85/15 training split.
They pool the original Training and Testing directories, clean duplicates and
other flagged images, and assign 4,673 training, 1,003 validation, and 1,001 test
images. See `data_preparation/README.md` for the cleaning protocol and limitations.
Class IDs: glioma=0, meningioma=1, notumor=2, pituitary=3.

The loader uses `source_path` to read images from `dataset/`; no duplicate image
folders are necessary. Imported manifests are unchanged. Do not regenerate them.
All compared models must train afresh on these memberships. Similarity groups
are not verified patient identifiers.

## Run

```bash
source .venv/bin/activate
python scripts/verify_shared_split.py
```

Open and run these notebooks in order:
1. `notebooks/inspect_dataset.ipynb`
2. `notebooks/data_pipeline.ipynb`
3. `notebooks/model.ipynb`

Notebook paths work from the repository root or `notebooks/`. The pipeline
notebook imports `src/data_pipeline.py` so there is one preprocessing implementation.
Input images become RGB, 224×224, normalized to [0,1]. Augmentation lives inside
the model and runs only during training. Test data must not guide model selection.

Install the recorded environment dependencies using `pip install -r requirements.txt`.
The separate preparation requirements are needed only to reproduce the upstream audit.
