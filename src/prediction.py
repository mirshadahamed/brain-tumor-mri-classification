"""Single-image inference with the frozen CNN and training preprocessing."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from src.data_pipeline import ROOT, IMAGE_SIZE

DEFAULT_RUN = 'cnn_20260928T045158815871Z'


def predict_image(image_path, run_id=DEFAULT_RUN):
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path}')
    checkpoint = ROOT / 'models' / run_id / 'best.keras'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint}')
    config = json.loads((ROOT / 'results' / run_id / 'config.json').read_text())
    mapping = config['class_to_index']
    classes = sorted(mapping, key=mapping.get)
    with Image.open(path) as image:
        display_image = image.convert('RGB')
        resized = display_image.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        batch = np.asarray(resized, dtype=np.float32)[None, ...] / 255.0
    model = tf.keras.models.load_model(checkpoint, compile=False)
    probabilities = model(batch, training=False).numpy()[0]
    if probabilities.shape != (len(classes),) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid model output')
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-5)
    winner = int(probabilities.argmax())
    return {
        'image': display_image,
        'predicted_class': classes[winner],
        'probabilities': dict(zip(classes, map(float, probabilities))),
        'image_path': str(path),
    }
