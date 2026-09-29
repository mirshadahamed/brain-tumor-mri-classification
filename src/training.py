"""Train a fresh CNN using only the fixed training and validation manifests."""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from src.data_pipeline import ROOT, BATCH_SIZE, make_dataset
from src.model import SEED, build_model


def train_cnn(smoke_test=True, epochs=50):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    run_id = ('smoke_' if smoke_test else 'cnn_') + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'results' / run_id
    checkpoint_dir = ROOT / 'models' / run_id
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_dir / 'best.keras'
    train = make_dataset('train', training=True)
    validation = make_dataset('validation')
    if smoke_test:
        train, validation = train.take(2), validation.take(2)
    model = build_model()
    manifests = ROOT / 'data_preparation' / 'manifests'
    config = {
        'smoke_test': smoke_test, 'seed': SEED, 'batch_size': BATCH_SIZE,
        'maximum_epochs': 1 if smoke_test else epochs,
        'optimizer': 'Adam', 'initial_learning_rate': 0.001,
        'loss': 'SparseCategoricalCrossentropy', 'monitor': 'val_loss',
        'early_stopping_patience': 8, 'reduce_lr_patience': 3,
        'reduce_lr_factor': 0.5, 'minimum_learning_rate': 1e-6,
        'tensorflow': tf.__version__, 'python': platform.python_version(),
        'platform': platform.platform(),
        'devices': [str(d) for d in tf.config.list_physical_devices()],
        'class_to_index': json.loads((manifests / 'split_summary.json').read_text())['class_to_idx'],
        'manifest_sha256': {s: hashlib.sha256((manifests / f'{s}.csv').read_bytes()).hexdigest() for s in ['train', 'validation']},
        'model_source_sha256': hashlib.sha256((ROOT / 'src/model.py').read_bytes()).hexdigest(),
    }
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    with (output / 'model_summary.txt').open('w') as stream:
        model.summary(print_fn=lambda line: stream.write(line + '\n'))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(output / 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    history = model.fit(train, validation_data=validation, epochs=config['maximum_epochs'], callbacks=callbacks)
    elapsed = time.perf_counter() - start
    values = {key: [float(v) for v in series] for key, series in history.history.items()}
    if not all(np.isfinite(series).all() for series in values.values()):
        raise RuntimeError('Non-finite training metrics; inspect the run before continuing.')
    (output / 'history.json').write_text(json.dumps(values, indent=2))
    # Reload the selected checkpoint, not merely the last epoch's model.
    best = tf.keras.models.load_model(checkpoint)
    images, _ = next(iter(validation))
    probabilities = best(images[:2], training=False)
    tf.debugging.assert_all_finite(probabilities, 'Invalid checkpoint predictions')
    tf.debugging.assert_near(tf.reduce_sum(probabilities, axis=1), tf.ones(2), atol=1e-5)
    summary = {'training_seconds': elapsed, 'epochs_completed': len(values['loss']),
               'best_epoch': int(np.argmin(values['val_loss'])) + 1,
               'best_validation_loss': min(values['val_loss']),
               'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_reload_passed': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ['accuracy', 'loss']):
        for prefix, label in [('', 'Training'), ('val_', 'Validation')]:
            axis.plot(range(1, len(values[metric]) + 1), values[prefix + metric], label=label)
        axis.set(xlabel='Epoch', ylabel=metric.capitalize())
        axis.legend(); axis.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=150); plt.close(fig)
    print(f'Completed in {elapsed:.1f}s. Results: {output}')
    print('Checkpoint reload passed. Test set was not used.')
    return {'model': best, 'history': values, 'output_dir': output, 'checkpoint': checkpoint, 'summary': summary}
