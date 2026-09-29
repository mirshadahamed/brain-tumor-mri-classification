"""Final evaluation of a frozen checkpoint; never trains or tunes the model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc, log_loss)
from PIL import Image
from src.data_pipeline import ROOT, make_dataset


def evaluate_run(run_id='cnn_20260928T045158815871Z'):
    output = ROOT / 'results' / run_id / 'evaluation'
    # Reuse completed results instead of silently evaluating again.
    if (output / 'metrics.json').exists():
        return json.loads((output / 'metrics.json').read_text()), output
    checkpoint = ROOT / 'models' / run_id / 'best.keras'
    manifest = ROOT / 'data_preparation/manifests/test.csv'
    frame = pd.read_csv(manifest)
    config = json.loads((ROOT / 'results' / run_id / 'config.json').read_text())
    mapping = config['class_to_index']
    classes = sorted(mapping, key=mapping.get)
    assert [mapping[c] for c in classes] == list(range(len(classes)))
    assert all(mapping[c] == i for c, i in zip(frame['class'], frame['class_id']))
    model = tf.keras.models.load_model(checkpoint, compile=False)
    probabilities = model.predict(make_dataset('test'), verbose=1)
    assert probabilities.shape == (len(frame), len(classes))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    truth = frame['class_id'].to_numpy()
    predicted = probabilities.argmax(axis=1)
    report = classification_report(truth, predicted, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        'test_images': len(frame), 'accuracy': accuracy_score(truth, predicted),
        'cross_entropy': log_loss(truth, probabilities, labels=list(range(len(classes)))),
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'],
        'macro_roc_auc_ovr': roc_auc_score(truth, probabilities, multi_class='ovr', average='macro'),
        'checkpoint': str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'test_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'class_to_index': mapping,
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(output / 'classification_report.csv')
    predictions = frame[['source_path', 'class', 'class_id']].copy()
    predictions['predicted_class'] = [classes[i] for i in predicted]
    predictions['confidence'] = probabilities.max(axis=1)
    for i, name in enumerate(classes):
        predictions[f'probability_{name}'] = probabilities[:, i]
    predictions.to_csv(output / 'predictions.csv', index=False)
    errors = predictions.loc[predicted != truth]
    errors.to_csv(output / 'misclassified.csv', index=False)
    metrics['misclassified_images'] = len(errors)
    import matplotlib.pyplot as plt
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(classes))))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues'); fig.colorbar(im, ax=ax)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted class', ylabel='True class', title='Custom CNN — final test set')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(output / 'confusion_matrix.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(classes):
        fpr, tpr, _ = roc_curve(truth == i, probabilities[:, i])
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='One-vs-rest ROC curves')
    ax.legend(); fig.tight_layout(); fig.savefig(output / 'roc_curves.png', dpi=150); plt.close(fig)
    if len(errors):
        fig, axes = plt.subplots(3, 4, figsize=(12, 10))
        for ax in axes.flat: ax.axis('off')
        for ax, (_, row) in zip(axes.flat, errors.head(12).iterrows()):
            with Image.open(ROOT / 'dataset' / row['source_path']) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(f"True: {row['class']}\nPred: {row['predicted_class']} ({row['confidence']:.1%})")
        fig.suptitle('First 12 misclassified images in manifest order')
        fig.tight_layout(); fig.savefig(output / 'misclassified_examples.png', dpi=150); plt.close(fig)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics, output
