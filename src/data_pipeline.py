from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_image(path):
    """Pillow handles grayscale, RGB, RGBA, and palette images."""
    filepath = path.numpy().decode("utf-8")

    with Image.open(filepath) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def prepare_example(path, label):
    image = tf.py_function(
        func=load_image,
        inp=[path],
        Tout=tf.float32,
    )
    image.set_shape((*IMAGE_SIZE, 3))
    return image, tf.cast(label, tf.int32)


def make_dataset(split, training=False):
    frame = pd.read_csv(ROOT / "data_preparation" / "manifests" / f"{split}.csv")

    paths = [
        str(ROOT / "dataset" / filepath)
        for filepath in frame["source_path"]
    ]
    labels = frame["class_id"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        prepare_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)

    train_dataset = make_dataset("train", training=True)
    validation_dataset = make_dataset("validation")

    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel minimum:", float(tf.reduce_min(images)))
        print("Pixel maximum:", float(tf.reduce_max(images)))
        print("Labels:", labels.numpy())

    print("Training batches:", int(train_dataset.cardinality()))
    print("Validation batches:", int(validation_dataset.cardinality()))
    print("Pipeline check completed.")