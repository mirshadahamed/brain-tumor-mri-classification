"""Custom CNN architecture for the shared four-class MRI experiment."""

import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42


def build_model():
    """Build a fresh CNN for normalized RGB images and integer class labels.

    Outputs follow the shared manifest class IDs: glioma, meningioma,
    notumor, pituitary. Augmentation and dropout are training-only.
    """
    inputs = layers.Input(shape=(224, 224, 3), name="image")

    # Input images are already normalized by data_pipeline.py.
    # RandomRotation uses fractions of a full turn: 15 / 360.
    x = layers.RandomRotation(
        factor=15 / 360,
        fill_mode="reflect",
        seed=SEED,
    )(inputs)
    x = layers.RandomZoom(
        height_factor=0.1,
        width_factor=0.1,
        fill_mode="reflect",
        seed=SEED + 1,
    )(x)
    x = layers.RandomFlip(
        mode="horizontal",
        seed=SEED + 2,
    )(x)

    # Blocks 1–3: two convolutions per block.
    for block, filters, dropout in [
        (1, 32, 0.25),
        (2, 64, 0.25),
        (3, 128, 0.30),
    ]:
        x = layers.Conv2D(
            filters, 3,
            padding="same",
            activation="relu",
            name=f"block{block}_conv1",
        )(x)
        x = layers.BatchNormalization(
            name=f"block{block}_bn",
        )(x)
        x = layers.Conv2D(
            filters, 3,
            padding="same",
            activation="relu",
            name=f"block{block}_conv2",
        )(x)
        x = layers.MaxPooling2D(
            pool_size=2,
            name=f"block{block}_pool",
        )(x)
        x = layers.Dropout(
            dropout,
            name=f"block{block}_dropout",
        )(x)

    # Block 4: one convolution.
    x = layers.Conv2D(
        256, 3,
        padding="same",
        activation="relu",
        name="block4_conv",
    )(x)
    x = layers.BatchNormalization(name="block4_bn")(x)
    x = layers.MaxPooling2D(pool_size=2, name="block4_pool")(x)
    x = layers.Dropout(0.30, name="block4_dropout")(x)

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.50, name="dense_dropout")(x)
    outputs = layers.Dense(
        4,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = models.Model(inputs, outputs, name="custom_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model

