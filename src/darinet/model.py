"""DARINet: dual self-attention modified residual-inception network for plant disease.

Ported from ``notebooks/Proposed_Plant_Disease.ipynb`` (Cassava). The Rice and
Maize notebooks use the identical network with a different number of output
classes.
"""

from __future__ import annotations

from keras import Model, layers

# Inception branch widths: [1x1, 1x1->3x3 reduce, 3x3, 1x1->3x3 reduce, 3x3, 3x3, pool-proj 1x1]
INCEPTION_FILTERS = (16, 32, 64, 32, 64, 128, 32)


def residual_channel_attention(x, gamma: int = 2):
    """Squeeze-and-excitation style channel attention with a residual add."""
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(channels // gamma, activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Add()([layers.Multiply()([x, s]), x])


def residual_spatial_attention(x, reduction_ratio: int = 16):
    """Spatial attention map from 1x1 convs, applied multiplicatively with a residual add."""
    channels = x.shape[-1]
    h = layers.Conv2D(channels // reduction_ratio, 1, padding="same", activation="relu")(x)
    h = layers.Conv2D(channels // reduction_ratio, 1, padding="same", activation="relu")(h)
    h = layers.Conv2D(channels, 1, padding="same", activation="relu")(h)
    attention_map = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(h)
    return layers.Add()([layers.Multiply()([attention_map, x]), x])


def inception_block(x, filters: tuple[int, ...] = INCEPTION_FILTERS):
    """Modified inception block with depthwise-separable 3x3 convolutions."""
    p1 = layers.Conv2D(filters[0], 1, padding="same", activation="relu")(x)

    p2 = layers.Conv2D(filters[1], 1, padding="same", activation="relu")(x)
    p2 = layers.SeparableConv2D(filters[2], 3, padding="same", activation="relu")(p2)

    p3 = layers.Conv2D(filters[3], 1, padding="same", activation="relu")(x)
    p3 = layers.SeparableConv2D(filters[4], 3, padding="same", activation="relu")(p3)
    p3 = layers.SeparableConv2D(filters[5], 3, padding="same", activation="relu")(p3)

    p4 = layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    p4 = layers.Conv2D(filters[6], 1, padding="same", activation="relu")(p4)

    return layers.Concatenate(axis=-1)([p1, p2, p3, p4])


def build_darinet(input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = 5) -> Model:
    """Build DARINet.

    Args:
        input_shape: Input leaf image shape.
        num_classes: 5 for Cassava, 3 for Rice, 4 for Maize (as in the notebooks).
    """
    inputs = layers.Input(input_shape)

    x = layers.Conv2D(64, 3, activation="relu")(inputs)  # 'valid' padding, as in the notebook
    inception = inception_block(x)
    x1 = layers.Conv2D(64, 3, padding="same", activation="relu")(inception)
    x = layers.Add()([x, x1])
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    channel = residual_channel_attention(x)
    spatial = residual_spatial_attention(x)
    x = layers.Add()([x, channel, spatial])
    x = layers.Conv2D(256, 3, activation="relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="darinet")


def compile_model(model: Model, learning_rate: float | None = None) -> Model:
    """Compile as in the notebooks: Adam (Keras default lr) + categorical cross-entropy."""
    import keras

    optimizer = (
        keras.optimizers.Adam() if learning_rate is None else keras.optimizers.Adam(learning_rate)
    )
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
