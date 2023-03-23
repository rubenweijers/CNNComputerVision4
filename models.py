import tensorflow as tf
from tensorflow.keras.layers import (AveragePooling2D, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential


def make_model(kernel_size: int = 3, pool_size: int = 2, pooling: str = "max", dropout_value: float = None, conv_act: str = "relu"):
    """The model used for the experiments."""

    if pooling == "max":
        pooling = MaxPooling2D
    elif pooling == "avg":
        pooling = AveragePooling2D
    else:
        raise ValueError("Pooling must be either 'max' or 'average'.")

    if conv_act == "relu":
        conv_act = "relu"
    elif conv_act == "swish":
        conv_act = tf.nn.swish
    else:
        raise ValueError("conv_act must be either 'relu' or 'swish'.")

    model = Sequential()
    model.add(Conv2D(64, kernel_size, activation=conv_act, input_shape=(28, 28, 1)))
    model.add(pooling(pool_size))
    model.add(Conv2D(32, kernel_size, activation=conv_act))
    model.add(pooling(pool_size))
    model.add(Conv2D(16, kernel_size, activation=conv_act))
    model.add(pooling(pool_size))
    model.add(Flatten())

    if dropout_value is not None:
        model.add(Dropout(dropout_value))

    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model
