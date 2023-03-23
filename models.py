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


class DecayingLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, batch_size, total_size) -> None:
        super().__init__()
        self.lr = lr
        self.n_steps = total_size // batch_size  # Number of steps per epoch, e.g. number of batches per epoch

    def __call__(self, step):
        """Decrease the learning rate at a 1/2 of the value every 5 epochs"""
        epoch = step / self.n_steps  # Current epoch, e.g. epoch 3 or epoch 3.5
        return self.lr * tf.math.pow(0.5, tf.math.floor(epoch / 5))


if __name__ == "__main__":
    model = make_model()
    model.summary()
