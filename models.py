import tensorflow as tf
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def make_model(kernel_size: int = 3, pool_size: int = 2, pooling_type: str = "max",
               dropout_value: float = None, conv_act: str = "relu", normalise: bool = False):
    """The model used for the experiments."""

    if pooling_type == "max":
        pooling = MaxPooling2D
    elif pooling_type == "avg":
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

    if normalise:
        model.add(BatchNormalization())

    model.add(pooling(pool_size))
    model.add(Conv2D(32, kernel_size, activation=conv_act))

    if normalise:
        model.add(BatchNormalization())

    model.add(pooling(pool_size))
    model.add(Conv2D(16, kernel_size, activation=conv_act))

    if normalise:
        model.add(BatchNormalization())

    model.add(pooling(pool_size))
    model.add(Flatten())

    if dropout_value is not None:
        model.add(Dropout(dropout_value))

    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model


def prepare_model(model, learning_rate: float = 0.01, batch_size: int = 64, total_size: int = 48_000):
    # Decrease the learning rate at a 1/2 of the value every 5 epochs
    learning_rate_schedule = DecayingLRSchedule(learning_rate, batch_size=batch_size, total_size=total_size)
    opt = Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class DecayingLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, batch_size, total_size) -> None:
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.total_size = total_size

        self.n_steps = self.total_size // self.batch_size  # Number of steps per epoch, e.g. number of batches per epoch

    def __call__(self, step):
        """Decrease the learning rate at a 1/2 of the value every 5 epochs"""
        epoch = step / self.n_steps  # Current epoch, e.g. epoch 3 or epoch 3.5
        return self.lr * tf.math.pow(0.5, tf.math.floor(epoch / 5))

    def get_config(self):
        return {"lr": self.lr, "batch_size": self.batch_size, "total_size": self.total_size}


if __name__ == "__main__":
    model = make_model()
    model.summary()
