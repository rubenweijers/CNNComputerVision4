import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from data_processing import load_data
from models import DecayingLRSchedule


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))

    ax = plt.gca().set_aspect("equal")  # Set aspect ratio to square
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def plot_learning_rate_schedule(epochs: 15, learning_rate: 0.001, batch_size: int = 64, total_size: int = 48_000):
    """Plot the DecayingLRSchedule."""
    schedule = DecayingLRSchedule(learning_rate, batch_size=batch_size, total_size=total_size)
    n_steps = total_size // batch_size  # Number of steps per epoch, each step is a batch

    x = list(range(epochs))
    # Schedule is calculated based on steps, not epochs
    # So we multiply the epoch by the number of steps per epoch
    y = [schedule(n_steps * epoch) for epoch in x]
    plt.plot(x, y)
    plt.xlim(0, epochs)
    plt.ylim(0, None)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Evaluate the model
    model = load_model("./models/model_baseline.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule})
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)  # Predictions
    y_test = np.argmax(y_test, axis=1)  # Ground truth

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Plot LR schedule
    plot_learning_rate_schedule(epochs=15, learning_rate=0.01, batch_size=64, total_size=48_000)
