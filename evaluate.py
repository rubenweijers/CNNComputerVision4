import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from data_processing import load_data
from models import DecayingLRSchedule


def plot_confusion_matrix(y_test, y_pred, variation: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))

    ax = plt.gca().set_aspect("equal")  # Set aspect ratio to square
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title(f"Confusion matrix for model: {variation}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"./img/cm_{variation}.png")
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
    plt.savefig("./img/learning_rate_schedule.png")
    plt.show()


def plot_history(history: dict, variation: str) -> None:
    """Plot the history of a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    ax1.plot(history["loss"], label="Loss")
    ax1.plot(history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(np.arange(0, len(history["loss"]), 1))
    ax1.legend()

    ax2.plot(history["accuracy"], label="Accuracy")
    ax2.plot(history["val_accuracy"], label="Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(np.arange(0, len(history["accuracy"]), 1))
    ax2.legend()

    plt.suptitle(f"Model: {variation}")

    ax1.yaxis.grid(True)  # Gridlines, only in the horizontal direction
    ax2.yaxis.grid(True)

    ax1.set_ylim(0, None)  # Loss has no upper limit
    ax2.set_ylim(0.5, 1)

    plt.show()

    fig.savefig(f"./img/plot_{variation}.png")


if __name__ == "__main__":
    # Evaluate the model
    model_variation = "baseline"  # Choose from: {baseline, nike, collegedropout, normaliser2000, averagejoe}
    model = load_model(f"./models/model_{model_variation}.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule})

    # Load the history
    with open(f"./history/history_{model_variation}.pkl", "rb") as f:
        model_history = pickle.load(f)

    # Show the model summary with graphviz
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=f"./img/summary_{model_variation}.png")

    # Plot loss and accuracy, based on history
    plot_history(model_history, model_variation)

    # Load the data and plot the confusion matrix
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)  # Predictions
    y_test = np.argmax(y_test, axis=1)  # Ground truth

    plot_confusion_matrix(y_test, y_pred, model_variation)

    # Plot LR schedule, same for all variations
    # plot_learning_rate_schedule(epochs=15, learning_rate=0.01, batch_size=64, total_size=48_000)
