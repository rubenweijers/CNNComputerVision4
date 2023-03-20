import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from data_processing import load_data


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))

    ax = plt.gca().set_aspect("equal")  # Set aspect ratio to square
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def plot_learning_rate_schedule(epochs: 15, learning_rate: 0.001):
    x = list(range(epochs))
    y = [learning_rate * 0.5 ** (i // 5) for i in x]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    # model = load_model("./models/model_baseline.h5")

    # X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # # Evaluate the model
    # y_pred = model.predict(X_test)
    # y_pred = np.argmax(y_pred, axis=1)  # Predictions
    # y_test = np.argmax(y_test, axis=1)  # Ground truth

    # Plot confusion matrix
    # plot_confusion_matrix(y_test, y_pred)

    # Plot LR schedule
    plot_learning_rate_schedule(epochs=15, learning_rate=0.001)
