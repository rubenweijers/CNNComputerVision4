from tensorflow.keras.models import load_model

from data_processing import load_data
from models import DecayingLRSchedule

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    scores_val = {}
    scores_train = {}
    for variation in ["baseline", "nike", "collegedropout", "normaliser2000", "averagejoe"]:
        # Load model
        model = load_model(f"./models/model_{variation}.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule})

        # Evaluate model on validation set
        loss_val, accuracy_val = model.evaluate(X_val, y_val, verbose=0)
        print(f"Model: {variation} - Validation loss: {loss_val:.4f} - Validation accuracy: {accuracy_val:.4f}")

        # Evaluate model on training set
        loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
        print(f"Model: {variation} - Training loss: {loss_train:.4f} - Training accuracy: {accuracy_train:.4f}")

        # Add to dictionary
        scores_val[variation] = accuracy_val
        scores_train[variation] = accuracy_train

    # Sort by accuracy
    scores_val_sorted = sorted(scores_val.items(), key=lambda x: x[1], reverse=True)
    scores_train_sorted = sorted(scores_train.items(), key=lambda x: x[1], reverse=True)

    # Print sorted scores, round to 4 decimals
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_val_sorted])
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_train_sorted])
