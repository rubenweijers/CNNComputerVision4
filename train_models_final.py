import pickle

import numpy as np
from tensorflow.keras.callbacks import TensorBoard

from data_processing import load_data
from models import make_model, prepare_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Combine train and validation sets for final training
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    kernel_size = 3
    pool_size = 2
    pooling_type = "max"
    dropout_value = None
    conv_act = "relu"
    normalise = False

    learning_rate = 0.01
    batch_size = 64
    total_size = X_train.shape[0]
    epochs = 15

    model_variation = "nike"  # Choose from: {nike, averagejoe}, only two options for final training

    if model_variation == "nike":
        conv_act = "swish"
    elif model_variation == "averagejoe":
        pooling_type = "avg"
    else:
        raise ValueError(f"Invalid model variation: {model_variation}")

    # Create the model
    model = make_model(kernel_size=kernel_size, pool_size=pool_size, pooling_type=pooling_type,
                       dropout_value=dropout_value, conv_act=conv_act)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size)

    tensorboard_callback = TensorBoard(log_dir=f"./logs/{model_variation}_final")
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(
        X_val, y_val), batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save(f"./models/model_{model_variation}_final.h5")

    # Save the history
    with open(f"./history/history_{model_variation}_final.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
