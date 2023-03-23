from tensorflow.keras.callbacks import TensorBoard

from data_processing import load_data
from models import make_model, prepare_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    kernel_size = 3
    pool_size = 2
    pooling = "max"
    dropout_value = None
    conv_act = "relu"
    learning_rate = 0.01
    batch_size = 64
    total_size = X_train.shape[0]
    epochs = 1

    # Create the model
    model = make_model(kernel_size=kernel_size, pool_size=pool_size, pooling=pooling, dropout_value=dropout_value, conv_act=conv_act)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size)

    tensorboard_callback = TensorBoard(log_dir="./logs")  # Tensorboard callback
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save("./models/model_baseline.h5")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
