from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from data_processing import load_data
from models import DecayingLRSchedule, make_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    batch_size = 64
    epochs = 10
    learning_rate = 0.01

    model = make_model(kernel_size=3, pool_size=2, pooling="max", dropout_value=None, conv_act="relu")  # Baseline model params
    tensorboard_callback = TensorBoard(log_dir="./logs")  # Tensorboard callback

    # Decrease the learning rate at a 1/2 of the value every 5 epochs
    learning_rate_schedule = DecayingLRSchedule(learning_rate, batch_size=batch_size, total_size=X_train.shape[0])
    opt = Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[tensorboard_callback])

    model.save("./models/model_baseline.h5")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
