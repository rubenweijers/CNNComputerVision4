from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam

from data_processing import load_data
from models import model_baseline

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    model = model_baseline()

    # Tensorboard callback
    tensorboard_callback = TensorBoard(log_dir="./logs")

    # Decrease the learning rate at a 1/2 of the value every 5 epochs
    learning_rate_schedule = LearningRateScheduler(lambda epoch: learning_rate * 0.5 ** (epoch // 5))
    opt = Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[tensorboard_callback])

    model.save("./models/model_baseline.h5")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
