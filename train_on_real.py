import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from model import build_model, load_cifar10

def main():
    x_train, y_train, x_val, y_val = load_cifar10()

    model = build_model()
    epochs = 15
    batch_size = 32

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )

    model.save('classifier_on_real.h5')
    np.save('history_real.npy', history.history)

    print("Обучение на реальных данных завершено")

if __name__ == '__main__':
    main()