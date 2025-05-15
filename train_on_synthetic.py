import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import torch
from model import build_model, load_cifar10, load_synthetic


def main():
    x_synthetic, y_synthetic_raw = load_synthetic()
    y_synthetic = to_categorical(y_synthetic_raw, 10)
    _, _, x_val, y_val = load_cifar10()

    model = build_model()
    epochs = 10
    batch_size = 32

    history = model.fit(
        x_synthetic, y_synthetic,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )

    model.save('classifier_on_synthetic.h5')
    np.save('history_synthetic.npy', history.history)

    print("Обучение на синтетических данных завершено")


if __name__ == '__main__':
    main()
