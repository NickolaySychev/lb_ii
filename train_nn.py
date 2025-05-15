import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import torch
import matplotlib.pyplot as plt

def load_synthetic(path_images='synthetic_images.pt', path_labels='synthetic_labels.pt'):
    images = torch.load(path_images)
    labels = torch.load(path_labels)
    images = images.numpy().astype('float32')
    images = np.transpose(images, (0, 2, 3, 1))  
    labels = labels.numpy().astype('int')
    return images, labels

def load_cifar10():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    return x_train, y_train, x_val, y_val

def build_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=7.5e-4)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    x_synthetic, y_synthetic_raw = load_synthetic()
    y_synthetic = to_categorical(y_synthetic_raw, 10)
    x_train, y_train, x_val, y_val = load_cifar10()

    model_real = build_model()
    model_synthetic = build_model()

    epochs = 20  
    batch_size = 64

    history_real = model_real.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )

    history_synthetic = model_synthetic.fit(
        x_synthetic, y_synthetic,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )

    model_real.save('classifier_on_real.h5')
    model_synthetic.save('classifier_on_synthetic.h5')
    np.save('history_real.npy', history_real.history)
    np.save('history_synthetic.npy', history_synthetic.history)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_real.history['val_accuracy'], label='Реальные данные (Val Acc)')
    plt.plot(history_synthetic.history['val_accuracy'], label='Синтетические данные (Val Acc)')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('Валидационная точность')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_real.history['val_loss'], label='Реальные данные (Val Loss)')
    plt.plot(history_synthetic.history['val_loss'], label='Синтетические данные (Val Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Валидационные потери')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    acc_diff = np.array(history_real.history['val_accuracy']) - np.array(history_synthetic.history['val_accuracy'])
    plt.figure(figsize=(6, 4))
    plt.plot(acc_diff, label='Разница (Реальные - Синтетические)')
    plt.xlabel('Эпоха')
    plt.ylabel('Разница в точности')
    plt.title('Разница в валидационной точности')
    plt.legend()
    plt.grid(True)
    plt.show()

    real_final_acc = history_real.history['val_accuracy'][-1]
    synthetic_final_acc = history_synthetic.history['val_accuracy'][-1]
    real_final_loss = history_real.history['val_loss'][-1]
    synthetic_final_loss = history_synthetic.history['val_loss'][-1]
    print(f"Реальные данные: Финальная Val Acc = {real_final_acc:.4f}, Val Loss = {real_final_loss:.4f}")
    print(f"Синтетические данные: Финальная Val Acc = {synthetic_final_acc:.4f}, Val Loss = {synthetic_final_loss:.4f}")
    print(f"Разница (Реальные - Синтетические): {real_final_acc - synthetic_final_acc:.4f}")

if __name__ == '__main__':
    main()