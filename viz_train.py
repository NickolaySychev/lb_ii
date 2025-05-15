import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Функция для загрузки модели и истории
def load_model_and_history(model_path, history_path):
    # Загружаем модель из файла .h5
    model = tf.keras.models.load_model(model_path)
    
    # Загружаем историю обучения
    history = np.load(history_path, allow_pickle=True).item()
    
    return model, history

# Функция для построения графиков
def plot_training_history(history_real, history_synthetic):
    # Графики точности
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_real['val_accuracy'], label='Реальные данные (Val Acc)')
    plt.plot(history_synthetic['val_accuracy'], label='Синтетические данные (Val Acc)')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('Валидационная точность')
    plt.legend()
    plt.grid(True)

    # Графики потерь
    plt.subplot(1, 2, 2)
    plt.plot(history_real['val_loss'], label='Реальные данные (Val Loss)')
    plt.plot(history_synthetic['val_loss'], label='Синтетические данные (Val Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Валидационные потери')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # График разницы в точности
    acc_diff = np.array(history_real['val_accuracy']) - np.array(history_synthetic['val_accuracy'])
    plt.figure(figsize=(6, 4))
    plt.plot(acc_diff, label='Разница (Реальные - Синтетические)')
    plt.xlabel('Эпоха')
    plt.ylabel('Разница в точности')
    plt.title('Разница в валидационной точности')
    plt.legend()
    plt.grid(True)
    plt.show()

# Главная функция
def main():
    # Путь к файлам моделей и истории
    model_real_path = 'classifier_on_real.h5'
    model_synthetic_path = 'classifier_on_synthetic.h5'
    history_real_path = 'history_real.npy'
    history_synthetic_path = 'history_synthetic.npy'

    # Загружаем модели и историю
    model_real, history_real = load_model_and_history(model_real_path, history_real_path)
    model_synthetic, history_synthetic = load_model_and_history(model_synthetic_path, history_synthetic_path)

    # Строим графики
    plot_training_history(history_real, history_synthetic)

if __name__ == '__main__':
    main()
