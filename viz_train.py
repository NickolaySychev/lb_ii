import numpy as np
import matplotlib.pyplot as plt


def plot_comparison():
    history_synthetic = np.load('history_synthetic.npy', allow_pickle=True).item()
    history_real = np.load('history_real.npy', allow_pickle=True).item()

    epochs_synthetic = range(1, len(history_synthetic['accuracy']) + 1)
    epochs_real = range(1, len(history_real['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Точность на обучающих данных
    ax1.plot(epochs_synthetic, history_synthetic['accuracy'], 'b-', label='Точность на синтетических данных (train)')
    ax1.plot(epochs_real, history_real['accuracy'], 'r-', label='Точность на реальных данных (train)')

    # Валидационная точность
    ax1.plot(epochs_synthetic, history_synthetic['val_accuracy'], 'b--', label='Точность на синтетических данных (val)')
    ax1.plot(epochs_real, history_real['val_accuracy'], 'r--', label='Точность на реальных данных (val)')

    ax1.set_title('Сравнение точности моделей')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True)

    # Потери на обучающих данных
    ax2.plot(epochs_synthetic, history_synthetic['loss'], 'b-', label='Потери на синтетических данных (train)')
    ax2.plot(epochs_real, history_real['loss'], 'r-', label='Потери на реальных данных (train)')

    # Валидационные потери
    ax2.plot(epochs_synthetic, history_synthetic['val_loss'], 'b--', label='Потери на синтетических данных (val)')
    ax2.plot(epochs_real, history_real['val_loss'], 'r--', label='Потери на реальных данных (val)')

    ax2.set_title('Сравнение потерь моделей')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_plot.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_comparison()
