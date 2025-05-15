import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model_real = tf.keras.models.load_model('classifier_on_real.h5')
model_synth = tf.keras.models.load_model('classifier_on_synthetic.h5')

(_, _), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
x_val = x_val.astype('float32') / 255.0
y_val = y_val.flatten()

N = 10
indices = np.random.choice(len(x_val), size=N, replace=False)
images = x_val[indices]
true_labels = y_val[indices]

pred_real = np.argmax(model_real.predict(images), axis=1)
pred_synth = np.argmax(model_synth.predict(images), axis=1)

plt.figure(figsize=(18, 6))
for i in range(N):
    plt.subplot(2, N//2, i+1)
    plt.imshow(images[i])
    plt.axis('off')

    real_ok = pred_real[i] == true_labels[i]
    synth_ok = pred_synth[i] == true_labels[i]

    if real_ok and synth_ok:
        color = 'green'
    elif real_ok or synth_ok:
        color = 'orange'
    else:
        color = 'red'

    plt.title(f"True: {class_names[true_labels[i]]}\n"
              f"Real: {class_names[pred_real[i]]} ({'✓' if real_ok else '✗'})\n"
              f"Synth: {class_names[pred_synth[i]]} ({'✓' if synth_ok else '✗'})",
              color=color)
plt.suptitle('Сравнение предсказаний двух моделей (реальная и синтетическая)', fontsize=16)
plt.tight_layout()
plt.show()
