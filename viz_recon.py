import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from torchvision import transforms
from vae import VAE  # Импортируй свою модель VAE отсюда, если она в другом файле
import torch.nn.functional as F

# Названия классов CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# === Загрузка моделей классификаторов (на TensorFlow) ===
model_real = tf.keras.models.load_model('classifier_on_real.h5')
model_synth = tf.keras.models.load_model('classifier_on_synthetic.h5')

# === Загрузка изображений CIFAR-10 ===
(_, _), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
x_val = x_val.astype('float32') / 255.0
y_val = y_val.flatten()

# Выбор случайных изображений
N = 10
indices = np.random.choice(len(x_val), size=N, replace=False)
images_tf = x_val[indices]
true_labels = y_val[indices]

# === Прогон через классификаторы ===
pred_real = np.argmax(model_real.predict(images_tf), axis=1)
pred_synth = np.argmax(model_synth.predict(images_tf), axis=1)

# === Загрузка VAE и прогон через него ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=256).to(device)
vae.load_state_dict(torch.load("vae_model.pt", map_location=device))
vae.eval()

# Подготовка изображений для VAE
transform = transforms.Compose([transforms.ToTensor()])
images_pt = torch.stack([transform(img) for img in images_tf]).to(device)

with torch.no_grad():
    reconstructions, _, _ = vae(images_pt)
reconstructions_np = reconstructions.cpu().permute(0, 2, 3, 1).numpy()

# === Визуализация: 3 строки — оригинал, классификаторы, восстановление VAE ===
plt.figure(figsize=(20, 6))

for i in range(N):
    # Оригинал
    plt.subplot(3, N, i+1)
    plt.imshow(images_tf[i])
    plt.axis('off')
    plt.title("Оригинал")

    # Предсказания
    real_ok = pred_real[i] == true_labels[i]
    synth_ok = pred_synth[i] == true_labels[i]
    if real_ok and synth_ok:
        color = 'green'
    elif real_ok or synth_ok:
        color = 'orange'
    else:
        color = 'red'
    
    plt.subplot(3, N, N + i + 1)
    plt.imshow(images_tf[i])
    plt.axis('off')
    plt.title(
        f"True: {class_names[true_labels[i]]}\n"
        f"Real: {class_names[pred_real[i]]} ({'✓' if real_ok else '✗'})\n"
        f"Synth: {class_names[pred_synth[i]]} ({'✓' if synth_ok else '✗'})",
        color=color,
        fontsize=8
    )

    # Восстановление VAE
    plt.subplot(3, N, 2 * N + i + 1)
    plt.imshow(reconstructions_np[i])
    plt.axis('off')
    plt.title("VAE")

plt.suptitle('Оригинал — Предсказания моделей — Восстановление VAE', fontsize=16)
plt.tight_layout()
plt.show()
