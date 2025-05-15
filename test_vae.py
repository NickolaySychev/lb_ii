import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae import VAE 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 256
model_path = 'vae_model.pt'

model = VAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=10, shuffle=True)
images, _ = next(iter(loader))
images = images.to(device)

with torch.no_grad():
    mu, logvar = model.encode(images)
    z = model.reparameterize(mu, logvar)
    recon_images = model.decode(z)

images = images.cpu().numpy()
recon_images = recon_images.cpu().numpy()

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
    axes[0, i].axis('off')
    axes[1, i].imshow(np.transpose(recon_images[i], (1, 2, 0)))
    axes[1, i].axis('off')

axes[0, 0].set_title('Оригинал')
axes[1, 0].set_title('VAE')
plt.tight_layout()
plt.show()
