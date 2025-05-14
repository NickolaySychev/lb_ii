import torch
from vae import VAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_cifar10_val():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    x_val = torch.stack([x for x, _ in test_dataset])
    y_val = torch.tensor([y for _, y in test_dataset])
    return x_val, y_val

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(latent_dim=256).to(device)
    model.load_state_dict(torch.load('vae_model.pt', map_location=device))
    model.eval()

    train_loader, _ = load_cifar10(batch_size=128)
    latent_dim = model.latent_dim

    sum_mu = torch.zeros(10, latent_dim, device=device)
    sum_logvar = torch.zeros(10, latent_dim, device=device)
    counts = torch.zeros(10, device=device)

    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            mu, logvar = model.encode(x)
        for c in range(10):
            mask = (labels == c)
            if mask.any():
                sum_mu[c] += mu[mask].sum(dim=0)
                sum_logvar[c] += logvar[mask].sum(dim=0)
                counts[c] += mask.sum()

    z_means = sum_mu / counts.unsqueeze(1)
    logvar_means = sum_logvar / counts.unsqueeze(1)
    std_means = torch.exp(0.5 * logvar_means)

    repeats_per_class = 5000
    batch_size = 128
    sigma = 0.3

    all_imgs = []
    all_labels = []

    for c in range(10):
        z_base = z_means[c].unsqueeze(0).repeat(repeats_per_class, 1)
        std_c = std_means[c].unsqueeze(0).repeat(repeats_per_class, 1)
        noise = sigma * std_c * torch.randn_like(z_base)
        z_synth = z_base + noise

        for i in range(0, repeats_per_class, batch_size):
            z_batch = z_synth[i:i+batch_size].to(device)
            with torch.no_grad():
                imgs = model.decode(z_batch)
            all_imgs.append(imgs.cpu())
            all_labels.append(torch.full((imgs.size(0),), c, dtype=torch.long))

    synthetic_images = torch.cat(all_imgs, dim=0)
    synthetic_labels = torch.cat(all_labels, dim=0)

    plt.figure(figsize=(15, 3))
    for c in range(10):
        img = synthetic_images[synthetic_labels == c][0].permute(1, 2, 0).numpy()
        plt.subplot(1, 10, c + 1)
        plt.imshow(img)
        plt.title(f"Класс {c}")
        plt.axis('off')
    plt.savefig('synthetic_samples.png')
    plt.show()

    torch.save(synthetic_images, 'synthetic_images.pt')
    torch.save(synthetic_labels, 'synthetic_labels.pt')
    print(f"Synthetic dataset generated: {synthetic_images.shape[0]} samples")

if __name__ == '__main__':
    main()