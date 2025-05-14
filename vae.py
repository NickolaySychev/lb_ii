import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 2x2
            nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 512 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 32x32
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 512, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_div

def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_vae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = load_cifar10(batch_size=32)
    model = VAE(latent_dim=256).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-4)
    recon_vals, kl_vals = [], []

    num_epochs = 20
    kl_anneal = 100    

    for epoch in range(1, num_epochs+1):
        model.train()
        recon_sum = 0
        kl_sum = 0
        beta = min(1.0, epoch / kl_anneal)

        for x, i in tqdm(train_loader, desc=f"Ep {epoch}/{num_epochs}"):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = VAE.loss_function(recon, x, mu, logvar, beta)
            loss.backward()
            opt.step()

            recon_sum += F.mse_loss(recon, x, reduction='sum').item()
            kl_sum    += (-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())).item()

        N = len(train_loader.dataset)
        recon_vals.append(recon_sum/N)
        kl_vals.append(kl_sum/N)
        print(f"Epoch {epoch} | Recon={recon_vals[-1]:.4f} | KL={kl_vals[-1]:.4f} | β={beta:.2f}")

    torch.save(model.state_dict(), 'vae_model.pt')
    np.save('recon_vals.npy', np.array(recon_vals))
    np.save('kl_vals.npy', np.array(kl_vals))
    plt.plot(recon_vals, label='Recon MSE')
    plt.plot(kl_vals, label='KL Div')
    plt.legend()
    plt.title("Training VAE Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    train_vae()