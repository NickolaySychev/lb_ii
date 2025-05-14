import numpy as np
import matplotlib.pyplot as plt

recon_vals = np.load('recon_vals.npy')
kl_vals    = np.load('kl_vals.npy')

plt.figure(figsize=(10, 5))
plt.plot(recon_vals, label='Reconstruction MSE')
plt.plot(kl_vals, label='KL Divergence')
plt.title("VAE Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
