import torch
import matplotlib.pyplot as plt

synthetic_images = torch.load('synthetic_images.pt')
synthetic_labels = torch.load('synthetic_labels.pt')

synthetic_images = synthetic_images.permute(0, 2, 3, 1)  
synthetic_images = synthetic_images.numpy()

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def show_images(images, labels, n_images=10):
    fig, axes = plt.subplots(1, n_images, figsize=(15, 3))
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].set_title(f"{class_names[labels[i]]}")
        axes[i].axis('off') 
    plt.show()

indices = torch.randint(0, synthetic_images.shape[0], (10,))
show_images(synthetic_images[indices], synthetic_labels[indices])