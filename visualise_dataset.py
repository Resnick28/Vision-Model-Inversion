import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def visualize_dataset_samples(loader, dataset_info, num_samples=16, denormalize=True):
    """
    Visualize random samples from the dataset
    
    Args:
        loader: DataLoader to sample from
        dataset_info: Dictionary with dataset information
        num_samples: Number of samples to display
        denormalize: Whether to denormalize images
    """
    save_path = "results/dataset_samples.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Denormalize images for visualization
    if denormalize:
        if dataset_info['name'] == 'mnist':
            mean, std = 0.1307, 0.3081
            images = images * std + mean
        elif dataset_info['name'] == 'fashion_mnist':
            mean, std = 0.2860, 0.3530
            images = images * std + mean
        elif dataset_info['name'] == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            images = images * std + mean
        elif dataset_info['name'] == 'svhn':
            mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1)
            std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1)
            images = images * std + mean
    
    # Create a grid of images
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            if dataset_info['channels'] == 1:
                ax.imshow(images[i].squeeze(), cmap='gray')
            else:
                ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure
    plt.close(fig)  # Close to free memory