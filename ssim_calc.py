# ===== SSIM CALCULATION =====

import torch
import numpy as np
import matplotlib as plt

import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def generate_and_evaluate_mnist_ssim(generator_model, train_loader, device, 
                                   latent_dim=100, num_classes=10, 
                                   samples_per_class=50, visualize=True):
    """
    Generate MNIST images for each class using the trained generator and calculate SSIM scores
    
    Args:
        generator_model: The trained generator model
        train_loader: DataLoader containing the MNIST dataset
        device: Device to run the model on (cuda or cpu)
        latent_dim: Dimension of the latent space (default: 100)
        num_classes: Number of classes in the dataset (default: 10 for MNIST)
        samples_per_class: Number of samples to generate per class
        visualize: Whether to visualize some generated samples and SSIM results
    
    Returns:
        tuple: (mean_ssim, per_class_ssim_dict)
    """
    generator_model.eval()
    
    all_generated_images = []
    all_class_labels = []
    
    print(f"Generating {samples_per_class} images for each of the {num_classes} classes...")
    for class_idx in range(num_classes):
        
        labels = torch.zeros(samples_per_class, num_classes).to(device)
        labels[:, class_idx] = 1.0
        
        z = torch.randn(samples_per_class, latent_dim).to(device)
        
        with torch.no_grad():
            generated_imgs = generator_model(z, labels)
        
        # Normalize to [0,1] range from [-1,1] range (Tanh output)
        generated_imgs = (generated_imgs + 1) / 2.0
            
        # Move to CPU and convert to numpy
        generated_imgs_np = generated_imgs.cpu().detach().numpy()
        
        # Ensure correct shape [N, H, W] for SSIM comparison
        if len(generated_imgs_np.shape) == 4 and generated_imgs_np.shape[1] == 1:
            generated_imgs_np = generated_imgs_np.squeeze(1)
            
        # Store generated images and their class labels
        all_generated_images.append(generated_imgs_np)
        all_class_labels.extend([class_idx] * samples_per_class)
    
    # Concatenate all generated images
    all_generated_images = np.concatenate(all_generated_images, axis=0)
    all_class_labels = np.array(all_class_labels)
    
    # If visualization is enabled, show some generated images
    if visualize:
        # Display a grid of generated images (2 per class)
        fig, axes = plt.subplots(num_classes, 2, figsize=(6, 15))
        for cls in range(num_classes):
            # Find images of this class
            class_indices = np.where(all_class_labels == cls)[0][:2]
            for i, idx in enumerate(class_indices):
                axes[cls, i].imshow(all_generated_images[idx], cmap='gray')
                axes[cls, i].axis('off')
                axes[cls, i].set_title(f"Class {cls}")
        plt.tight_layout()
        plt.suptitle("Generated MNIST Samples")
        plt.show()
    
    # Collect original MNIST images for comparison
    original_images = []
    original_labels = []
    
    print("Loading original MNIST images for comparison...")
    for images, labels in train_loader:
        # Extract images and convert to numpy
        images_np = images.numpy()
        
        # If image shape is [N, 1, H, W], convert to [N, H, W]
        if images_np.shape[1] == 1:
            images_np = images_np.squeeze(1)
            
        original_images.append(images_np)
        original_labels.append(labels.numpy())
        
        # Limit the number of original images to process
        if len(original_images) * images.shape[0] >= samples_per_class * num_classes * 4:
            break    
    
    # Concatenate collected images and labels
    original_images = np.concatenate(original_images, axis=0)
    original_labels = np.concatenate(original_labels, axis=0)
    
    # Calculate SSIM for each class
    class_ssim_scores = {i: [] for i in range(num_classes)}
    all_ssim_scores = []
    
    print("Calculating SSIM scores...")
    # For each generated image
    for gen_idx, (gen_img, gen_class) in enumerate(tqdm(zip(all_generated_images, all_class_labels))):
        # Find original images of the same class
        matching_indices = np.where(original_labels == gen_class)[0]
        matching_originals = original_images[matching_indices]
        
        # Calculate SSIM with each matching original image
        best_ssim = -1
        
        # Limit number of comparisons for efficiency
        max_comparisons = min(100, len(matching_indices))
        for i in range(max_comparisons):
            orig_img = matching_originals[i]
            # Calculate SSIM between generated and original image
            try:
                ssim_val = ssim(gen_img, orig_img, data_range=1.0)
                if ssim_val > best_ssim:
                    best_ssim = ssim_val
            except Exception as e:
                print(f"Error calculating SSIM: {e}")
                print(f"Gen image shape: {gen_img.shape}, Original image shape: {orig_img.shape}")
                continue
        
        # Record best SSIM score
        if best_ssim != -1:
            class_ssim_scores[gen_class].append(best_ssim)
            all_ssim_scores.append(best_ssim)
    
    # Calculate mean SSIM overall and for each class
    mean_ssim = np.mean(all_ssim_scores)
    class_mean_ssims = {cls: np.mean(scores) if scores else 0 for cls, scores in class_ssim_scores.items()}
    
    # Print results
    print(f"Overall mean SSIM: {mean_ssim:.4f}")
    print("SSIM by class:")
    for cls in range(num_classes):
        print(f"  Class {cls}: {class_mean_ssims[cls]:.4f}")
    
    # Visualization of SSIM results
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.hist(all_ssim_scores, bins=20)
        plt.title(f"SSIM Distribution (Mean: {mean_ssim:.4f})")
        plt.xlabel("SSIM Score")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.show()
        
        # Per-class SSIM visualization
        plt.figure(figsize=(12, 6))
        classes = list(range(num_classes))
        scores = [class_mean_ssims[c] for c in classes]
        plt.bar(classes, scores)
        plt.title("SSIM Score by MNIST Class")
        plt.xlabel("Class")
        plt.ylabel("Mean SSIM")
        plt.xticks(classes)
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    
    return mean_ssim, class_mean_ssims