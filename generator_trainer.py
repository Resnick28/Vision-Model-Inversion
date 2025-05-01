import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from loss_fn import (
    total_variation_loss, gradient_minimization_loss, kl_divergence,
    pixel_range_loss, cosine_similarity_loss, feature_orthogonality_loss
)
from utils import add_linf_perturbation

class ReconstructionTrainer:
    def __init__(self, classifier, generator, device, dataset_info, train_loader=None, 
                 perturb_strength=1, lambda_min=0.01, alpha=1, alpha_prime=1, beta=1, beta_prime=1, 
                  gamma=1, delta=1, eta1=1, eta2=1, eta3=1, eta4=1):
        
        self.classifier = classifier.eval()
        self.generator = generator
        self.device = device
        self.dataset_info = dataset_info
        self.train_loader = train_loader
        self.recon_criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        
        self.perturb_strength = perturb_strength
        self.max_batch_size = 60000*10
        self.lambda_min = lambda_min
        self.a_values = nn.Parameter(
            torch.sqrt(torch.rand(self.max_batch_size, device=device) + self.lambda_min)
        )
        self.theta = torch.cat([p.flatten() for p in self.classifier.parameters()])
        self.ssim_list = []
        self.ssim_results = []

        self.alpha = alpha
        self.alpha_prime = alpha_prime
        self.beta = beta
        self.beta_prime = beta_prime
        self.gamma = gamma
        self.delta = delta
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.eta4 = eta4

        self.optimizer = torch.optim.Adam([
            {'params': self.generator.parameters(), 'lr': 0.003},
            {'params': [self.a_values], 'lr': 0.01}
        ])
        self.loss_history = {k: [] for k in ['total_loss', 'kl_loss', 'ce_loss', 
                                           'perturbed_kl_loss', 'perturbed_ce_loss', 
                                           'cosine_loss', 'ortho_loss', 'var_loss', 
                                           'pixel_loss', 'grad_loss', 'stationary_loss']}
        self.epoch = 0

    def stationary_loss(self, gen_imgs, labels, a_values, logits):
        lambda_values = a_values ** 2
        target_scores = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('-inf'))
        max_other_scores = (logits + mask).max(dim=1)[0]
        
        margins = target_scores - max_other_scores
        weighted_margins = (margins * lambda_values).sum()
        grads = torch.autograd.grad(weighted_margins, self.classifier.parameters(), create_graph=True)
        gradient_sum = torch.cat([g.flatten() for g in grads])
        
        return torch.norm(self.theta - gradient_sum, p=2) ** 2
    
    def calculate_ssim_scores(self, batch_size, num_classes):
        """Calculate SSIM scores between generated and real images per class"""
        self.generator.eval()
        threshold = 0.4
        
        # Generate images
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        sampled_labels = torch.randint(0, num_classes, (batch_size,)).to(self.device)
        labels_onehot = torch.zeros(batch_size, num_classes, device=self.device)
        labels_onehot.scatter_(1, sampled_labels.unsqueeze(1), 1)
        gen_imgs = self.generator(z, labels_onehot)
        
        # Collect real images by class
        real_images_by_class = defaultdict(list)
        for real_imgs, real_labels in self.train_loader:
            for img, label in zip(real_imgs, real_labels):
                real_images_by_class[label.item()].append(self.denormalize(img).cpu().numpy())
        
        reconstruction_by_class = {}
        all_matched_ssim = []
        reconstructed_count = total_matched = 0
        reconst_img = []
        max_ssim_lst = []
        for label_idx in set(sampled_labels.cpu().numpy()):
            if label_idx not in real_images_by_class or not real_images_by_class[label_idx]:
                continue
                
            # Get images for this class
            gen_indices = [i for i, label in enumerate(sampled_labels) if label.item() == label_idx]
            gen_images = [gen_imgs[i].detach().cpu().numpy().transpose(1, 2, 0) for i in gen_indices]
            real_images = [img.transpose(1, 2, 0) for img in real_images_by_class[label_idx]]
            
            # Calculate SSIM matrix
            ssim_matrix = np.zeros((len(gen_images), len(real_images)))
            for i, gen_img in enumerate(gen_images):
                for j, real_img in enumerate(real_images):
                    ssim_matrix[i, j] = ssim(gen_img, real_img, channel_axis=-1, data_range=1.0)
                    
            # Apply Hungarian algorithm to maximize total SSIM
            row_indices, col_indices = linear_sum_assignment(-ssim_matrix)
            
            # Process matches
            class_matches = []
            class_ssim_scores = []
            class_reconstructed = 0
            max_ssim = 0
            
            for i, j in zip(row_indices, col_indices):
                ssim_score = ssim_matrix[i, j]
                match_info = {'gen_idx': gen_indices[i], 'real_idx': j, 'ssim': ssim_score, 
                            'reconstructed': ssim_score >= threshold}
                max_ssim = max(max_ssim, ssim_score)
                if ssim_score != 0:
                    class_matches.append(match_info)
                    class_ssim_scores.append(ssim_score)
                    all_matched_ssim.append(ssim_score)
                
                if ssim_score >= threshold:
                    reconst_img.append(gen_images[i])
                    class_reconstructed += 1
                    
            avg_ssim = sum(class_ssim_scores) / len(class_ssim_scores) if class_ssim_scores else 0
            reconstruction_by_class[label_idx] = {
                'matches': class_matches, 'avg_ssim': avg_ssim, 'max_ssim': max_ssim,
                'reconstructed': class_reconstructed, 'total': len(class_matches)
            }
            max_ssim_lst.append(max_ssim)
            reconstructed_count += class_reconstructed
            total_matched += len(class_matches)
        
        # Calculate overall metrics
        reconstruction_percentage = (reconstructed_count / total_matched * 100) if total_matched > 0 else 0
        avg_ssim_reconstructed = sum(all_matched_ssim) / len(all_matched_ssim) if all_matched_ssim else 0
        max_ssim_reconstructed = sum(max_ssim_lst) / len(max_ssim_lst) if max_ssim_lst else 0
        return {
            "by_class": {
                class_idx: {
                    'avg_ssim': info['avg_ssim'], 'reconstructed': info['reconstructed'],
                    'total': info['total'], 'percentage': (info['reconstructed'] / info['total'] * 100) if info['total'] > 0 else 0
                } for class_idx, info in reconstruction_by_class.items()
            },
            "overall_avg_ssim": avg_ssim_reconstructed,
            "overall_max_ssim_avg": max_ssim_reconstructed,
            "reconstructed": reconstructed_count,
            "total_matched": total_matched,
            "reconstruction_percentage": reconstruction_percentage,
            "threshold": threshold,
            "detailed_matches": reconstruction_by_class,
            "Reconstructed Images": reconst_img,
        }
            
    def train_step(self, batch_size):
        self.generator.train()
        self.optimizer.zero_grad()
        
        num_classes = self.dataset_info['num_classes']
        self.latent_dim = self.generator.latent_dim
        
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        sampled_labels = torch.randint(0, num_classes, (batch_size,)).to(self.device)
        labels_onehot = torch.zeros(batch_size, num_classes, device=self.device)
        labels_onehot.scatter_(1, sampled_labels.unsqueeze(1), 1)
        
        gen_imgs = self.generator(z, labels_onehot)
        var_loss = total_variation_loss(gen_imgs)
        grad_loss = gradient_minimization_loss(self.classifier, gen_imgs, sampled_labels, self.recon_criterion)
        pixel_loss = pixel_range_loss(gen_imgs)
        
        # Process perturbed images
        perturbed_imgs = add_linf_perturbation(gen_imgs, self.perturb_strength)
        perturbed_logits = self.classifier(perturbed_imgs)
        perturbed_pdf = F.softmax(perturbed_logits, dim=1)
        perturbed_ce_loss = self.criterion(perturbed_logits, sampled_labels)
        perturbed_kl_loss = kl_divergence(perturbed_pdf, labels_onehot)
        perturbation_accuracy = (torch.argmax(perturbed_pdf, dim=1) == sampled_labels).float().mean()

        # Process original images
        logits = self.classifier(gen_imgs)
        penultimate_features = self.classifier.get_penultimate(gen_imgs)
        output_pdf = F.softmax(logits, dim=1)
        ce_loss = self.criterion(logits, sampled_labels)
        kl_loss = kl_divergence(output_pdf, labels_onehot)
        cosine_loss = cosine_similarity_loss(penultimate_features)
        ortho_loss = feature_orthogonality_loss(penultimate_features)
        accuracy = (torch.argmax(logits, dim=1) == sampled_labels).float().mean()

        batch_a_values = self.a_values[:batch_size]
        stationary_loss = self.stationary_loss(gen_imgs, sampled_labels, batch_a_values, logits)        
    
        ## Approach 1
        '''
        epoch_threshold = 400
        alpha = self.alpha if self.epoch < epoch_threshold else 0
        beta = self.beta if self.epoch < epoch_threshold else 0

        epoch_threshold = 400  
        alpha_prime = 0 if self.epoch < epoch_threshold else self.alpha_prime
        beta_prime = 0 if self.epoch < epoch_threshold else self.beta_prime
        '''

        ## Approach 2
        '''
        epoch_threshold = 500
        eta1 = 0 if self.epoch < epoch_threshold else self.eta1

        if accuracy.item() > 0.9:
            alpha = 0
            beta = 0
            alpha_prime = self.alpha_prime
            beta_prime = self.beta_prime
            eta1 = 0
        else:
            alpha = self.alpha*10
            beta = self.beta*10
            eta1 = self.eta1
            alpha_prime = 0
            beta_prime = 0
        '''

        ## Approach 3
        '''
        loss_magnitudes = {
            "kl": kl_loss.detach(),
            "perturbed_kl": perturbed_kl_loss.detach(),
            "ce": ce_loss.detach(),
            "perturbed_ce": perturbed_ce_loss.detach(),
            "cosine": cosine_loss.detach(),
            "ortho": ortho_loss.detach(),
            "var": var_loss.detach(),
            "pixel": pixel_loss.detach(),
            "grad": grad_loss.detach(),
            "recon": stationary_loss.detach()
        }
        median_loss = torch.median(torch.tensor(list(loss_magnitudes.values())))
        median_loss = torch.max(median_loss, torch.tensor(1e-8))

        alpha = self.alpha * torch.clamp(median_loss / loss_magnitudes["kl"], 0.1, 10.0)
        beta = self.beta * torch.clamp(median_loss / loss_magnitudes["ce"], 0.1, 10.0)
        alpha_prime = self.alpha_prime * torch.clamp(median_loss / loss_magnitudes["perturbed_kl"], 0.1, 10.0)
        beta_prime = self.beta_prime * torch.clamp(median_loss / loss_magnitudes["perturbed_ce"], 0.1, 10.0)
        gamma = self.gamma * torch.clamp(median_loss / loss_magnitudes["cosine"], 0.1, 10.0)
        delta = self.delta * torch.clamp(median_loss / loss_magnitudes["ortho"], 0.1, 10.0)
        eta1 = self.eta1 * torch.clamp(median_loss / loss_magnitudes["var"], 0.1, 10.0)
        eta2 = self.eta2 * torch.clamp(median_loss / loss_magnitudes["pixel"], 0.1, 10.0)
        eta3 = self.eta3 * torch.clamp(median_loss / loss_magnitudes["grad"], 0.1, 10.0)
        eta4 = self.eta4 * torch.clamp(median_loss / loss_magnitudes["recon"], 0.1, 10.0)

        total_loss = (alpha * kl_loss + alpha_prime * perturbed_kl_loss + 
                    beta * ce_loss + beta_prime * perturbed_ce_loss + 
                    gamma * cosine_loss + delta * ortho_loss + 
                    eta1 * var_loss + eta2 * pixel_loss + eta3 * grad_loss + 
                    eta4 * stationary_loss)
        '''
            

        total_loss = (self.alpha * kl_loss + self.alpha_prime * perturbed_kl_loss + 
                     self.beta * ce_loss + self.beta_prime * perturbed_ce_loss + 
                     self.gamma * cosine_loss + self.delta * ortho_loss + 
                     self.eta1 * self.epoch * var_loss + self.eta2 * pixel_loss + self.eta3 * grad_loss + 
                     self.eta4 * stationary_loss)
                    
        total_loss.backward()
        self.optimizer.step()
        self.epoch += 1
        
        losses = {
            'total_loss': total_loss.item(), 'kl_loss': kl_loss.item(),
            'ce_loss': ce_loss.item(), 'perturbed_kl_loss': perturbed_kl_loss.item(),
            'perturbed_ce_loss': perturbed_ce_loss.item(), 'cosine_loss': cosine_loss.item(),
            'ortho_loss': ortho_loss.item(), 'var_loss': var_loss.item(),
            'pixel_loss': pixel_loss.item(), 'grad_loss': grad_loss.item(),
            'accuracy': accuracy.item(), 'perturb_acc': perturbation_accuracy.item(),
            'stationary_loss': stationary_loss.item()
        }
        
        for key in self.loss_history:
            if key in losses:
                self.loss_history[key].append(losses[key])
        
        return losses
    
    def train_epoch(self, batch_size, num_classes, steps_per_epoch):
        epoch_total_loss = 0
        with tqdm(total=steps_per_epoch, desc="Training", leave=False) as pbar:
            for step in range(steps_per_epoch):
                losses = self.train_step(batch_size)
                epoch_total_loss += losses['total_loss']
                pbar.set_postfix(loss=losses['total_loss'])
                pbar.update(1)
        return epoch_total_loss / steps_per_epoch, losses
    
    def generate_samples(self, num_samples, num_classes):
        self.generator.eval()
        samples = []
        with torch.no_grad():
            for class_idx in range(num_classes):
                labels = torch.zeros(num_samples, num_classes).to(self.device)
                labels[:, class_idx] = 1
                z = torch.randn(num_samples, self.latent_dim).to(self.device)
                samples.append(self.generator(z, labels))
        return samples
    
    def train(self, num_epochs, batch_size, num_classes, steps_per_epoch=100, 
              save_interval=10, results_dir='results', print_results=True, ssim_interval=1,
              patience=30, min_delta=0.01):
        os.makedirs(results_dir, exist_ok=True)
        # early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
        
        for epoch in range(num_epochs):
            avg_loss, last_losses = self.train_epoch(batch_size, num_classes, steps_per_epoch)
            if print_results:
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in last_losses.items()])
                print(f"Epoch [{epoch+1}/{num_epochs}] - {loss_str}")
            
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save_generated_samples(epoch+1, num_classes, results_dir)
                self.plot_loss_curves(epoch+1, results_dir)
                self.save_model(epoch+1, avg_loss, results_dir)

            if (epoch + 1) % ssim_interval == 0:
                ssim_results = self.calculate_ssim_scores(batch_size, num_classes)
                self.ssim_results.append(ssim_results)
                current_ssim = ssim_results['overall_avg_ssim']
                print(f"Epoch [{epoch+1}/{num_epochs}] - SSIM: {ssim_results['overall_avg_ssim']:.4f}")
                print(f"Avg of Max SSIM: {ssim_results['overall_max_ssim_avg']:.4f}")
                print(f"Reconstruction Percentage: {ssim_results['reconstruction_percentage']:.2f}%")
                self.ssim_list.append((epoch + 1, current_ssim))

                # Uncomment if you want to see SSIM per class
                # print("SSIM by class:")
                # for class_idx, ssim in ssim_results['by_class'].items():
                #     print(f"Class {class_idx} SSIM: {ssim:.4f}")
                # Check if training should be stopped
                # if early_stopping(epoch + 1, current_ssim):
                #     print(f"Training stopped early at epoch {epoch+1}")
                #     # Save best model
                #     best_epoch = early_stopping.best_epoch
                #     print(f"Loading best model from epoch {best_epoch}")
                #     checkpoint_path = os.path.join(results_dir, f"generator_checkpoint_epoch_{best_epoch}.pt")
                #     checkpoint = torch.load(checkpoint_path)
                #     self.generator.load_state_dict(checkpoint['generator_state_dict'])
                #     break
                print("-" * 50)
                
        self.save_generated_samples(num_epochs, num_classes, results_dir)
        self.plot_loss_curves(num_epochs, results_dir)
        self.plot_ssim_scores(results_dir)
        self.save_model(num_epochs, avg_loss, results_dir, is_final=True)
        return self.loss_history
    
    def normalize(self, images):
        if self.dataset_info['name'] == 'mnist':
            mean, std = 0.1307, 0.3081
            images = (images - mean) / std
        elif self.dataset_info['name'] == 'fashion_mnist':
            mean, std = 0.2860, 0.3530
            images = (images - mean) / std
        elif self.dataset_info['name'] == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            images = (images - mean) / std
        elif self.dataset_info['name'] == 'svhn':
            mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1)
            std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1)
            images = (images - mean) / std
        return images

    
    def denormalize(self, images):
        if self.dataset_info['name'] == 'mnist':
            mean, std = 0.1307, 0.3081
            images = images * std + mean
        elif self.dataset_info['name'] == 'fashion_mnist':
            mean, std = 0.2860, 0.3530
            images = images * std + mean
        elif self.dataset_info['name'] == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            images = images * std + mean
        elif self.dataset_info['name'] == 'svhn':
            mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1)
            std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1)
            images = images * std + mean
        return images

    def save_generated_samples(self, epoch, num_classes, results_dir):
        samples_per_class = 5
        samples = self.generate_samples(samples_per_class, num_classes)
        grid_size = min(num_classes, 10)
        
        fig, axes = plt.subplots(grid_size, samples_per_class, figsize=(samples_per_class * 2, grid_size * 2))
        for i in range(grid_size):
            for j in range(samples_per_class):
                ax = axes[i, j] if grid_size > 1 else axes[j]
                img = samples[i][j].cpu().detach().numpy().transpose(1, 2, 0).squeeze()
                # img = self.denormalize(img)
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                if j == 0:
                    ax.set_title(f"Class {i}")
                    
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'generated_samples_epoch_{epoch}.png'))
        plt.close()
    
    def plot_ssim_scores(self, results_dir):
        ssim_scores = self.ssim_list
        epochs, scores = zip(*ssim_scores)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, scores, marker='o', label='SSIM Score')
        plt.title('SSIM Overall Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ssim_scores_over_epochs.png'))
        plt.close()
        
    def plot_loss_curves(self, epoch, results_dir):
        key_losses = ['total_loss', 'kl_loss', 'ce_loss', 'perturbed_kl_loss', 'perturbed_ce_loss', 'stationary_loss']
        other_losses = ['cosine_loss', 'ortho_loss', 'var_loss', 'pixel_loss', 'grad_loss']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        for loss_name in key_losses:
            if loss_name in self.loss_history and self.loss_history[loss_name]:
                ax1.plot(self.loss_history[loss_name], label=loss_name)
        ax1.set_title('Primary Losses')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True)
        
        for loss_name in other_losses:
            if loss_name in self.loss_history and self.loss_history[loss_name]:
                ax2.plot(self.loss_history[loss_name], label=loss_name)
        ax2.set_title('Secondary Losses')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Loss Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'loss_curves_epoch_{epoch}.png'))
        plt.close()
    
    def save_model(self, epoch, loss, results_dir, is_final=False):
        filename = os.path.join(results_dir, "generator_final.pt" if is_final else f"generator_checkpoint_epoch_{epoch}.pt")
        save_data = self.generator.state_dict() if is_final else {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(save_data, filename)
        print(f"Model saved to {filename}")


class EarlyStopping:
    """Early stops the training if SSIM doesn't improve after a given patience."""
    def __init__(self, patience=30, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): If True, prints a message for each improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, ssim_score):
        # Higher SSIM is better
        if self.best_score is None:
            self.best_score = ssim_score
            self.best_epoch = epoch
            return False

        if ssim_score > self.best_score + self.min_delta:
            self.best_score = ssim_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"SSIM improved to {ssim_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"SSIM didn't improve from {self.best_score:.4f}. Counter {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best SSIM: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop