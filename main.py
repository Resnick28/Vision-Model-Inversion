# Privacy-Preserving Properties of Vision Classifiers

import numpy as np
import torch
import torch.nn as nn
from data_loader import load_dataset
from visualise_dataset import visualize_dataset_samples
from classifier_model import MLP, CNN, VisionTransformer
from classifier_trainer import ClassifierTrainer
from generator_model import Generator, Generator1
from utils import weights_initialization_gen_custom
from generator_trainer import ReconstructionTrainer
from ssim_calc import generate_and_evaluate_mnist_ssim
from hyperparam_search import bayesian_optimization
import pickle

# Setup environment
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 

# config
train_size=100
dataset = "mnist"
cls_model_name = "mlp"  # "cnn" or "vit"
num_epochs_gen = 100
num_classes=10

# Data preparation
train_loader, test_loader, dataset_info = load_dataset(dataset_name=dataset, batch_size=100, train_size=train_size)
print(dataset_info)
visualize_dataset_samples(train_loader, dataset_info)

# Train classifier
if cls_model_name == "mlp":
    cls_model = MLP().to(device)
elif cls_model_name == "cnn":
    cls_model = CNN().to(device)
elif cls_model_name == "vit":
    cls_model = VisionTransformer().to(device)
classifier_trainer = ClassifierTrainer(cls_model, train_loader, test_loader, device)
# classifier_trainer.train(num_epochs=30)
# classifier_trainer.plot_training_metrics()
state_dict = torch.load(f'best_model_{cls_model.__class__.__name__}.pt')
cls_model.load_state_dict(state_dict)
cls_model.eval()

# Hyper parameter search
params = [{"alpha": 1, "alpha_prime": 1, "beta": 1, "beta_prime": 1, 
           "gamma": 1, "delta": 1, "eta1": 1, "eta2": 1, "eta3": 1, "eta4": 0},]
bayesian_optimization(cls_model, device, dataset_info, train_loader, initial_points = params)
exit(0)


# Train generator for privacy reconstruction
generator_model = Generator1().to(device)
# generator_model = Generator().to(device)
generator_model.apply(weights_initialization_gen_custom)
# print(summary( generator_model, input_data={"z": torch.randn(1, 100).to(device), "labels": torch.randn(1, 10).to(device)}))
# Hyperparameter to loss mapping:
    # alpha         -> kl_loss (Kullback-Leibler divergence loss)
    # alpha_prime   -> perturbed_kl_loss (Perturbed KL divergence loss)
    # beta          -> ce_loss (Cross-entropy loss)
    # beta_prime    -> perturbed_ce_loss (Perturbed cross-entropy loss) 
    # gamma         -> cosine_loss (Cosine similarity/distance loss)
    # delta         -> ortho_loss (Orthogonality loss)
    # eta1          -> var_loss (Variance loss)
    # eta2          -> pixel_loss (Pixel-level loss)
    # eta3          -> grad_loss (Gradient loss)
    # eta4          -> reconstruction_loss (Reconstruction loss)
trainer = ReconstructionTrainer(cls_model, generator_model, device, 
                                dataset_info, train_loader=train_loader,
                    # perturb_strength=1.2, lambda_min=0.01,
                    # alpha=8, alpha_prime=4, beta=4, 
                    # beta_prime=8, gamma=4, delta=2, 
                    # eta1=0, eta2=0, eta3=8, eta4=0)

                    # perturb_strength=1, lambda_min=0.01,
                    # alpha=0, alpha_prime=0, beta=0, 
                    # beta_prime=0, gamma=0, delta=0, 
                    # eta1=0, eta2=0, eta3=0, eta4=1)

                    # perturb_strength=1.2, lambda_min=0.01,
                    # alpha=5, alpha_prime=5, beta=10, 
                    # beta_prime=10, gamma=10, delta=7, 
                    # eta1=2, eta2=0, eta3=2, eta4=2)

                    perturb_strength=1.2, lambda_min=0.01,
                    alpha=5, alpha_prime=5, beta=10, 
                    beta_prime=10, gamma=10, delta=7, 
                    eta1=2, eta2=0, eta3=2, eta4=2)
# alpha=8.422848, alpha_prime=4.497541, beta=3.951502, beta_prime=9.266589, 
# gamma=7.272720, delta=3.265408, 
# eta1=5.704440, eta2=5.208343, eta3=9.611720, eta4=8.445338

# alpha: 5.283144, alpha_prime: 4.653375, beta: 9.940105, beta_prime: 9.727171,
# gamma: 9.727725, delta: 7.183189
# eta1: 2.455994, eta2: 2.581795, eta3: 2.697019, eta4: 2.372256

trainer.train(num_epochs=num_epochs_gen, batch_size=num_classes*train_size, num_classes=num_classes)

filename = f"results_{dataset}_{cls_model_name}_train{train_size}.pkl"
with open(filename, 'wb') as f:
    pickle.dump(trainer.ssim_results, f)

# Evaluate reconstruction quality
# mean_ssim, class_mean_ssims = generate_and_evaluate_mnist_ssim(
#     generator_model, train_loader, device,
#     latent_dim=100, num_classes=10, samples_per_class=50, visualize=False
# )