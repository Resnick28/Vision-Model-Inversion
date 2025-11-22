# EE-691

Privacy Preserving Properties of Vision Classifiers: Vulnerability Analysis of Inversion Attacks

## Abstract

The project investigates the privacy vulnerabilities of vision classifiers (MLPs, CNNs, and Vision Transformers) by attempting to reconstruct private training data from the trained model weights. We utilized Stationary Loss based on the Karush-Kuhn-Tucker (KKT) conditions of the training objective, which significantly enhances the ability to recover training samples compared to standard reconstruction losses.

## Key Findings:

Stationary Loss Dominance: The mathematical trace of gradient descent left in model weights is the strongest signal for data reconstruction.

Architecture Vulnerability: Simple MLPs are significantly more vulnerable to inversion attacks than CNNs or Vision Transformers due to weight sharing and attention mechanisms.

## Architecture

The framework consists of two main stages:

Target Model Training: Training a vision classifier (MLP, CNN, or ViT) on a target dataset (e.g., MNIST).

Network Inversion (Attack): Training a Conditioned Generator to reconstruct the original training data by optimizing a complex loss landscape derived from the fixed target classifier.

### The Conditioned Generator

The generator takes a latent vector $z$ and a class label $y$ as input and attempts to generate an image $\hat{x}$ that minimizes the specific inversion losses defined below.

## Loss Functions

The reconstruction is guided by a weighted sum of several loss functions (implemented in loss_fn.py):

$\mathcal{L}_{stationary}$ (Stationary Loss): Enforces that the reconstructed images sit at stable points where the gradient of the target model is minimized.

$\mathcal{L}_{class}$ (Cross-Entropy): Ensures the target model classifies the reconstruction with high confidence.

$\mathcal{L}_{prior}$ (Image Priors): Includes Total Variation (TV) and Pixel Range losses to encourage realistic image statistics.

$\mathcal{L}_{feat}$ (Feature Matching): Cosine similarity and Orthogonality losses on penultimate layer features to ensure diversity.

$\mathcal{L}_{grad}$ (Gradient Minimization): Encourages samples that yield small model gradients.

## Project Structure

```text
├── main.py                 # Entry point for training classifiers and running reconstruction
├── classifier_model.py     # Definitions for MLP, CNN, and VisionTransformer architectures
├── classifier_trainer.py   # Training loop for the target classifiers
├── generator_model.py      # Architecture of the Conditioned Generator
├── generator_trainer.py    # Training loop for the reconstruction (attack) phase
├── loss_fn.py              # Implementation of custom loss functions (TV, Orthogonality, etc.)
├── data_loader.py          # Data loading utilities (MNIST, FashionMNIST, CIFAR10, SVHN)
├── ssim_calc.py            # Metrics for evaluating reconstruction quality (SSIM)
├── hyperparam_search.py    # Bayesian Optimization for finding optimal loss weights
├── utils.py                # Helper functions (weight initialization, perturbation)
└── report.pdf              # Research paper detailing methodology and results
```

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-image scikit-optimize scipy
```


### 1. Train a Target Classifier

To train the model you wish to attack (e.g., an MLP on MNIST), configure main.py:

```python
# In main.py
cls_model_name = "mlp"  # Options: "mlp", "cnn", "vit"
train_size = 60000      # Number of images used to train the target
```

Run the script:

```bash
python main.py
```

This will save the best model as best_model_MLP.pt.

### 2. Run the Reconstruction Attack

Once the classifier is trained, main.py will automatically proceed to train the Generator to reconstruct the data. You can adjust the loss weights in main.py to test different hypotheses (e.g., disabling Stationary Loss).

# Hyperparameters in main.py
```python
trainer = ReconstructionTrainer(
    cls_model, generator_model, device, dataset_info,
    alpha=5,        # KL Divergence weight
    beta=10,        # Cross Entropy weight
    eta4=2,         # Stationary Loss weight (Crucial)
    ...
)
```

### 3. Hyperparameter Optimization (Optional)

To automatically find the best weights for the attack loss functions using Bayesian Optimization:

Uncomment the bayesian_optimization call in main.py.

Run main.py.

This uses ```skopt``` (Scikit-Optimize) to maximize the Structural Similarity Index (SSIM) of the reconstructed images.

## Results & Evaluation

The quality of reconstruction is evaluated using SSIM (Structural Similarity Index).

MLP: High reconstruction quality. The stationary loss effectively recovers digit shapes.

CNN/ViT: Lower reconstruction quality, demonstrating higher privacy resilience.
