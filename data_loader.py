# ===== DATA LOADING AND PREPARATION =====
# This module handles the loading and preparation of datasets for training and testing.

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random

def load_dataset(dataset_name='mnist', batch_size=100, train_size=60000):
    """
    Parameters
    ----------
    dataset_name : str, default='mnist'
        Name of the dataset to load. Options: 'mnist', 'fashion_mnist', 'cifar10', 'svhn'.
    batch_size : int, default=100
        Number of samples per batch in the data loaders.
    train_size : int, default=60000
        Number of samples to include in the training set. If less than the total available
        training samples, a balanced subset will be created.
        
    Returns
    -------
    tuple
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test set
        - dataset_info (dict): Dictionary with dataset metadata including:
          - name: Dataset name
          - channels: Number of image channels
          - img_size: Image dimensions
          - num_classes: Number of classification classes
          - train_size: Number of training samples
          - test_size: Number of test samples
          
    Raises
    ------
    ValueError
        If an unsupported dataset name is provided.
    """
    random.seed(42)

    # Dataset configs as [transform, class, channels, img_size]
    configs = {
        'mnist': [
            transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            torchvision.datasets.MNIST, 1, 28
        ],
        'fashion_mnist': [
            transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]),
            torchvision.datasets.FashionMNIST, 1, 28
        ],
        'cifar10': [
            transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
            torchvision.datasets.CIFAR10, 3, 32
        ],
        'svhn': [
            transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
            torchvision.datasets.SVHN, 3, 32
        ]
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    transform, dataset_class, channels, img_size = configs[dataset_name]
    is_svhn = dataset_name == 'svhn'
    
    # Configure dataset loading parameters
    train_kwargs = {'root': './data', 'transform': transform}
    test_kwargs = {'root': './data', 'transform': transform}
    if is_svhn:
        train_kwargs['split'], test_kwargs['split'] = 'train', 'test'
    else:
        train_kwargs['train'], test_kwargs['train'] = True, False
    
    train_full = dataset_class(**train_kwargs)
    test_dataset = dataset_class(**test_kwargs)
    
    # Create balanced subset if needed
    if train_size < len(train_full):
        if not is_svhn:
            # Get targets and create balanced subset
            targets = getattr(train_full, 'targets', getattr(train_full, 'train_labels', None))
            targets = targets.numpy() if hasattr(targets, 'numpy') else targets
            
            # Group indices by class
            by_class = {}
            for i, t in enumerate(targets):
                t_item = t.item() if hasattr(t, 'item') else t
                by_class.setdefault(t_item, []).append(i)
            
            # Select balanced subset
            indices = []
            n_classes = len(by_class)
            per_class = train_size // n_classes
            for i, (_, idxs) in enumerate(by_class.items()):
                samples = per_class + (1 if i < train_size % n_classes else 0)
                indices.extend(random.sample(idxs, min(samples, len(idxs))))
            train_dataset = Subset(train_full, indices[:train_size])
        else:
            train_dataset = Subset(train_full, random.sample(range(len(train_full)), train_size))
    else:
        train_dataset = train_full
    
    # Create loaders with shared parameters
    loader_args = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True}
    return (
        DataLoader(train_dataset, shuffle=True, **loader_args),
        DataLoader(test_dataset, shuffle=False, **loader_args),
        {'name': dataset_name, 'channels': channels, 'img_size': img_size, 'num_classes': 10, 
         'train_size': len(train_dataset), 'test_size': len(test_dataset)}
    )