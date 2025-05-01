# ===== MODEL ARCHITECTURES =====

import torch.nn as nn
import torch

class MLP(nn.Module):
    """
    Enhanced Multi-Layer Perceptron with dropout and batch normalization
    """
    def __init__(self, input_size=784, hidden_layers=[512, 256], num_classes=10, dropout_rate=0.4):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Build the network dynamically based on hidden_layers
        layers = []
        prev_size = input_size
        
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(0.2))
            # layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        # Output layer
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)
    
    def get_penultimate(self, x):
        """Return the features from the penultimate layer for inversion"""
        x = self.flatten(x)
        return self.feature_extractor(x)
    
    def forward(self, x):
        features = self.get_penultimate(x)
        return self.classifier(features)

class CNN(nn.Module):
    """
    Enhanced Convolutional Neural Network with proper architecture
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size of flattened features after convolutions and pooling
        self.feature_size = self._get_feature_size(in_channels)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def _get_feature_size(self, in_channels):
        """Calculate the size of flattened features after convolutions"""
        # For 28x28 input (MNIST/FashionMNIST)
        if in_channels == 1:
            return 128 * 3 * 3  # After 3 max-pooling operations: 28 -> 14 -> 7 -> 3
        # For 32x32 input (CIFAR-10/SVHN)
        else:
            return 128 * 4 * 4  # After 3 max-pooling operations: 32 -> 16 -> 8 -> 4
    
    def get_penultimate(self, x):
        """Return the features from the penultimate layer for inversion"""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier[:-1](x)  # All layers except the final linear
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VisionTransformer(nn.Module):
    """
    Simple Vision Transformer for image classification
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64, 
                 num_heads=4, mlp_ratio=4, num_layers=2, num_classes=10, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Number of patches
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.pre_head = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def get_penultimate(self, x):
        """Return the features from the penultimate layer for inversion"""
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # MLP head
        x = self.norm(x[:, 0])
        x = self.pre_head(x)
        return x
    
    def forward(self, x):
        x = self.get_penultimate(x)
        x = self.dropout(x)
        x = self.head(x)
        return x