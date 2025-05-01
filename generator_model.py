# ===== GENERATOR FOR RECONSTRUCTION =====

import torch.nn as nn
import torch

class Generator1(nn.Module):
    def __init__(self, nz = 100, ngf = 64, nc=1, n_classes=10):  # Default nc=1 for grayscale output, n_classes=10
        super(Generator1, self).__init__()
        self.nz = nz
        self.n_classes = n_classes
        self.latent_dim = nz

        self.embedding = nn.Linear(n_classes, nz)

        # Adjust layers before concatenation to take nz*2 channels
        self.layers_before_concat = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf, 4, 1, 0),  # Adjusted input channels to nz*2
            nn.BatchNorm2d(ngf),
            #nn.Dropout2d(0.1),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf * 2, 4, 1, 0),
            nn.BatchNorm2d(ngf * 2),
            #nn.Dropout2d(0.1),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(ngf * 4),
            #nn.Dropout2d(0.1),
            nn.ReLU(True),
        )

        # Layers after concatenation remain the same
        self.layers_after_concat = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + 1, ngf * 8, 4, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.1),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout2d(0.1),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout2d(0.1),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, 0),
            nn.BatchNorm2d(ngf),
            nn.Dropout2d(0.1),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        latent_vector = z
        conditioning_vector = labels
        # Get the argmax of the conditioning vector
        argmax_index = torch.argmax(conditioning_vector, dim=1)

        # Create a class matrix based on the argmax index
        class_matrix = torch.zeros((latent_vector.size(0), self.n_classes, self.n_classes)).to(latent_vector.device)
        for i, idx in enumerate(argmax_index):
            class_matrix[i, idx, :] = 1
            class_matrix[i, :, idx] = 1
            
        embedding_vector = self.embedding(conditioning_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = latent_vector.unsqueeze(2).unsqueeze(3)
        concat1 = torch.cat([latent_vector, embedding_vector], dim=1)
        upsample1 = self.layers_before_concat(concat1)
        class_matrix = class_matrix.unsqueeze(1)
        concat2 = torch.cat([upsample1, class_matrix], dim=1)
        upsample2 = self.layers_after_concat(concat2)

        return upsample2



class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_size=28, channels=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels
        
        # Initial size after first upsampling
        self.init_size = img_size // 4

        self.process_labels = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Vector conditioning
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # First upsampling blocks
        self.conv_blocks1 = nn.Sequential(
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128, affine=True),
            nn.Upsample(size=(10, 10), mode='nearest'),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.Dropout(0.1)
        )

        
        # Matrix conditioning layers
        self.matrix_cond = nn.Sequential(
            nn.Conv2d(128 + 1, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Second upsampling blocks
        self.conv_blocks2 = nn.Sequential(
            nn.Upsample(size=(28, 28), mode='nearest'),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            # nn.Tanh()  # Output in range [-1, 1]
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def create_hot_conditioning_matrix(self, labels):
        """
        Creates the Hot Conditioning Matrix as described in the paper.
        For each label, creates an N×N matrix where the corresponding row is set to 1.
        """
        batch_size = labels.size(0)
        # Get the label index for each item in the batch
        label_indices = torch.argmax(labels, dim=1)
        
        # Create N×N matrices for each sample in the batch
        matrices = torch.zeros(batch_size, self.num_classes, self.num_classes, device=labels.device)
        
        # Fill the matrices according to the label index
        for i in range(batch_size):
            # Set the entire row corresponding to the label to 1
            matrices[i, label_indices[i], :] = 1.0
            matrices[i, :, label_indices[i]] = 1.0
            
        return matrices
    
    def forward(self, z, labels):
        # Vector conditioning
        labels = self.process_labels(labels)
        z = torch.cat([z, labels], dim=1)
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks1(out)
        
        # Matrix conditioning
        hot_matrices = self.create_hot_conditioning_matrix(labels)
        
        # Reshape the hot matrices to match spatial dimensions for concatenation
        hot_matrices_expanded = hot_matrices.unsqueeze(1).expand(-1, 1, 10, 10)
        
        # Concatenate feature maps with hot matrices
        out = torch.cat([out, hot_matrices_expanded], dim=1)
        out = self.matrix_cond(out)
        
        # Final layers
        out = self.conv_blocks2(out)
        
        return out