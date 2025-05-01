# ===== TRAINER CLASSES =====

import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm

class ClassifierTrainer:
    """
    Trainer class for the classifiers
    """
    def __init__(self, model, train_loader, test_loader, device,
                 learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data, target in tqdm(self.train_loader, desc="Training", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self):
        """Evaluate the model on the test set"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Calculate average loss and accuracy
        avg_val_loss = val_loss / len(self.test_loader)
        val_accuracy = 100.0 * correct / total
        
        self.val_losses.append(avg_val_loss)
        self.val_accs.append(val_accuracy)
        
        return avg_val_loss, val_accuracy
    
    def train(self, num_epochs=10):
        """Train the model for specified number of epochs"""
        best_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f'best_model_{self.model.__class__.__name__}.pt')
        
        print(f'Best validation accuracy: {best_acc:.2f}%')
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs
    
    def plot_training_metrics(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accs, label='Training Accuracy')
        ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()

        save_path = "results/classifier_loss.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        plt.savefig(save_path)  # Save the figure
        plt.close(fig)  # Close to free memory