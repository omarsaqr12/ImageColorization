#!/usr/bin/env python3
"""
Combined Dataset Training Script for Colorization Models
Trains ECCV16 and SIGGRAPH17 models from scratch on CIFAR-10 + ImageNet (3k samples)
This approach provides faster training while maintaining good diversity
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skimage import color
import json
import glob
import random
from pathlib import Path
import matplotlib.pyplot as plt

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17

# Import perceptual loss modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceptual_loss import PerceptualLoss

class CIFAR10ColorizationDataset(Dataset):
    """CIFAR-10 dataset adapted for colorization training"""
    
    def __init__(self, root_dir='./data', train=True, transform=None):
        """
        Args:
            root_dir (str): Path to CIFAR-10 dataset directory
            train (bool): Whether to use training or test set
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.cifar10_dataset = datasets.CIFAR10(
            root=root_dir, 
            train=train, 
            download=True, 
            transform=None  # We'll handle transforms manually
        )
        self.transform = transform
        
        print(f"Loaded CIFAR-10 {'training' if train else 'test'} set: {len(self.cifar10_dataset)} images")
    
    def __len__(self):
        return len(self.cifar10_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.cifar10_dataset[idx]  # Ignore class label
        
        try:
            # Convert PIL image to RGB if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Convert to LAB color space
            image_np = np.array(image)
            lab_image = color.rgb2lab(image_np)
            
            # Extract L and AB channels
            l_channel = lab_image[:, :, 0]  # Grayscale
            ab_channels = lab_image[:, :, 1:]  # Color channels
            
            # Normalize L channel to [0, 1]
            l_channel = (l_channel + 50) / 100  # LAB L range is [0, 100]
            
            # Normalize AB channels to [-1, 1]
            ab_channels = ab_channels / 128  # LAB AB range is [-128, 128]
            
            # Convert to tensors
            l_tensor = torch.from_numpy(l_channel).float().unsqueeze(0)  # Add channel dimension
            ab_tensor = torch.from_numpy(ab_channels).float().permute(2, 0, 1)  # CHW format
            
            return l_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading CIFAR-10 image {idx}: {e}")
            # Return dummy tensors if loading fails
            dummy_l = torch.zeros(1, 256, 256)
            dummy_ab = torch.zeros(2, 256, 256)
            return dummy_l, dummy_ab

class ImageNetTrainingDataset(Dataset):
    """ImageNet dataset for training colorization models"""
    
    def __init__(self, root_dir, transform=None, max_samples=3000, train_split=0.8):
        """
        Args:
            root_dir (str): Path to ImageNet dataset directory
            transform (callable, optional): Optional transform to be applied on a sample
            max_samples (int, optional): Maximum number of samples to load (default 3000)
            train_split (float): Fraction of data to use for training
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        # Limit samples to 3k for faster training
        if len(self.image_paths) > max_samples:
            self.image_paths = random.sample(self.image_paths, max_samples)
        
        # Split into train/val
        random.shuffle(self.image_paths)
        split_idx = int(len(self.image_paths) * train_split)
        self.train_paths = self.image_paths[:split_idx]
        self.val_paths = self.image_paths[split_idx:]
        
        print(f"Found {len(self.image_paths)} images in ImageNet dataset (limited to {max_samples})")
        print(f"Training samples: {len(self.train_paths)}")
        print(f"Validation samples: {len(self.val_paths)}")
    
    def __len__(self):
        return len(self.train_paths)
    
    def __getitem__(self, idx):
        image_path = self.train_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Convert to LAB color space
            image_np = np.array(image)
            lab_image = color.rgb2lab(image_np)
            
            # Extract L and AB channels
            l_channel = lab_image[:, :, 0]  # Grayscale
            ab_channels = lab_image[:, :, 1:]  # Color channels
            
            # Normalize L channel to [0, 1]
            l_channel = (l_channel + 50) / 100  # LAB L range is [0, 100]
            
            # Normalize AB channels to [-1, 1]
            ab_channels = ab_channels / 128  # LAB AB range is [-128, 128]
            
            # Convert to tensors
            l_tensor = torch.from_numpy(l_channel).float().unsqueeze(0)  # Add channel dimension
            ab_tensor = torch.from_numpy(ab_channels).float().permute(2, 0, 1)  # CHW format
            
            return l_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy tensors if loading fails
            dummy_l = torch.zeros(1, 256, 256)
            dummy_ab = torch.zeros(2, 256, 256)
            return dummy_l, dummy_ab
    
    def get_val_dataset(self):
        """Get validation dataset"""
        return ImageNetValidationDataset(self.val_paths, self.transform)

class ImageNetValidationDataset(Dataset):
    """Validation dataset for ImageNet"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Convert to LAB color space
            image_np = np.array(image)
            lab_image = color.rgb2lab(image_np)
            
            # Extract L and AB channels
            l_channel = lab_image[:, :, 0]
            ab_channels = lab_image[:, :, 1:]
            
            # Normalize
            l_channel = (l_channel + 50) / 100
            ab_channels = ab_channels / 128
            
            # Convert to tensors
            l_tensor = torch.from_numpy(l_channel).float().unsqueeze(0)
            ab_tensor = torch.from_numpy(ab_channels).float().permute(2, 0, 1)
            
            return l_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading validation image {image_path}: {e}")
            dummy_l = torch.zeros(1, 256, 256)
            dummy_ab = torch.zeros(2, 256, 256)
            return dummy_l, dummy_ab

class ImageNetTrainer:
    def __init__(self, device='cpu', use_perceptual_loss=False, perceptual_weight=0.1,
                 perceptual_layers=['relu1_2', 'relu2_2', 'relu3_3'],
                 perceptual_norm='L1', vgg_type='vgg16'):
        self.device = device
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.training_history = {}
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
        
        # Perceptual loss (if enabled)
        if use_perceptual_loss:
            self.perceptual_loss_fn = PerceptualLoss(
                perceptual_layers=perceptual_layers,
                perceptual_norm=perceptual_norm,
                vgg_type=vgg_type,
                device=device
            )
            print(f"Perceptual loss enabled: weight={perceptual_weight}, layers={perceptual_layers}, norm={perceptual_norm}, vgg={vgg_type}")
        else:
            self.perceptual_loss_fn = None
    
    def _get_model_filename(self, model_name):
        """Generate model filename based on configuration to avoid overwriting"""
        parts = [model_name, 'imagenet']
        
        if self.use_perceptual_loss:
            parts.append('perceptual')
            parts.append(f"w{self.perceptual_weight}")
            parts.append(f"{self.perceptual_norm.lower()}")
        
        return '_'.join(parts) + '_best_model.pth'
        
    def setup_model(self, model_name, learning_rate=1e-4):
        """Setup model, optimizer, and scheduler"""
        print(f"Setting up {model_name} model...")
        
        if model_name == 'eccv16':
            model = eccv16(pretrained=False)  # Train from scratch
        elif model_name == 'siggraph17':
            model = siggraph17(pretrained=False)  # Train from scratch
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.schedulers[model_name] = scheduler
        self.training_history[model_name] = {
            'train_loss': [],
            'val_loss': [],
            'train_perceptual_loss': [],
            'val_perceptual_loss': [],
            'epochs': []
        }
        
        print(f"✅ {model_name} model setup complete")
    
    def lab_to_rgb(self, lab_tensor):
        """Convert LAB tensor to RGB tensor"""
        # Detach from computation graph if needed, then convert to numpy for skimage processing
        lab_np = lab_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        rgb_np = np.zeros_like(lab_np)
        
        for i in range(lab_np.shape[0]):
            # Denormalize LAB channels
            lab_denorm = lab_np[i].copy()
            lab_denorm[:, :, 0] = lab_denorm[:, :, 0] * 100 - 50  # L: [0,1] -> [0,100]
            lab_denorm[:, :, 1:] = lab_denorm[:, :, 1:] * 128  # AB: [-1,1] -> [-128,128]
            
            rgb_np[i] = color.lab2rgb(lab_denorm)
            # Ensure RGB values are in [0,1] range
            rgb_np[i] = np.clip(rgb_np[i], 0, 1)
        
        # Convert back to tensor and reattach to computation graph if original tensor required grad
        rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
        rgb_tensor = rgb_tensor.to(self.device)
        
        # If original tensor required grad, make the output require grad too
        if lab_tensor.requires_grad:
            rgb_tensor = rgb_tensor.requires_grad_(True)
        
        return rgb_tensor
    
    def train_epoch(self, model_name, train_loader, criterion):
        """Train for one epoch"""
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        
        model.train()
        total_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = 0
        
        for batch_idx, (l_channels, ab_channels) in enumerate(train_loader):
            l_channels = l_channels.to(self.device)
            ab_channels = ab_channels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_ab = model(l_channels)
            
            # Base loss (MSE on AB channels)
            base_loss = criterion(predicted_ab, ab_channels)
            
            # Perceptual loss (if enabled)
            perceptual_loss = torch.tensor(0.0, device=self.device)
            if self.use_perceptual_loss:
                # Reconstruct RGB images for perceptual loss
                predicted_lab = torch.cat([l_channels, predicted_ab], dim=1)
                target_lab = torch.cat([l_channels, ab_channels], dim=1)
                
                predicted_rgb = self.lab_to_rgb(predicted_lab)
                target_rgb = self.lab_to_rgb(target_lab)
                
                perceptual_loss = self.perceptual_loss_fn(predicted_rgb, target_rgb)
                loss = base_loss + self.perceptual_weight * perceptual_loss
            else:
                loss = base_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_perceptual_loss += perceptual_loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                if self.use_perceptual_loss:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, "
                          f"Base: {base_loss.item():.6f}, Perceptual: {perceptual_loss.item():.6f}")
                else:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches
        return avg_loss, avg_perceptual_loss
    
    def validate_epoch(self, model_name, val_loader, criterion):
        """Validate for one epoch"""
        model = self.models[model_name]
        
        model.eval()
        total_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for l_channels, ab_channels in val_loader:
                l_channels = l_channels.to(self.device)
                ab_channels = ab_channels.to(self.device)
                
                predicted_ab = model(l_channels)
                
                # Base loss (MSE on AB channels)
                base_loss = criterion(predicted_ab, ab_channels)
                
                # Perceptual loss (if enabled)
                perceptual_loss = torch.tensor(0.0, device=self.device)
                if self.use_perceptual_loss:
                    # Reconstruct RGB images for perceptual loss
                    predicted_lab = torch.cat([l_channels, predicted_ab], dim=1)
                    target_lab = torch.cat([l_channels, ab_channels], dim=1)
                    
                    predicted_rgb = self.lab_to_rgb(predicted_lab)
                    target_rgb = self.lab_to_rgb(target_lab)
                    
                    perceptual_loss = self.perceptual_loss_fn(predicted_rgb, target_rgb)
                    loss = base_loss + self.perceptual_weight * perceptual_loss
                else:
                    loss = base_loss
                
                total_loss += loss.item()
                total_perceptual_loss += perceptual_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches
        return avg_loss, avg_perceptual_loss
    
    def train_model(self, model_name, train_loader, val_loader, epochs=50):
        """Train a model"""
        print(f"\n🚀 Starting training for {model_name}...")
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_perceptual = self.train_epoch(model_name, train_loader, criterion)
            
            # Validate
            val_loss, val_perceptual = self.validate_epoch(model_name, val_loader, criterion)
            
            # Update learning rate
            self.schedulers[model_name].step(val_loss)
            
            # Record history
            self.training_history[model_name]['train_loss'].append(train_loss)
            self.training_history[model_name]['val_loss'].append(val_loss)
            self.training_history[model_name]['train_perceptual_loss'].append(train_perceptual)
            self.training_history[model_name]['val_perceptual_loss'].append(val_perceptual)
            self.training_history[model_name]['epochs'].append(epoch + 1)
            
            if self.use_perceptual_loss:
                print(f"Train Loss: {train_loss:.6f} (Perceptual: {train_perceptual:.6f}), "
                      f"Val Loss: {val_loss:.6f} (Perceptual: {val_perceptual:.6f})")
            else:
                print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_filename = self._get_model_filename(model_name)
                torch.save(self.models[model_name].state_dict(), model_filename)
                print(f"  💾 Saved best model (Val Loss: {val_loss:.6f})")
                print(f"  📁 Model saved as: {model_filename}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break
        
        print(f"✅ Training completed for {model_name}")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    def plot_training_history(self, model_name):
        """Plot training history"""
        history = self.training_history[model_name]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
        plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name.upper()} Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title(f'{model_name.upper()} Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_imagenet_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history plot saved: {model_name}_imagenet_training_history.png")
    
    def save_training_results(self):
        """Save training results"""
        results = {
            'metadata': {
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': self.device,
                'dataset': 'CIFAR-10 + ImageNet (3k samples)'
            },
            'training_history': self.training_history
        }
        
        with open('imagenet_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("📄 Training results saved: imagenet_training_results.json")

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get ImageNet path (optional - can skip if not available)
    imagenet_path = input("Enter the path to your ImageNet dataset (or press Enter to skip): ").strip()
    
    imagenet_available = False
    if imagenet_path and os.path.exists(imagenet_path):
        imagenet_available = True
        print(f"✅ ImageNet dataset found at: {imagenet_path}")
    else:
        print("⚠️  ImageNet dataset not found or skipped. Will use CIFAR-10 only.")
    
    # Get training parameters
    try:
        epochs = int(input("Enter number of epochs (default 30): ") or "30")
        batch_size = int(input("Enter batch size (default 16): ") or "16")
        use_perceptual = input("Use perceptual loss? (y/n, default n): ").strip().lower() == 'y'
        if use_perceptual:
            perceptual_weight = float(input("Perceptual loss weight (default 0.1): ") or "0.1")
        else:
            perceptual_weight = 0.1
    except ValueError:
        epochs = 30
        batch_size = 16
        use_perceptual = False
        perceptual_weight = 0.1
    
    print(f"\nTraining Configuration:")
    print(f"  Dataset: CIFAR-10 + {'ImageNet (3k samples)' if imagenet_available else 'ImageNet (skipped)'}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Perceptual Loss: {'Enabled' if use_perceptual else 'Disabled'}")
    if use_perceptual:
        print(f"    - Weight: {perceptual_weight}")
    
    # Prepare datasets
    print(f"\n📁 Preparing datasets...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    cifar10_train = CIFAR10ColorizationDataset(
        root_dir='./data',
        train=True,
        transform=transform
    )
    cifar10_val = CIFAR10ColorizationDataset(
        root_dir='./data',
        train=False,
        transform=transform
    )
    
    datasets_to_combine = [cifar10_train]
    val_datasets_to_combine = [cifar10_val]
    
    # Load ImageNet dataset if available
    if imagenet_available:
        print("Loading ImageNet dataset (3k samples)...")
        imagenet_dataset = ImageNetTrainingDataset(
            root_dir=imagenet_path,
            transform=transform,
            max_samples=3000
        )
        imagenet_val = imagenet_dataset.get_val_dataset()
        
        datasets_to_combine.append(imagenet_dataset)
        val_datasets_to_combine.append(imagenet_val)
    
    # Combine datasets
    combined_train_dataset = ConcatDataset(datasets_to_combine)
    combined_val_dataset = ConcatDataset(val_datasets_to_combine)
    
    print(f"✅ Combined dataset prepared")
    print(f"  Training samples: {len(combined_train_dataset)}")
    print(f"  Validation samples: {len(combined_val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"✅ Data loaders created")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = ImageNetTrainer(
        device=device,
        use_perceptual_loss=use_perceptual,
        perceptual_weight=perceptual_weight
    )
    
    # Train both models
    models_to_train = ['eccv16', 'siggraph17']
    
    for model_name in models_to_train:
        # Setup model
        trainer.setup_model(model_name)
        
        # Train model
        trainer.train_model(model_name, train_loader, val_loader, epochs=epochs)
        
        # Plot training history
        trainer.plot_training_history(model_name)
    
    # Save results
    trainer.save_training_results()
    
    print("\n🎉 Training completed!")
    print("Check the following files:")
    trainer_temp = ImageNetTrainer(
        device=device,
        use_perceptual_loss=use_perceptual,
        perceptual_weight=perceptual_weight
    )
    print(f"- {trainer_temp._get_model_filename('eccv16')}")
    print(f"- {trainer_temp._get_model_filename('siggraph17')}")
    print("- eccv16_imagenet_training_history.png")
    print("- siggraph17_imagenet_training_history.png")
    print("- imagenet_training_results.json")

if __name__ == "__main__":
    main()
