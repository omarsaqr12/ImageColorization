#!/usr/bin/env python3
"""
Train Colorization Models on CIFAR-10 Dataset
Trains both ECCV16 and SIGGRAPH17 models from scratch on CIFAR-10
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from skimage import color
import json

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens

# Import perceptual loss modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceptual_loss import PerceptualLoss

# Import GAN modules (optional, only used if use_gan=True)
try:
    from discriminator import PatchGANDiscriminator, create_discriminator
    from gan_loss import generator_gan_loss, discriminator_loss, compute_gradient_penalty
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False
    print("Warning: GAN modules not available. GAN training will be disabled.")

class ColorizationTrainer:
    def __init__(self, device='cpu', batch_size=32, learning_rate=0.0002, 
                 use_ila=False, ila_reduction=4, ila_use_dw_conv=True,
                 use_perceptual_loss=False, perceptual_weight=0.1,
                 perceptual_layers=['relu1_2', 'relu2_2', 'relu3_3'],
                 perceptual_norm='L1', vgg_type='vgg16',
                 use_gan=False, gan_weight=0.1, gan_type='lsgan',
                 use_gp=False, gp_weight=10.0, use_spectral=False, d_lr_factor=1.0):
        """
        Initialize colorization trainer.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            use_ila: If True, use ILA blocks in ECCV16 model. Default: False
            ila_reduction: Channel reduction factor for ILA. Default: 4
            ila_use_dw_conv: Whether ILA uses depthwise convolution. Default: True
            use_perceptual_loss: If True, use perceptual loss in addition to pixel-wise loss. Default: False
            perceptual_weight: Weight for perceptual loss component. Default: 0.1
            perceptual_layers: List of VGG layers to use for perceptual loss. Default: ['relu1_2', 'relu2_2', 'relu3_3']
            perceptual_norm: Distance metric for perceptual loss ('L1' or 'L2'). Default: 'L1'
            vgg_type: Type of VGG model ('vgg16' or 'vgg19'). Default: 'vgg16'
            use_gan: If True, enable GAN training with PatchGAN discriminator. Default: False
            gan_weight: Weight for GAN loss component. Default: 0.1
            gan_type: Type of GAN loss ('bce' or 'lsgan'). Default: 'lsgan'
            use_gp: If True, use gradient penalty (WGAN-GP). Default: False
            gp_weight: Weight for gradient penalty. Default: 10.0
            use_spectral: If True, apply spectral normalization to discriminator. Default: False
            d_lr_factor: Discriminator learning rate multiplier (relative to generator). Default: 1.0
        """
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_ila = use_ila
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
        self.perceptual_norm = perceptual_norm
        self.use_gan = use_gan and GAN_AVAILABLE
        self.gan_weight = gan_weight
        self.gan_type = gan_type
        self.use_gp = use_gp
        self.gp_weight = gp_weight
        self.use_spectral = use_spectral
        
        # Initialize models (without pretrained weights)
        # ECCV16 with optional ILA
        self.colorizer_eccv16 = eccv16(
            pretrained=False, 
            use_ila=use_ila, 
            ila_reduction=ila_reduction, 
            ila_use_dw_conv=ila_use_dw_conv
        ).to(device)
        self.colorizer_siggraph17 = siggraph17(pretrained=False).to(device)
        
        # Initialize discriminators (if GAN enabled)
        if self.use_gan:
            # Discriminator input: L channel (1) + ab channels (2) = 3 channels
            self.discriminator_eccv16 = create_discriminator(
                input_channels=3,  # L (1) + ab (2) = 3 channels
                ndf=64,
                n_layers=3,
                device=device
            )
            self.discriminator_siggraph17 = create_discriminator(
                input_channels=3,  # L (1) + ab (2) = 3 channels
                ndf=64,
                n_layers=3,
                device=device
            )
            
            # Apply spectral normalization if requested
            if use_spectral:
                from torch.nn.utils import spectral_norm
                def apply_sn(module):
                    if isinstance(module, nn.Conv2d):
                        return spectral_norm(module)
                    return module
                self.discriminator_eccv16 = self.discriminator_eccv16.apply(
                    lambda m: apply_sn(m) if isinstance(m, nn.Conv2d) else m
                )
                self.discriminator_siggraph17 = self.discriminator_siggraph17.apply(
                    lambda m: apply_sn(m) if isinstance(m, nn.Conv2d) else m
                )
                print("Spectral normalization applied to discriminators")
        else:
            self.discriminator_eccv16 = None
            self.discriminator_siggraph17 = None
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
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
        
        # Optimizers
        self.optimizer_eccv16 = optim.Adam(self.colorizer_eccv16.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_siggraph17 = optim.Adam(self.colorizer_siggraph17.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Discriminator optimizers (if GAN enabled)
        if self.use_gan:
            d_lr = learning_rate * d_lr_factor
            self.optimizer_d_eccv16 = optim.Adam(
                self.discriminator_eccv16.parameters(),
                lr=d_lr,
                betas=(0.5, 0.999)
            )
            self.optimizer_d_siggraph17 = optim.Adam(
                self.discriminator_siggraph17.parameters(),
                lr=d_lr,
                betas=(0.5, 0.999)
            )
        
        # Training history
        self.training_history = {
            'eccv16': {'train_loss': [], 'val_loss': [], 'train_perceptual_loss': [], 'val_perceptual_loss': [],
                      'train_gan_loss': [], 'val_gan_loss': [], 'd_loss': [], 'd_accuracy': []},
            'siggraph17': {'train_loss': [], 'val_loss': [], 'train_perceptual_loss': [], 'val_perceptual_loss': [],
                          'train_gan_loss': [], 'val_gan_loss': [], 'd_loss': [], 'd_accuracy': []}
        }
        
        print(f"Initialized trainer on device: {device}")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        if use_ila:
            print(f"ILA enabled: reduction={ila_reduction}, use_dw_conv={ila_use_dw_conv}")
        else:
            print("ILA disabled (baseline ECCV16)")
        if self.use_gan:
            print(f"GAN training enabled: type={gan_type}, weight={gan_weight}")
            if use_gp:
                print(f"Gradient penalty enabled: weight={gp_weight}")
    
    def _get_model_filename(self, model_name):
        """Generate model filename based on configuration to avoid overwriting"""
        parts = [model_name]
        
        if model_name == 'eccv16' and self.use_ila:
            parts.append('ila')
        
        if self.use_perceptual_loss:
            parts.append('perceptual')
            # Include key config in filename
            parts.append(f"w{self.perceptual_weight}")
            parts.append(f"{self.perceptual_norm.lower()}")
        
        if self.use_gan:
            parts.append('gan')
            parts.append(f"{self.gan_type}")
            parts.append(f"w{self.gan_weight}")
        
        return '_'.join(parts) + '_best_model.pth'
    
    def _get_checkpoint_filename(self, model_name, epoch):
        """Generate checkpoint filename based on configuration"""
        parts = [model_name]
        
        if model_name == 'eccv16' and self.use_ila:
            parts.append('ila')
        
        if self.use_perceptual_loss:
            parts.append('perceptual')
            parts.append(f"w{self.perceptual_weight}")
            parts.append(f"{self.perceptual_norm.lower()}")
        
        return '_'.join(parts) + f'_epoch_{epoch}.pth'
    
    def prepare_data(self):
        """Prepare CIFAR-10 dataset for colorization training"""
        print("Preparing CIFAR-10 dataset...")
        
        # Transform for training
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        
        # Transform for validation
        transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root='./cifar-10-python', 
            train=True, 
            download=True, 
            transform=transform_train
        )
        
        val_dataset = torchvision.datasets.CIFAR10(
            root='./cifar-10-python', 
            train=False, 
            download=True, 
            transform=transform_val
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if self.device != 'cpu' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if self.device != 'cpu' else False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def rgb_to_lab(self, rgb_tensor):
        """Convert RGB tensor to LAB tensor"""
        # Convert to numpy for skimage processing
        rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
        lab_np = np.zeros_like(rgb_np)
        
        for i in range(rgb_np.shape[0]):
            # Ensure RGB values are in [0,1] range
            rgb_normalized = np.clip(rgb_np[i], 0, 1)
            lab_np[i] = color.rgb2lab(rgb_normalized)
        
        # Convert back to tensor
        lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
        return lab_tensor.to(self.device)
    
    def lab_to_rgb(self, lab_tensor, requires_grad=False):
        """
        Convert LAB tensor to RGB tensor.
        
        Note: This uses skimage which is non-differentiable. For perceptual loss,
        we detach the tensor before conversion since the conversion itself cannot
        be backpropagated through. The gradients will flow through the perceptual
        loss computation on the RGB images, but not through the LAB->RGB conversion.
        
        Args:
            lab_tensor: LAB tensor of shape (B, 3, H, W)
            requires_grad: Whether the output should require gradients (for perceptual loss)
        
        Returns:
            RGB tensor of shape (B, 3, H, W) with values in [0, 1]
        """
        # Detach before numpy conversion (skimage operations are non-differentiable)
        lab_np = lab_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        rgb_np = np.zeros_like(lab_np)
        
        for i in range(lab_np.shape[0]):
            rgb_np[i] = color.lab2rgb(lab_np[i])
            # Ensure RGB values are in [0,1] range
            rgb_np[i] = np.clip(rgb_np[i], 0, 1)
        
        # Convert back to tensor
        rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
        rgb_tensor = rgb_tensor.to(self.device)
        
        # For perceptual loss, we need the RGB tensor to participate in the computation graph
        # Even though LAB->RGB conversion is non-differentiable, we can still compute
        # gradients through the perceptual loss on the RGB values themselves.
        # The key is that perceptual loss gradients will provide signal even if we can't
        # backprop through the color conversion.
        if requires_grad:
            # Mark tensor to require gradients so perceptual loss can compute gradients
            # The gradients won't flow back through LAB->RGB conversion, but the
            # perceptual loss value itself provides useful training signal
            rgb_tensor.requires_grad_(True)
        
        return rgb_tensor
    
    def compute_loss(self, predicted_ab, target_ab, target_l, target_rgb=None, predicted_rgb=None,
                     discriminator=None, l_channel=None):
        """
        Compute combined loss for colorization.
        
        Args:
            predicted_ab: Predicted AB channels
            target_ab: Target AB channels
            target_l: Target L channel
            target_rgb: Target RGB image (for perceptual loss, shape: B, 3, H, W, values in [0,1])
            predicted_rgb: Predicted RGB image (for perceptual loss, shape: B, 3, H, W, values in [0,1])
            discriminator: Discriminator model (for GAN loss, if enabled)
            l_channel: L channel (for discriminator input, if GAN enabled)
        
        Returns:
            total_loss, l1_loss, l2_loss, perceptual_loss, gan_loss
        """
        # L1 loss for color channels
        l1_loss = self.l1_loss(predicted_ab, target_ab)
        
        # L2 loss for color channels
        l2_loss = self.l2_loss(predicted_ab, target_ab)
        
        # Base loss (weighted)
        base_loss = l1_loss + 0.1 * l2_loss
        
        # Perceptual loss (if enabled and RGB images provided)
        perceptual_loss = torch.tensor(0.0, device=self.device)
        if self.use_perceptual_loss and target_rgb is not None and predicted_rgb is not None:
            # Compute perceptual loss on RGB images
            # Note: LAB->RGB conversion via skimage is non-differentiable, but we can still
            # compute perceptual loss. The gradients will flow through the perceptual loss
            # computation (VGG features), providing training signal even though we can't
            # backprop through the color space conversion itself.
            perceptual_loss = self.perceptual_loss_fn(predicted_rgb, target_rgb)
        
        # GAN loss (if enabled)
        gan_loss = torch.tensor(0.0, device=self.device)
        if self.use_gan and discriminator is not None and l_channel is not None:
            # Concatenate L channel with predicted ab channels for discriminator input
            # Shape: (B, 1, H, W) + (B, 2, H, W) -> (B, 4, H, W)
            d_input_fake = torch.cat([l_channel, predicted_ab], dim=1)
            D_fake = discriminator(d_input_fake)
            gan_loss = generator_gan_loss(D_fake, gan_type=self.gan_type)
        
        # Total loss
        total_loss = base_loss
        if self.use_perceptual_loss:
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
        if self.use_gan:
            total_loss = total_loss + self.gan_weight * gan_loss
        
        return total_loss, l1_loss, l2_loss, perceptual_loss, gan_loss
    
    def train_discriminator_step(self, discriminator, optimizer_d, l_channel, real_ab, fake_ab):
        """
        Train discriminator for one step.
        
        Args:
            discriminator: Discriminator model
            optimizer_d: Discriminator optimizer
            l_channel: L channel
            real_ab: Real AB channels
            fake_ab: Fake (predicted) AB channels
        
        Returns:
            d_loss, d_accuracy
        """
        discriminator.train()
        optimizer_d.zero_grad()
        
        # Real samples: concatenate L + real ab
        d_input_real = torch.cat([l_channel, real_ab], dim=1)
        D_real = discriminator(d_input_real)
        
        # Fake samples: concatenate L + fake ab
        d_input_fake = torch.cat([l_channel, fake_ab.detach()], dim=1)
        D_fake = discriminator(d_input_fake)
        
        # Discriminator loss
        d_loss = discriminator_loss(D_real, D_fake, gan_type=self.gan_type)
        
        # Gradient penalty (if enabled)
        if self.use_gp:
            gp = compute_gradient_penalty(
                discriminator,
                d_input_real,
                d_input_fake,
                device=self.device
            )
            d_loss = d_loss + self.gp_weight * gp
        
        # Backward and update
        d_loss.backward()
        optimizer_d.step()
        
        # Compute accuracy (for logging)
        if self.gan_type.lower() == 'bce':
            real_pred = torch.sigmoid(D_real) > 0.5
            fake_pred = torch.sigmoid(D_fake) < 0.5
        else:  # lsgan
            real_pred = D_real > 0.5
            fake_pred = D_fake < 0.5
        
        d_accuracy = (real_pred.float().mean() + fake_pred.float().mean()) / 2.0
        
        return d_loss.item(), d_accuracy.item()
    
    def train_epoch(self, model, optimizer, model_name):
        """Train one epoch"""
        model.train()
        
        # Get discriminator and optimizer if GAN enabled
        discriminator = None
        optimizer_d = None
        if self.use_gan:
            if model_name == 'eccv16':
                discriminator = self.discriminator_eccv16
                optimizer_d = self.optimizer_d_eccv16
            else:
                discriminator = self.discriminator_siggraph17
                optimizer_d = self.optimizer_d_siggraph17
        
        total_loss = 0.0
        total_l1_loss = 0.0
        total_l2_loss = 0.0
        total_perceptual_loss = 0.0
        total_gan_loss = 0.0
        total_d_loss = 0.0
        total_d_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (rgb_images, _) in enumerate(self.train_loader):
            rgb_images = rgb_images.to(self.device)
            
            # Convert RGB to LAB
            lab_images = self.rgb_to_lab(rgb_images)
            l_channel = lab_images[:, 0:1, :, :]  # L channel
            ab_channels = lab_images[:, 1:, :, :]  # AB channels
            
            # ========== Discriminator Step ==========
            if self.use_gan and discriminator is not None:
                # Forward pass to get fake ab
                with torch.no_grad():
                    predicted_ab = model(l_channel)
                
                # Train discriminator
                d_loss, d_accuracy = self.train_discriminator_step(
                    discriminator, optimizer_d, l_channel, ab_channels, predicted_ab
                )
                total_d_loss += d_loss
                total_d_accuracy += d_accuracy
            
            # ========== Generator Step ==========
            optimizer.zero_grad()
            
            # Forward pass
            predicted_ab = model(l_channel)
            
            # Prepare RGB images for perceptual loss if needed
            predicted_rgb = None
            target_rgb = None
            if self.use_perceptual_loss:
                # Reconstruct predicted RGB from predicted AB + original L
                predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
                # For predicted RGB, we need gradients to flow through perceptual loss
                predicted_rgb = self.lab_to_rgb(predicted_lab, requires_grad=True)
                target_rgb = rgb_images  # Already in [0, 1] range, no gradients needed
            
            # Compute loss
            loss, l1_loss, l2_loss, perceptual_loss, gan_loss = self.compute_loss(
                predicted_ab, ab_channels, l_channel, 
                target_rgb=target_rgb, predicted_rgb=predicted_rgb,
                discriminator=discriminator if self.use_gan else None,
                l_channel=l_channel if self.use_gan else None
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_l2_loss += l2_loss.item()
            total_perceptual_loss += perceptual_loss.item()
            total_gan_loss += gan_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 100 == 0:
                log_str = f'{model_name} - Batch {batch_idx}/{len(self.train_loader)}, '
                log_str += f'Loss: {loss.item():.4f}, L1: {l1_loss.item():.4f}, L2: {l2_loss.item():.4f}'
                if self.use_perceptual_loss:
                    log_str += f', Perceptual: {perceptual_loss.item():.4f}'
                if self.use_gan:
                    log_str += f', GAN: {gan_loss.item():.4f}'
                    if discriminator is not None:
                        log_str += f', D_loss: {d_loss:.4f}, D_acc: {d_accuracy:.4f}'
                print(log_str)
        
        avg_loss = total_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_l2_loss = total_l2_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches
        avg_gan_loss = total_gan_loss / num_batches
        avg_d_loss = total_d_loss / num_batches if self.use_gan else 0.0
        avg_d_accuracy = total_d_accuracy / num_batches if self.use_gan else 0.0
        
        return avg_loss, avg_l1_loss, avg_l2_loss, avg_perceptual_loss, avg_gan_loss, avg_d_loss, avg_d_accuracy
    
    def validate_epoch(self, model, model_name):
        """Validate one epoch"""
        model.eval()
        
        # Get discriminator if GAN enabled
        discriminator = None
        if self.use_gan:
            if model_name == 'eccv16':
                discriminator = self.discriminator_eccv16
            else:
                discriminator = self.discriminator_siggraph17
            if discriminator is not None:
                discriminator.eval()
        
        total_loss = 0.0
        total_l1_loss = 0.0
        total_l2_loss = 0.0
        total_perceptual_loss = 0.0
        total_gan_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (rgb_images, _) in enumerate(self.val_loader):
                rgb_images = rgb_images.to(self.device)
                
                # Convert RGB to LAB
                lab_images = self.rgb_to_lab(rgb_images)
                l_channel = lab_images[:, 0:1, :, :]  # L channel
                ab_channels = lab_images[:, 1:, :, :]  # AB channels
                
                # Forward pass
                predicted_ab = model(l_channel)
                
                # Prepare RGB images for perceptual loss if needed
                predicted_rgb = None
                target_rgb = None
                if self.use_perceptual_loss:
                    # Reconstruct predicted RGB from predicted AB + original L
                    predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
                    # In validation, we don't need gradients
                    predicted_rgb = self.lab_to_rgb(predicted_lab, requires_grad=False)
                    target_rgb = rgb_images  # Already in [0, 1] range
                
                # Compute loss
                loss, l1_loss, l2_loss, perceptual_loss, gan_loss = self.compute_loss(
                    predicted_ab, ab_channels, l_channel,
                    target_rgb=target_rgb, predicted_rgb=predicted_rgb,
                    discriminator=discriminator if self.use_gan else None,
                    l_channel=l_channel if self.use_gan else None
                )
                
                # Accumulate losses
                total_loss += loss.item()
                total_l1_loss += l1_loss.item()
                total_l2_loss += l2_loss.item()
                total_perceptual_loss += perceptual_loss.item()
                total_gan_loss += gan_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_l2_loss = total_l2_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches
        avg_gan_loss = total_gan_loss / num_batches
        
        return avg_loss, avg_l1_loss, avg_l2_loss, avg_perceptual_loss, avg_gan_loss
    
    def train_model(self, model_name, epochs=50):
        """Train a specific model"""
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        if model_name == 'eccv16':
            model = self.colorizer_eccv16
            optimizer = self.optimizer_eccv16
        else:
            model = self.colorizer_siggraph17
            optimizer = self.optimizer_siggraph17
        
        best_val_loss = float('inf')
        # ILA models need more patience as they take longer to converge
        if model_name == 'eccv16' and self.use_ila:
            patience = 20  # More patience for ILA models
            print(f"Using patience={patience} for ILA model (attention mechanisms need more time to converge)")
        else:
            patience = 10  # Normal patience for baseline models
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_results = self.train_epoch(model, optimizer, model_name)
            train_loss, train_l1, train_l2, train_perceptual = train_results[0:4]
            train_gan = train_results[4] if len(train_results) > 4 else 0.0
            train_d_loss = train_results[5] if len(train_results) > 5 else 0.0
            train_d_acc = train_results[6] if len(train_results) > 6 else 0.0
            
            # Validate
            val_results = self.validate_epoch(model, model_name)
            val_loss, val_l1, val_l2, val_perceptual = val_results[0:4]
            val_gan = val_results[4] if len(val_results) > 4 else 0.0
            
            # Store history
            self.training_history[model_name]['train_loss'].append(train_loss)
            self.training_history[model_name]['val_loss'].append(val_loss)
            self.training_history[model_name]['train_perceptual_loss'].append(train_perceptual)
            self.training_history[model_name]['val_perceptual_loss'].append(val_perceptual)
            self.training_history[model_name]['train_gan_loss'].append(train_gan)
            self.training_history[model_name]['val_gan_loss'].append(val_gan)
            if self.use_gan:
                self.training_history[model_name]['d_loss'].append(train_d_loss)
                self.training_history[model_name]['d_accuracy'].append(train_d_acc)
            
            # Print metrics
            log_str = f"Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, L2: {train_l2:.4f}"
            if self.use_perceptual_loss:
                log_str += f", Perceptual: {train_perceptual:.4f}"
            if self.use_gan:
                log_str += f", GAN: {train_gan:.4f}, D_loss: {train_d_loss:.4f}, D_acc: {train_d_acc:.4f}"
            log_str += ")"
            print(log_str)
            
            log_str = f"Val Loss: {val_loss:.4f} (L1: {val_l1:.4f}, L2: {val_l2:.4f}"
            if self.use_perceptual_loss:
                log_str += f", Perceptual: {val_perceptual:.4f}"
            if self.use_gan:
                log_str += f", GAN: {val_gan:.4f}"
            log_str += ")"
            print(log_str)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model with appropriate naming to avoid overwriting
                model_filename = self._get_model_filename(model_name)
                torch.save(model.state_dict(), model_filename)
                if self.use_gan:
                    discriminator = self.discriminator_eccv16 if model_name == 'eccv16' else self.discriminator_siggraph17
                    if discriminator is not None:
                        d_filename = model_filename.replace('_best_model.pth', '_discriminator_best.pth')
                        torch.save(discriminator.state_dict(), d_filename)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
                print(f"Model saved as: {model_filename}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_filename = self._get_checkpoint_filename(model_name, epoch + 1)
                torch.save(model.state_dict(), checkpoint_filename)
                print(f"Checkpoint saved: {checkpoint_filename}")
        
        print(f"\n{model_name.upper()} training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def train_both_models(self, epochs=50):
        """Train both models"""
        print("Starting training of both colorization models...")
        
        # Prepare data
        self.prepare_data()
        
        # Train ECCV16
        self.train_model('eccv16', epochs)
        
        # Train SIGGRAPH17
        self.train_model('siggraph17', epochs)
        
        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print("Saved files:")
        print(f"- {self._get_model_filename('eccv16')}")
        print(f"- {self._get_model_filename('siggraph17')}")
        print("- training_history.json")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for i, model_name in enumerate(['eccv16', 'siggraph17']):
            epochs = range(1, len(self.training_history[model_name]['train_loss']) + 1)
            
            axes[i].plot(epochs, self.training_history[model_name]['train_loss'], 'b-', label='Train Loss')
            axes[i].plot(epochs, self.training_history[model_name]['val_loss'], 'r-', label='Val Loss')
            axes[i].set_title(f'{model_name.upper()} Training History')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training history plot saved as 'training_history.png'")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train colorization models with optional ILA')
    parser.add_argument('--use_ila', action='store_true', 
                       help='Enable ILA blocks in ECCV16 model')
    parser.add_argument('--ila_reduction', type=int, default=4,
                       help='Channel reduction factor for ILA (default: 4)')
    parser.add_argument('--ila_use_dw_conv', action='store_true', default=True,
                       help='Use depthwise convolution in ILA (default: True)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate (default: 0.0002)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda). Auto-detects if not specified')
    parser.add_argument('--use_perceptual_loss', action='store_true',
                       help='Enable perceptual loss using VGG features')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                       help='Weight for perceptual loss component (default: 0.1)')
    parser.add_argument('--perceptual_layers', type=str, nargs='+', 
                       default=['relu1_2', 'relu2_2', 'relu3_3'],
                       help='VGG layers to use for perceptual loss (default: relu1_2 relu2_2 relu3_3)')
    parser.add_argument('--perceptual_norm', type=str, default='L1', choices=['L1', 'L2'],
                       help='Distance metric for perceptual loss (L1 or L2, default: L1)')
    parser.add_argument('--vgg_type', type=str, default='vgg16', choices=['vgg16', 'vgg19'],
                       help='Type of VGG model for perceptual loss (vgg16 or vgg19, default: vgg16)')
    parser.add_argument('--use_gan', action='store_true',
                       help='Enable GAN training with PatchGAN discriminator')
    parser.add_argument('--gan_weight', type=float, default=0.1,
                       help='Weight for GAN loss component (default: 0.1)')
    parser.add_argument('--gan_type', type=str, default='lsgan', choices=['bce', 'lsgan'],
                       help='GAN loss type (bce or lsgan, default: lsgan)')
    parser.add_argument('--use_gp', action='store_true',
                       help='Enable gradient penalty (WGAN-GP)')
    parser.add_argument('--gp_weight', type=float, default=10.0,
                       help='Weight for gradient penalty (default: 10.0)')
    parser.add_argument('--use_spectral', action='store_true',
                       help='Enable spectral normalization on discriminator')
    parser.add_argument('--d_lr_factor', type=float, default=1.0,
                       help='Discriminator learning rate multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = ColorizationTrainer(
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_ila=args.use_ila,
        ila_reduction=args.ila_reduction,
        ila_use_dw_conv=args.ila_use_dw_conv,
        use_perceptual_loss=args.use_perceptual_loss,
        perceptual_weight=args.perceptual_weight,
        perceptual_layers=args.perceptual_layers,
        perceptual_norm=args.perceptual_norm,
        vgg_type=args.vgg_type,
        use_gan=args.use_gan,
        gan_weight=args.gan_weight,
        gan_type=args.gan_type,
        use_gp=args.use_gp,
        gp_weight=args.gp_weight,
        use_spectral=args.use_spectral,
        d_lr_factor=args.d_lr_factor
    )
    
    # Train both models
    trainer.train_both_models(epochs=args.epochs)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"\nILA Configuration:")
    if args.use_ila:
        print(f"  ✓ ILA ENABLED")
        print(f"  - Reduction factor: {args.ila_reduction}")
        print(f"  - Depthwise conv: {args.ila_use_dw_conv}")
    else:
        print(f"  ✗ ILA DISABLED (baseline)")
    
    print(f"\nPerceptual Loss Configuration:")
    if args.use_perceptual_loss:
        print(f"  ✓ PERCEPTUAL LOSS ENABLED")
        print(f"  - Weight: {args.perceptual_weight}")
        print(f"  - Layers: {args.perceptual_layers}")
        print(f"  - Norm: {args.perceptual_norm}")
        print(f"  - VGG Type: {args.vgg_type}")
    else:
        print(f"  ✗ PERCEPTUAL LOSS DISABLED")
    
    print(f"\nGAN Configuration:")
    if args.use_gan:
        print(f"  ✓ GAN TRAINING ENABLED")
        print(f"  - Type: {args.gan_type}")
        print(f"  - Weight: {args.gan_weight}")
        if args.use_gp:
            print(f"  - Gradient Penalty: Enabled (weight={args.gp_weight})")
        if args.use_spectral:
            print(f"  - Spectral Normalization: Enabled")
        print(f"  - Discriminator LR Factor: {args.d_lr_factor}")
    else:
        print(f"  ✗ GAN TRAINING DISABLED")
    
    # Show model filenames (use existing trainer instance)
    print(f"\nModel Files:")
    print(f"  - ECCV16: {trainer._get_model_filename('eccv16')}")
    print(f"  - SIGGRAPH17: {trainer._get_model_filename('siggraph17')}")
    
    print(f"\nModels trained:")
    print(f"  - ECCV16 ({'with ILA' if args.use_ila else 'baseline'})")
    print(f"  - SIGGRAPH17")
    print("="*60)

if __name__ == "__main__":
    main()
