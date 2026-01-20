#!/usr/bin/env python3
"""
GAN Loss Functions for Colorization

This module provides GAN loss functions for training colorization models with
adversarial training. It supports both Binary Cross-Entropy (BCE) and Least Squares
GAN (LSGAN) loss types.

Why GAN Loss for Colorization?

1. **Realistic Color Distributions**: GAN loss encourages the generator to produce
   color distributions that match real images. Without GAN loss, colorization models
   often produce desaturated, "safe" colors that minimize pixel-wise error but look
   unrealistic. GAN loss pushes the model toward more vibrant, natural colors.

2. **Coherent Color Transitions**: The discriminator evaluates local patches, encouraging
   smooth and coherent color transitions between regions. This helps avoid artifacts like
   color bleeding or abrupt color changes.

3. **Saturated Colors**: Traditional L1/L2 losses tend to produce desaturated results
   because they penalize any deviation from the mean. GAN loss, by learning from real
   color distributions, encourages the model to produce the full range of saturated colors
   found in natural images.

Loss Types:

- **BCE (Binary Cross-Entropy)**: Standard GAN loss using sigmoid + binary cross-entropy.
  Classic approach, but can suffer from vanishing gradients when discriminator becomes
  too confident.

- **LSGAN (Least Squares GAN)**: Uses least squares loss instead of BCE. More stable
  training, provides stronger gradients, and often produces better results. The loss
  encourages the discriminator to push fake samples toward the real data distribution
  boundary rather than just classifying them.

Usage:

The generator loss combines:
  G_loss = base_loss + perceptual_weight * perceptual_loss + gan_weight * gan_loss

The discriminator loss is:
  D_loss = discriminator_loss(D_real, D_fake, gan_type)

Where:
  - D_real: Discriminator output on real images (should be high)
  - D_fake: Discriminator output on fake images (should be low)
"""

import torch
import torch.nn as nn


def generator_gan_loss(D_fake, gan_type='lsgan'):
    """
    Compute generator GAN loss.
    
    The generator wants the discriminator to classify fake images as real.
    For BCE: minimize -log(D_fake) (equivalent to minimizing BCE with label=1)
    For LSGAN: minimize (D_fake - 1)^2
    
    Args:
        D_fake (torch.Tensor): Discriminator output on fake (generated) images.
                              Shape: (B, 1, H, W) or (B, 1) - patch map or scalar
        gan_type (str): Type of GAN loss ('bce' or 'lsgan'). Default: 'lsgan'
    
    Returns:
        torch.Tensor: Generator GAN loss (scalar)
    """
    if gan_type.lower() == 'bce':
        # Binary cross-entropy loss: generator wants D_fake to be 1 (real)
        # Use BCEWithLogitsLoss for numerical stability
        # Target is 1 (real label)
        target = torch.ones_like(D_fake)
        loss = nn.functional.binary_cross_entropy_with_logits(D_fake, target)
        
    elif gan_type.lower() == 'lsgan':
        # Least squares GAN: generator wants D_fake to be close to 1
        # Loss = 0.5 * (D_fake - 1)^2
        loss = 0.5 * torch.mean((D_fake - 1.0) ** 2)
        
    else:
        raise ValueError(f"Unknown gan_type: {gan_type}. Must be 'bce' or 'lsgan'")
    
    return loss


def discriminator_loss(D_real, D_fake, gan_type='lsgan'):
    """
    Compute discriminator loss.
    
    The discriminator wants to:
    - Classify real images as real (D_real should be high)
    - Classify fake images as fake (D_fake should be low)
    
    Args:
        D_real (torch.Tensor): Discriminator output on real images.
                              Shape: (B, 1, H, W) or (B, 1) - patch map or scalar
        D_fake (torch.Tensor): Discriminator output on fake (generated) images.
                              Shape: (B, 1, H, W) or (B, 1) - patch map or scalar
        gan_type (str): Type of GAN loss ('bce' or 'lsgan'). Default: 'lsgan'
    
    Returns:
        torch.Tensor: Discriminator loss (scalar)
    """
    if gan_type.lower() == 'bce':
        # Binary cross-entropy loss
        # Real images should be classified as 1, fake as 0
        real_target = torch.ones_like(D_real)
        fake_target = torch.zeros_like(D_fake)
        
        loss_real = nn.functional.binary_cross_entropy_with_logits(D_real, real_target)
        loss_fake = nn.functional.binary_cross_entropy_with_logits(D_fake, fake_target)
        
        # Total loss is average of real and fake losses
        loss = (loss_real + loss_fake) * 0.5
        
    elif gan_type.lower() == 'lsgan':
        # Least squares GAN loss
        # Real images: D_real should be close to 1
        # Fake images: D_fake should be close to 0
        loss_real = 0.5 * torch.mean((D_real - 1.0) ** 2)
        loss_fake = 0.5 * torch.mean(D_fake ** 2)
        
        # Total loss is sum of real and fake losses
        loss = loss_real + loss_fake
        
    else:
        raise ValueError(f"Unknown gan_type: {gan_type}. Must be 'bce' or 'lsgan'")
    
    return loss


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device='cpu'):
    """
    Compute gradient penalty for Wasserstein GAN with Gradient Penalty (WGAN-GP).
    
    This is an optional regularization technique that can be used instead of or
    in addition to spectral normalization. It enforces the Lipschitz constraint
    by penalizing the gradient norm of the discriminator on interpolated samples.
    
    Args:
        discriminator (nn.Module): Discriminator model
        real_samples (torch.Tensor): Real samples, shape (B, C, H, W)
        fake_samples (torch.Tensor): Fake samples, shape (B, C, H, W)
        device (str): Device to compute on. Default: 'cpu'
    
    Returns:
        torch.Tensor: Gradient penalty value (scalar)
    """
    # Random interpolation coefficient
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    
    # Interpolate between real and fake samples
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Discriminator output on interpolated samples
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty: (||gradient|| - 1)^2
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def apply_spectral_norm(module):
    """
    Apply spectral normalization to a module.
    
    Spectral normalization stabilizes GAN training by constraining the Lipschitz
    constant of the discriminator. It can be applied to convolutional and linear layers.
    
    Args:
        module (nn.Module): Module to apply spectral normalization to
    
    Returns:
        nn.Module: Module with spectral normalization applied
    """
    from torch.nn.utils import spectral_norm
    
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return spectral_norm(module)
    return module

