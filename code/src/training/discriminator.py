#!/usr/bin/env python3
"""
PatchGAN Discriminator for Colorization

This module implements a lightweight PatchGAN discriminator with a 70x70 receptive field.
PatchGAN is an efficient discriminator architecture that classifies local image patches
as real or fake, rather than the entire image.

Why PatchGAN for Colorization?

1. **Local Texture and Color Coherence**: Colorization requires realistic local color
   distributions and texture patterns. PatchGAN's patch-based approach is ideal for
   evaluating whether small image regions have realistic color statistics and transitions.

2. **Efficiency**: By classifying patches instead of the full image, PatchGAN is much
   more parameter-efficient and faster than full-image discriminators. This is crucial
   for colorization where we need to process many images during training.

3. **70x70 Receptive Field**: A 70x70 receptive field is optimal for colorization because:
   - It's large enough to capture local color coherence (e.g., a sky region, a fabric pattern)
   - It's small enough to focus on texture and local color distributions rather than global
     scene semantics (which the generator should handle)
   - It encourages the generator to produce realistic colors at multiple scales

4. **Shape-Aware Input**: The discriminator receives both the L-channel (luminance) and
   the predicted colorized image (either as L+ab channels or as RGB). This allows it to:
   - Verify that colors are consistent with the luminance structure
   - Detect artifacts where colors don't align with edges and textures in the L channel
   - Ensure spatial coherence between luminance and chrominance

Architecture Details:

The discriminator uses a series of convolutional layers with:
- LeakyReLU activations (negative slope 0.2) for stable training
- Instance normalization for better training dynamics
- Strided convolutions to reduce spatial dimensions
- No fully connected layers (fully convolutional)

The output is a patch map where each spatial location corresponds to a 70x70 receptive
field in the input image. Each patch is classified as real (high value) or fake (low value).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with 70x70 receptive field.
    
    Input: Concatenation of L-channel and predicted colorized image
           Shape: (B, 3, H, W) for L+ab (1 L channel + 2 ab channels)
           Shape: (B, 4, H, W) for L+RGB (1 L channel + 3 RGB channels)
    
    Output: Patch map of real/fake logits
            Shape: (B, 1, H', W') where H' and W' depend on input size
    """
    
    def __init__(self, input_channels=3, ndf=64, n_layers=3):
        """
        Initialize PatchGAN discriminator.
        
        Args:
            input_channels (int): Number of input channels. Default: 3 (1 L + 2 ab channels)
            ndf (int): Number of discriminator filters in first conv layer. Default: 64
            n_layers (int): Number of layers in discriminator. Default: 3 (for 70x70 receptive field)
        """
        super(PatchGANDiscriminator, self).__init__()
        
        # Build discriminator layers
        # We use a sequence of conv layers with increasing filters
        # Each layer reduces spatial size by 2 (stride=2)
        
        layers = []
        
        # First layer: no normalization
        layers.append(nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers: with instance normalization
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x to prevent too many filters
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                   kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer: output single channel per patch
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                               kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer: 1 channel (real/fake logit per patch)
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, input_tensor):
        """
        Forward pass through discriminator.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
                                       where C=4 (L channel + ab channels or L + RGB)
        
        Returns:
            torch.Tensor: Patch map of real/fake logits, shape (B, 1, H', W')
        """
        return self.model(input_tensor)


def create_discriminator(input_channels=3, ndf=64, n_layers=3, device='cpu'):
    """
    Convenience function to create a PatchGAN discriminator.
    
    Args:
        input_channels (int): Number of input channels. Default: 3 (1 L + 2 ab channels)
        ndf (int): Number of discriminator filters. Default: 64
        n_layers (int): Number of layers. Default: 3
        device (str): Device to place model on. Default: 'cpu'
    
    Returns:
        PatchGANDiscriminator: Initialized discriminator model
    """
    discriminator = PatchGANDiscriminator(
        input_channels=input_channels,
        ndf=ndf,
        n_layers=n_layers
    )
    return discriminator.to(device)

