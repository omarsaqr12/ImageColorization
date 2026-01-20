#!/usr/bin/env python3
"""
Perceptual Loss Module

This module implements perceptual loss using VGG feature maps. Perceptual loss measures
the distance between images in the feature space of a pretrained network rather than
in pixel space.

Why Perceptual Loss?

Traditional pixel-wise losses (L1, L2) have limitations:
1. They treat each pixel independently, missing spatial relationships and texture patterns
2. They penalize small spatial misalignments heavily, even when perceptually similar
3. They don't capture high-level semantic similarity (e.g., two images of the same object
   in different poses may have high pixel-wise loss but low perceptual loss)

Perceptual loss addresses these issues by:
1. Using feature maps from pretrained networks (VGG) that encode hierarchical visual features
2. Measuring distance in feature space, which better aligns with human perception
3. Capturing texture, structure, and semantic content rather than exact pixel matches
4. Producing more visually realistic results in image generation tasks like colorization

For colorization specifically, perceptual loss helps ensure that:
- Textures are preserved (e.g., wood grain, fabric patterns)
- Structural elements are maintained (e.g., object boundaries, spatial relationships)
- Color transitions are smooth and natural
- The overall visual quality matches human expectations better than pixel-wise losses alone
"""

import torch
import torch.nn as nn
from typing import List, Dict

# Support both direct script execution (where src/training is on sys.path)
# and package-style imports (src.training.*).
try:
    from vgg_features import get_vgg_features  # type: ignore
except ImportError:  # pragma: no cover - fallback for package import
    from .vgg_features import get_vgg_features


class PerceptualLoss(nn.Module):
    """
    Compute perceptual loss between predicted and target images using VGG features.
    
    The loss is computed as the distance (L1 or L2) between feature maps extracted
    from specified layers of a pretrained VGG network.
    """
    
    def __init__(self, 
                 perceptual_layers=['relu1_2', 'relu2_2', 'relu3_3'],
                 perceptual_norm='L1',
                 vgg_type='vgg16',
                 device='cpu'):
        """
        Initialize perceptual loss module.
        
        Args:
            perceptual_layers (List[str]): List of VGG layer names to use for loss computation.
                                          Common choices: ['relu1_2', 'relu2_2', 'relu3_3']
                                          Early layers capture texture, later layers capture structure.
                                          Default: ['relu1_2', 'relu2_2', 'relu3_3']
            perceptual_norm (str): Distance metric to use ('L1' or 'L2'). Default: 'L1'
            vgg_type (str): Type of VGG model ('vgg16' or 'vgg19'). Default: 'vgg16'
            device (str): Device to run computation on ('cpu' or 'cuda'). Default: 'cpu'
        """
        super(PerceptualLoss, self).__init__()
        
        self.perceptual_layers = perceptual_layers
        self.perceptual_norm = perceptual_norm.upper()
        self.vgg_type = vgg_type
        self.device = device
        
        if self.perceptual_norm not in ['L1', 'L2']:
            raise ValueError(f"perceptual_norm must be 'L1' or 'L2', got '{perceptual_norm}'")
        
        # Initialize loss function
        if self.perceptual_norm == 'L1':
            self.loss_fn = nn.L1Loss()
        else:  # L2
            self.loss_fn = nn.MSELoss()
    
    def forward(self, pred_img, target_img):
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred_img (torch.Tensor): Predicted RGB image tensor of shape (B, 3, H, W) with values in [0, 1]
            target_img (torch.Tensor): Target RGB image tensor of shape (B, 3, H, W) with values in [0, 1]
        
        Returns:
            torch.Tensor: Perceptual loss value (scalar)
        """
        # Ensure images are on the correct device
        pred_img = pred_img.to(self.device)
        target_img = target_img.to(self.device)
        
        # Extract features from both images
        # Note: We need gradients for pred_img but not for target_img
        # However, for simplicity and correctness, we allow gradients for both
        # but VGG parameters remain frozen
        pred_features = get_vgg_features(
            pred_img, 
            layers=self.perceptual_layers,
            vgg_type=self.vgg_type,
            device=self.device,
            requires_grad=True  # Need gradients for backprop through pred_img
        )
        
        target_features = get_vgg_features(
            target_img,
            layers=self.perceptual_layers,
            vgg_type=self.vgg_type,
            device=self.device,
            requires_grad=False  # No need for gradients on target
        )
        
        # Compute loss for each layer and average
        total_loss = 0.0
        num_layers = len(self.perceptual_layers)
        
        for layer_name in self.perceptual_layers:
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            
            # Normalize by number of elements in feature map to make loss scale-invariant
            layer_loss = self.loss_fn(pred_feat, target_feat)
            total_loss += layer_loss
        
        # Average across layers
        perceptual_loss = total_loss / num_layers
        
        return perceptual_loss


def compute_perceptual_loss(pred_img, target_img, 
                           perceptual_layers=['relu1_2', 'relu2_2', 'relu3_3'],
                           perceptual_norm='L1',
                           vgg_type='vgg16',
                           device='cpu'):
    """
    Convenience function to compute perceptual loss.
    
    Args:
        pred_img (torch.Tensor): Predicted RGB image tensor of shape (B, 3, H, W) with values in [0, 1]
        target_img (torch.Tensor): Target RGB image tensor of shape (B, 3, H, W) with values in [0, 1]
        perceptual_layers (List[str]): List of VGG layer names to use. Default: ['relu1_2', 'relu2_2', 'relu3_3']
        perceptual_norm (str): Distance metric ('L1' or 'L2'). Default: 'L1'
        vgg_type (str): Type of VGG model ('vgg16' or 'vgg19'). Default: 'vgg16'
        device (str): Device to run computation on. Default: 'cpu'
    
    Returns:
        torch.Tensor: Perceptual loss value (scalar)
    """
    loss_fn = PerceptualLoss(
        perceptual_layers=perceptual_layers,
        perceptual_norm=perceptual_norm,
        vgg_type=vgg_type,
        device=device
    )
    
    return loss_fn(pred_img, target_img)

