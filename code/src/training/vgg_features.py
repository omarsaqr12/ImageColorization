#!/usr/bin/env python3
"""
VGG Feature Extraction Module

This module provides functionality to extract feature maps from pretrained VGG models
for use in perceptual loss computation. VGG networks are trained on ImageNet and
capture hierarchical visual features that are useful for measuring perceptual similarity
between images.

Why VGG features for perceptual loss:
- VGG networks learn hierarchical representations: early layers capture low-level features
  (edges, textures), while deeper layers capture high-level semantic features
- Feature-space distances better align with human perception than pixel-wise distances
- Pretrained VGG models provide rich, general-purpose visual features without requiring
  task-specific training
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Union


class VGGFeatureExtractor(nn.Module):
    """
    Extract feature maps from pretrained VGG models.
    
    Supports VGG16 and VGG19 architectures. Features are extracted from
    ReLU activation layers (e.g., relu1_2, relu2_2, relu3_3, relu4_3, relu5_3).
    """
    
    def __init__(self, vgg_type='vgg16', requires_grad=False):
        """
        Initialize VGG feature extractor.
        
        Args:
            vgg_type (str): Type of VGG model ('vgg16' or 'vgg19'). Default: 'vgg16'
            requires_grad (bool): Whether to require gradients for VGG parameters.
                                 Set to False for feature extraction only. Default: False
        """
        super(VGGFeatureExtractor, self).__init__()
        
        if vgg_type.lower() == 'vgg16':
            vgg_model = models.vgg16(pretrained=True)
        elif vgg_type.lower() == 'vgg19':
            vgg_model = models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unsupported VGG type: {vgg_type}. Use 'vgg16' or 'vgg19'")
        
        # Extract features from VGG (before classifier)
        self.features = vgg_model.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = requires_grad
        
        # Set to evaluation mode
        self.eval()
        
        # Map layer names to indices
        # VGG structure: conv layers + ReLU activations
        # relu1_2 = ReLU after 2nd conv in block 1
        # relu2_2 = ReLU after 2nd conv in block 2
        # relu3_3 = ReLU after 3rd conv in block 3
        # relu4_3 = ReLU after 3rd conv in block 4
        # relu5_3 = ReLU after 3rd conv in block 5
        
        if vgg_type.lower() == 'vgg16':
            # VGG16: 0-2 (conv1_1, relu1_1, conv1_2, relu1_2), 5-7 (conv2_1, relu2_1, conv2_2, relu2_2), etc.
            self.layer_indices = {
                'relu1_1': 1,
                'relu1_2': 3,
                'relu2_1': 6,
                'relu2_2': 8,
                'relu3_1': 11,
                'relu3_2': 13,
                'relu3_3': 15,
                'relu4_1': 18,
                'relu4_2': 20,
                'relu4_3': 22,
                'relu5_1': 25,
                'relu5_2': 27,
                'relu5_3': 29,
            }
        else:  # VGG19
            # VGG19: similar structure but with 4 convs in blocks 3, 4, 5
            self.layer_indices = {
                'relu1_1': 1,
                'relu1_2': 3,
                'relu2_1': 6,
                'relu2_2': 8,
                'relu3_1': 11,
                'relu3_2': 13,
                'relu3_3': 15,
                'relu3_4': 17,
                'relu4_1': 20,
                'relu4_2': 22,
                'relu4_3': 24,
                'relu4_4': 26,
                'relu5_1': 29,
                'relu5_2': 31,
                'relu5_3': 33,
                'relu5_4': 35,
            }
    
    def forward(self, x, layers=None):
        """
        Extract feature maps from specified layers.
        
        Args:
            x (torch.Tensor): Input RGB image tensor of shape (B, 3, H, W).
                             Should be normalized with ImageNet statistics:
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            layers (List[str]): List of layer names to extract (e.g., ['relu1_2', 'relu2_2', 'relu3_3']).
                               If None, extracts all available layers. Default: None
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping layer names to feature maps
        """
        if layers is None:
            layers = list(self.layer_indices.keys())
        
        # Validate layer names
        for layer in layers:
            if layer not in self.layer_indices:
                raise ValueError(f"Unknown layer: {layer}. Available layers: {list(self.layer_indices.keys())}")
        
        features = {}
        
        # Forward pass through VGG features
        for i, module in enumerate(self.features):
            x = module(x)
            
            # Check if current layer is one we want to extract
            for layer_name, layer_idx in self.layer_indices.items():
                if i == layer_idx and layer_name in layers:
                    features[layer_name] = x
        
        return features


def get_vgg_features(img_tensor, layers, vgg_type='vgg16', device='cpu', requires_grad=False):
    """
    Extract VGG feature maps from an image tensor.
    
    This is a convenience function that handles normalization and feature extraction.
    
    Args:
        img_tensor (torch.Tensor): RGB image tensor of shape (B, 3, H, W) with values in [0, 1]
        layers (List[str]): List of layer names to extract (e.g., ['relu1_2', 'relu2_2', 'relu3_3'])
        vgg_type (str): Type of VGG model ('vgg16' or 'vgg19'). Default: 'vgg16'
        device (str): Device to run computation on ('cpu' or 'cuda'). Default: 'cpu'
        requires_grad (bool): Whether to allow gradients to flow through. Default: False
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping layer names to feature maps
    """
    # Initialize feature extractor (will be cached if called multiple times)
    cache_key = f"{vgg_type}_{device}_{requires_grad}"
    if not hasattr(get_vgg_features, '_extractors'):
        get_vgg_features._extractors = {}
    
    if cache_key not in get_vgg_features._extractors:
        extractor = VGGFeatureExtractor(vgg_type=vgg_type, requires_grad=requires_grad)
        extractor = extractor.to(device)
        extractor.eval()
        get_vgg_features._extractors[cache_key] = extractor
    
    extractor = get_vgg_features._extractors[cache_key]
    
    # Normalize image to match VGG's expected input
    # VGG expects: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # Input should be in [0, 1] range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    # Normalize
    normalized_img = (img_tensor - mean) / std
    
    # Extract features (with or without gradients based on requires_grad)
    if requires_grad:
        features = extractor(normalized_img, layers=layers)
    else:
        with torch.no_grad():
            features = extractor(normalized_img, layers=layers)
    
    return features


# Initialize module-level extractor cache
get_vgg_features._extractors = {}

