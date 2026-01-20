#!/usr/bin/env python3
"""
Lightweight Student Model for Knowledge Distillation
Compact architecture that learns from teacher model.
"""

import torch
import torch.nn as nn
import sys
import os

# Add colorization module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../colorization'))
from colorizers.base_color import BaseColor


class LightweightStudent(BaseColor):
    """
    Lightweight student model with reduced channels.
    Architecture similar to ECCV16 but with fewer channels for efficiency.
    """
    
    def __init__(self, norm_layer=nn.BatchNorm2d, channel_reduction=2):
        """
        Initialize lightweight student model.
        
        Args:
            norm_layer: Normalization layer. Default: nn.BatchNorm2d
            channel_reduction: Channel reduction factor (2 = half channels, 4 = quarter). Default: 2
        """
        super(LightweightStudent, self).__init__()
        
        self.channel_reduction = channel_reduction
        
        # Reduced channel counts (remove max() to allow proper reduction)
        # Original teacher channels: [64, 128, 256, 512]
        c1 = 64 // channel_reduction
        c2 = 128 // channel_reduction
        c3 = 256 // channel_reduction
        c4 = 512 // channel_reduction
        
        # Ensure minimum channels for model to work (at least 16)
        c1 = max(16, c1)
        c2 = max(16, c2)
        c3 = max(16, c3)
        c4 = max(16, c4)
        
        # Encoder - similar structure but fewer channels
        model1 = [
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(c1),
        ]
        
        model2 = [
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(c2),
        ]
        
        model3 = [
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(c3),
        ]
        
        model4 = [
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(c4),
        ]
        
        # Middle layers with dilation (reduced)
        model5 = [
            nn.Conv2d(c4, c4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c4, c4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(c4),
        ]
        
        model6 = [
            nn.Conv2d(c4, c4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c4, c4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(c4),
        ]
        
        model7 = [
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(c4),
        ]
        
        # Decoder
        model8 = [
            nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        ]
        
        # Output layer - same as teacher (313 classes for ECCV16-style)
        model8 += [nn.Conv2d(c3, 313, kernel_size=1, stride=1, padding=0, bias=True)]
        
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        
        # Softmax and output conversion
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Student model initialized with {total_params/1e6:.2f}M parameters")
        print(f"   Channel reduction: {channel_reduction}x")
        print(f"   Channels: [{c1}, {c2}, {c3}, {c4}]")
    
    def forward(self, input_l, return_logits=False):
        """
        Forward pass through student model.
        
        Args:
            input_l: Grayscale L channel input [B, 1, H, W]
            return_logits: If True, also return logits before softmax. Default: False
        
        Returns:
            If return_logits=False: AB channel output [B, 2, H, W]
            If return_logits=True: dict with 'ab_output' and 'logits'
        """
        # Normalize input
        normalized_l = self.normalize_l(input_l)
        
        # Encoder
        conv1_2 = self.model1(normalized_l)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)  # [B, 313, H/4, W/4]
        
        # Get logits (before softmax)
        logits = conv8_3
        
        # Get probabilities (after softmax)
        probabilities = self.softmax(conv8_3)
        
        # Get final AB output
        out_reg = self.model_out(probabilities)
        ab_output = self.unnormalize_ab(self.upsample4(out_reg))
        
        if return_logits:
            return {
                'logits': logits,
                'probabilities': probabilities,
                'ab_output': ab_output
            }
        else:
            return ab_output


class MobileNetStyleStudent(BaseColor):
    """
    Even more lightweight student using depthwise separable convolutions
    (MobileNet-style) for maximum efficiency.
    """
    
    def __init__(self, norm_layer=nn.BatchNorm2d, width_multiplier=0.5):
        """
        Initialize MobileNet-style student.
        
        Args:
            norm_layer: Normalization layer. Default: nn.BatchNorm2d
            width_multiplier: Channel width multiplier (0.5 = half channels). Default: 0.5
        """
        super(MobileNetStyleStudent, self).__init__()
        
        self.width_multiplier = width_multiplier
        
        def make_divisible(v, divisor=8):
            """Make channels divisible by divisor."""
            return max(divisor, int(v * width_multiplier) // divisor * divisor)
        
        # Channel counts
        c1 = make_divisible(32)
        c2 = make_divisible(64)
        c3 = make_divisible(128)
        c4 = make_divisible(256)
        
        # Depthwise separable convolution helper
        def depthwise_separable_conv(in_channels, out_channels, stride=1):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                          padding=1, groups=in_channels, bias=False),
                norm_layer(in_channels),
                nn.ReLU(True),
                # Pointwise
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(True),
            )
        
        # Encoder with depthwise separable convolutions
        self.model1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(c1),
            nn.ReLU(True),
            depthwise_separable_conv(c1, c1, stride=1),
        )
        
        self.model2 = nn.Sequential(
            depthwise_separable_conv(c1, c2, stride=2),
            depthwise_separable_conv(c2, c2, stride=1),
        )
        
        self.model3 = nn.Sequential(
            depthwise_separable_conv(c2, c3, stride=2),
            depthwise_separable_conv(c3, c3, stride=1),
        )
        
        self.model4 = nn.Sequential(
            depthwise_separable_conv(c3, c4, stride=1),
            depthwise_separable_conv(c4, c4, stride=1),
        )
        
        # Middle layers
        self.model5 = nn.Sequential(
            depthwise_separable_conv(c4, c4, stride=1),
        )
        
        self.model6 = nn.Sequential(
            depthwise_separable_conv(c4, c4, stride=1),
        )
        
        self.model7 = nn.Sequential(
            depthwise_separable_conv(c4, c4, stride=1),
        )
        
        # Decoder
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c3, 313, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        # Output
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ MobileNet-style student initialized with {total_params/1e6:.2f}M parameters")
        print(f"   Width multiplier: {width_multiplier}x")
    
    def forward(self, input_l, return_logits=False):
        """
        Forward pass through MobileNet-style student.
        
        Args:
            input_l: Grayscale L channel input [B, 1, H, W]
            return_logits: If True, also return logits before softmax. Default: False
        
        Returns:
            If return_logits=False: AB channel output [B, 2, H, W]
            If return_logits=True: dict with 'ab_output' and 'logits'
        """
        normalized_l = self.normalize_l(input_l)
        
        conv1_2 = self.model1(normalized_l)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        
        logits = conv8_3
        probabilities = self.softmax(conv8_3)
        out_reg = self.model_out(probabilities)
        ab_output = self.unnormalize_ab(self.upsample4(out_reg))
        
        if return_logits:
            return {
                'logits': logits,
                'probabilities': probabilities,
                'ab_output': ab_output
            }
        else:
            return ab_output

