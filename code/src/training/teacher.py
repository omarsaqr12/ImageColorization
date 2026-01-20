#!/usr/bin/env python3
"""
Teacher Model Wrapper for Knowledge Distillation
Loads the full improved model and provides forward pass with logits/probabilities
and intermediate features for student training.
"""

import os
import sys
import torch
import torch.nn as nn

# Add colorization module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../colorization'))
from colorizers import eccv16, siggraph17


class TeacherWrapper(nn.Module):
    """
    Wrapper for teacher model that provides:
    - Forward pass returning logits/probabilities and intermediate features
    - Frozen weights during student training
    """
    
    def __init__(self, model_path=None, model_type='eccv16', device='cpu', 
                 use_ila=False, ila_reduction=4, ila_use_dw_conv=True):
        """
        Initialize teacher wrapper.
        
        Args:
            model_path: Path to saved teacher model checkpoint. If None, uses pretrained.
            model_type: Type of model ('eccv16' or 'siggraph17'). Default: 'eccv16'
            device: Device to load model on. Default: 'cpu'
            use_ila: Whether teacher uses ILA (only for ECCV16). Default: False
            ila_reduction: ILA channel reduction factor. Default: 4
            ila_use_dw_conv: Whether ILA uses depthwise conv. Default: True
        """
        super(TeacherWrapper, self).__init__()
        
        self.model_type = model_type
        self.device = device
        
        # Load model architecture
        if model_type == 'eccv16':
            self.model = eccv16(
                pretrained=(model_path is None),
                use_ila=use_ila,
                ila_reduction=ila_reduction,
                ila_use_dw_conv=ila_use_dw_conv
            )
        elif model_type == 'siggraph17':
            self.model = siggraph17(pretrained=(model_path is None))
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'eccv16' or 'siggraph17'")
        
        # Load weights if path provided
        if model_path is not None:
            if os.path.exists(model_path):
                print(f"Loading teacher weights from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                # Handle variant checkpoints that may have extra keys (e.g., backbone.*, multiscale.*, etc.)
                # Try loading with strict=False to ignore extra keys
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    print("✅ Teacher weights loaded (some keys may have been ignored)")
                except Exception as e:
                    print(f"Warning: Could not load all weights: {e}")
                    # Try loading only matching keys
                    model_dict = self.model.state_dict()
                    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                    if filtered_dict:
                        model_dict.update(filtered_dict)
                        self.model.load_state_dict(model_dict)
                        print(f"✅ Loaded {len(filtered_dict)}/{len(model_dict)} matching keys")
            else:
                print(f"Warning: Model path {model_path} not found. Using pretrained weights.")
        
        # Move to device and set to eval mode (frozen)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"✅ Teacher model ({model_type}) loaded and frozen")
        if model_path:
            print(f"   Model path: {model_path}")
        else:
            print(f"   Using pretrained weights")
    
    def forward(self, input_l, return_features=True):
        """
        Forward pass through teacher model.
        
        Args:
            input_l: Grayscale L channel input [B, 1, H, W]
            return_features: If True, return intermediate features. Default: True
        
        Returns:
            dict with keys:
                - 'logits': Raw logits before softmax (ECCV16: [B, 313, H/4, W/4], SIGGRAPH17: None)
                - 'probabilities': Soft probabilities (ECCV16: [B, 313, H/4, W/4], SIGGRAPH17: None)
                - 'ab_output': Final AB channel output [B, 2, H, W]
                - 'features': Dict of intermediate features (if return_features=True)
        """
        with torch.no_grad():  # No gradients needed for teacher
            if self.model_type == 'eccv16':
                return self._forward_eccv16(input_l, return_features)
            else:  # siggraph17
                return self._forward_siggraph17(input_l, return_features)
    
    def _forward_eccv16(self, input_l, return_features):
        """Forward pass for ECCV16 model."""
        features = {}
        
        # Normalize input
        normalized_l = self.model.normalize_l(input_l)
        
        # Encoder stages
        conv1_2 = self.model.model1(normalized_l)
        conv2_2 = self.model.model2(conv1_2)
        
        if self.model.use_ila:
            conv2_2 = self.model.ila1(conv2_2)
            if return_features:
                features['conv2_2'] = conv2_2
        
        conv3_3 = self.model.model3(conv2_2)
        if self.model.use_ila:
            conv3_3 = self.model.ila2(conv3_3)
            if return_features:
                features['conv3_3'] = conv3_3
        
        conv4_3 = self.model.model4(conv3_3)
        if self.model.use_ila:
            conv4_3 = self.model.ila3(conv4_3)
            if return_features:
                features['conv4_3'] = conv4_3
        
        conv5_3 = self.model.model5(conv4_3)
        conv6_3 = self.model.model6(conv5_3)
        conv7_3 = self.model.model7(conv6_3)
        conv8_3 = self.model.model8(conv7_3)  # [B, 313, H/4, W/4] - these are the logits
        
        # Get logits (before softmax)
        logits = conv8_3
        
        # Get probabilities (after softmax)
        probabilities = self.model.softmax(conv8_3)
        
        # Get final AB output
        out_reg = self.model.model_out(probabilities)
        ab_output = self.model.unnormalize_ab(self.model.upsample4(out_reg))
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'ab_output': ab_output
        }
        
        if return_features:
            features['conv1_2'] = conv1_2
            features['conv5_3'] = conv5_3
            features['conv6_3'] = conv6_3
            features['conv7_3'] = conv7_3
            result['features'] = features
        
        return result
    
    def _forward_siggraph17(self, input_l, return_features):
        """Forward pass for SIGGRAPH17 model."""
        features = {}
        
        # SIGGRAPH17 doesn't have explicit logits like ECCV16
        # We'll use the final feature maps as "logits" for distillation
        normalized_l = self.model.normalize_l(input_l)
        
        # Create dummy AB input (zeros) and mask
        input_B = torch.cat((normalized_l * 0, normalized_l * 0), dim=1)
        mask_B = normalized_l * 0
        
        # Encoder
        conv1_2 = self.model.model1(torch.cat((normalized_l, self.model.normalize_ab(input_B), mask_B), dim=1))
        conv2_2 = self.model.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model.model5(conv4_3)
        conv6_3 = self.model.model6(conv5_3)
        conv7_3 = self.model.model7(conv6_3)
        
        # Decoder
        conv8_up = self.model.model8up(conv7_3) + self.model.model3short8(conv3_3)
        conv8_3 = self.model.model8(conv8_up)
        conv9_up = self.model.model9up(conv8_3) + self.model.model2short9(conv2_2)
        conv9_3 = self.model.model9(conv9_up)
        conv10_up = self.model.model10up(conv9_3) + self.model.model1short10(conv1_2)
        conv10_2 = self.model.model10(conv10_up)
        
        # Final output
        ab_output = self.model.unnormalize_ab(self.model.model_out(conv10_2))
        
        # For SIGGRAPH17, we use conv10_2 as "logits" (it's the final feature before output)
        # We'll treat it as a 2-channel "logit" representation
        logits = conv10_2  # [B, 128, H, W]
        probabilities = torch.softmax(conv10_2.view(conv10_2.size(0), conv10_2.size(1), -1), dim=1)
        probabilities = probabilities.view_as(conv10_2)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'ab_output': ab_output
        }
        
        if return_features:
            features['conv1_2'] = conv1_2
            features['conv2_2'] = conv2_2
            features['conv3_3'] = conv3_3
            features['conv4_3'] = conv4_3
            features['conv5_3'] = conv5_3
            features['conv6_3'] = conv6_3
            features['conv7_3'] = conv7_3
            features['conv8_3'] = conv8_3
            features['conv9_3'] = conv9_3
            features['conv10_2'] = conv10_2
            result['features'] = features
        
        return result


def load_teacher(model_path=None, model_type='eccv16', device='cpu', 
                 use_ila=False, ila_reduction=4, ila_use_dw_conv=True):
    """
    Convenience function to load teacher model.
    
    Args:
        model_path: Path to teacher checkpoint. If None, uses pretrained.
        model_type: 'eccv16' or 'siggraph17'. Default: 'eccv16'
        device: Device to load on. Default: 'cpu'
        use_ila: Whether teacher uses ILA (ECCV16 only). Default: False
        ila_reduction: ILA reduction factor. Default: 4
        ila_use_dw_conv: Whether ILA uses depthwise conv. Default: True
    
    Returns:
        TeacherWrapper instance
    """
    return TeacherWrapper(
        model_path=model_path,
        model_type=model_type,
        device=device,
        use_ila=use_ila,
        ila_reduction=ila_reduction,
        ila_use_dw_conv=ila_use_dw_conv
    )

