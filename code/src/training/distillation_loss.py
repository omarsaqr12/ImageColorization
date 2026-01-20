#!/usr/bin/env python3
"""
Knowledge Distillation Loss Functions
Combines temperature-scaled KL divergence with reconstruction loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    1. Temperature-scaled KL divergence between teacher and student soft logits
    2. Reconstruction loss (MSE/CE) between student output and ground truth
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, reduction='mean'):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax scaling. Higher = softer probabilities.
                        Default: 4.0 (common in KD literature)
            alpha: Weight for distillation loss. Final loss = alpha * distill_loss + (1-alpha) * ce_loss.
                   Default: 0.7 (70% distillation, 30% ground truth)
            reduction: Reduction method for loss ('mean', 'sum', 'none'). Default: 'mean'
        """
        super(DistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        
        # KL divergence loss (for soft logits)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        
        # Reconstruction loss (MSE for AB channels)
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
        print(f"✅ DistillationLoss initialized:")
        print(f"   Temperature: {temperature}")
        print(f"   Alpha (distill weight): {alpha}")
        print(f"   Reduction: {reduction}")
    
    def forward(self, student_logits, teacher_logits, student_ab, target_ab):
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits [B, C, H, W] (e.g., [B, 313, H/4, W/4] for ECCV16)
            teacher_logits: Teacher model logits [B, C, H, W] (may have different shape for SIGGRAPH17)
            student_ab: Student AB channel output [B, 2, H, W]
            target_ab: Ground truth AB channels [B, 2, H, W]
        
        Returns:
            dict with keys:
                - 'total_loss': Combined distillation + reconstruction loss
                - 'distill_loss': KL divergence loss component
                - 'reconstruction_loss': MSE reconstruction loss component
        """
        # Handle shape mismatch between teacher and student logits
        # ECCV16: [B, 313, H/4, W/4], SIGGRAPH17: [B, 128, H, W]
        B_s, C_s, H_s, W_s = student_logits.shape
        B_t, C_t, H_t, W_t = teacher_logits.shape
        
        # If shapes don't match (e.g., SIGGRAPH17 teacher with ECCV16-style student),
        # use feature matching (MSE) instead of KL divergence
        if C_s != C_t or H_s != H_t or W_s != W_t:
            # Resize teacher features to match student spatial dimensions
            if H_t != H_s or W_t != W_s:
                teacher_logits = F.adaptive_avg_pool2d(teacher_logits, (H_s, W_s))
            
            # Handle channel mismatch: use first C_s channels or pad
            if C_t > C_s:
                # Teacher has more channels - take first C_s
                teacher_logits = teacher_logits[:, :C_s, :, :]
            elif C_t < C_s:
                # Teacher has fewer channels - pad with zeros
                padding = torch.zeros(B_t, C_s - C_t, H_s, W_s, 
                                    device=teacher_logits.device, dtype=teacher_logits.dtype)
                teacher_logits = torch.cat([teacher_logits, padding], dim=1)
            
            # Use MSE loss on features (feature matching) instead of KL divergence
            distill_loss = F.mse_loss(student_logits, teacher_logits) * (self.temperature ** 2)
        else:
            # Shapes match - use standard KL divergence (ECCV16 teacher)
            student_logits_flat = student_logits.view(B_s, C_s, -1).transpose(1, 2).contiguous()  # [B, H*W, C]
            teacher_logits_flat = teacher_logits.view(B_t, C_t, -1).transpose(1, 2).contiguous()  # [B, H*W, C]
            
            # Apply temperature scaling and softmax
            student_soft = F.log_softmax(student_logits_flat / self.temperature, dim=2)  # [B, H*W, C]
            teacher_soft = F.softmax(teacher_logits_flat / self.temperature, dim=2)  # [B, H*W, C]
            
            # KL divergence: KL(student || teacher)
            # Note: KLDivLoss expects log-probabilities for first arg, probabilities for second
            kl_div = self.kl_loss(student_soft, teacher_soft)  # [B, H*W, C]
            
            # Sum over class dimension, then reduce
            kl_div = kl_div.sum(dim=2)  # [B, H*W]
            
            if self.reduction == 'mean':
                distill_loss = kl_div.mean() * (self.temperature ** 2)  # Scale by T^2 (standard KD practice)
            elif self.reduction == 'sum':
                distill_loss = kl_div.sum() * (self.temperature ** 2)
            else:  # 'none'
                distill_loss = kl_div * (self.temperature ** 2)
        
        # Reconstruction loss (MSE on AB channels)
        reconstruction_loss = self.mse_loss(student_ab, target_ab)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * reconstruction_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'reconstruction_loss': reconstruction_loss
        }


class FeatureDistillationLoss(nn.Module):
    """
    Extended distillation loss that also matches intermediate features.
    Useful when teacher and student have similar architectures.
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, feature_weight=0.1, reduction='mean'):
        """
        Initialize feature distillation loss.
        
        Args:
            temperature: Temperature for softmax scaling. Default: 4.0
            alpha: Weight for logit distillation vs reconstruction. Default: 0.7
            feature_weight: Weight for feature matching loss. Default: 0.1
            reduction: Reduction method. Default: 'mean'
        """
        super(FeatureDistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.reduction = reduction
        
        # Base distillation loss
        self.distill_loss_fn = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            reduction=reduction
        )
        
        # Feature matching loss (MSE on intermediate features)
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, student_logits, teacher_logits, student_ab, target_ab,
                student_features=None, teacher_features=None):
        """
        Compute feature distillation loss.
        
        Args:
            student_logits: Student logits [B, C, H, W]
            teacher_logits: Teacher logits [B, C, H, W]
            student_ab: Student AB output [B, 2, H, W]
            target_ab: Ground truth AB [B, 2, H, W]
            student_features: Dict of student intermediate features. Optional.
            teacher_features: Dict of teacher intermediate features. Optional.
        
        Returns:
            dict with loss components
        """
        # Base distillation loss
        base_losses = self.distill_loss_fn(
            student_logits, teacher_logits, student_ab, target_ab
        )
        
        # Feature matching loss (if provided)
        feature_loss = torch.tensor(0.0, device=student_ab.device)
        if student_features is not None and teacher_features is not None:
            feature_losses = []
            
            # Match features at corresponding layers
            for key in student_features.keys():
                if key in teacher_features:
                    student_feat = student_features[key]
                    teacher_feat = teacher_features[key]
                    
                    # Handle size mismatches (e.g., if student has fewer channels)
                    if student_feat.shape != teacher_feat.shape:
                        # Option 1: Use adaptive pooling to match spatial dimensions
                        if student_feat.shape[2:] != teacher_feat.shape[2:]:
                            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, student_feat.shape[2:])
                        
                        # Option 2: Use 1x1 conv to match channels (if needed)
                        if student_feat.shape[1] != teacher_feat.shape[1]:
                            # For simplicity, just take first N channels or average pool
                            min_channels = min(student_feat.shape[1], teacher_feat.shape[1])
                            student_feat = student_feat[:, :min_channels]
                            teacher_feat = teacher_feat[:, :min_channels]
                    
                    # Compute MSE loss on features
                    feat_loss = self.mse_loss(student_feat, teacher_feat)
                    feature_losses.append(feat_loss)
            
            if feature_losses:
                feature_loss = sum(feature_losses) / len(feature_losses)
        
        # Total loss
        total_loss = base_losses['total_loss'] + self.feature_weight * feature_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': base_losses['distill_loss'],
            'reconstruction_loss': base_losses['reconstruction_loss'],
            'feature_loss': feature_loss
        }

