#!/usr/bin/env python3
"""
Knowledge Distillation Training Script
Trains a compact student model to learn from a full teacher model.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../colorization'))
from colorizers import eccv16, siggraph17

# Import KD components
from teacher import load_teacher
from student import LightweightStudent, MobileNetStyleStudent
from distillation_loss import DistillationLoss, FeatureDistillationLoss

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


class DistillationTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(self, device='cpu', batch_size=32, learning_rate=1e-4,
                 teacher_path=None, teacher_type='eccv16', teacher_use_ila=False,
                 student_type='lightweight', student_channel_reduction=2,
                 temperature=4.0, alpha=0.7, use_feature_loss=False, feature_weight=0.1):
        """
        Initialize distillation trainer.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for training
            learning_rate: Learning rate for student optimizer
            teacher_path: Path to teacher model checkpoint. If None, uses pretrained.
            teacher_type: Type of teacher model ('eccv16' or 'siggraph17'). Default: 'eccv16'
            teacher_use_ila: Whether teacher uses ILA (ECCV16 only). Default: False
            student_type: Type of student ('lightweight' or 'mobilenet'). Default: 'lightweight'
            student_channel_reduction: Channel reduction factor for student. Default: 2
            temperature: Temperature for distillation. Default: 4.0
            alpha: Weight for distillation loss. Default: 0.7
            use_feature_loss: Whether to use feature matching loss. Default: False
            feature_weight: Weight for feature loss. Default: 0.1
        """
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.alpha = alpha
        self.use_feature_loss = use_feature_loss
        
        print("="*80)
        print("KNOWLEDGE DISTILLATION TRAINING")
        print("="*80)
        
        # Load teacher
        print("\n📚 Loading teacher model...")
        self.teacher = load_teacher(
            model_path=teacher_path,
            model_type=teacher_type,
            device=device,
            use_ila=teacher_use_ila
        )
        
        # Initialize student
        print("\n🎓 Initializing student model...")
        if student_type == 'lightweight':
            self.student = LightweightStudent(
                channel_reduction=student_channel_reduction
            ).to(device)
        elif student_type == 'mobilenet':
            self.student = MobileNetStyleStudent(
                width_multiplier=1.0 / student_channel_reduction
            ).to(device)
        else:
            raise ValueError(f"Unknown student_type: {student_type}")
        
        # Count parameters
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        compression_ratio = teacher_params / student_params if student_params > 0 else 0
        
        print(f"\n📊 Model Comparison:")
        print(f"   Teacher: {teacher_params/1e6:.2f}M parameters")
        print(f"   Student: {student_params/1e6:.2f}M parameters")
        print(f"   Compression: {compression_ratio:.2f}x smaller")
        
        # Initialize loss
        if use_feature_loss:
            self.criterion = FeatureDistillationLoss(
                temperature=temperature,
                alpha=alpha,
                feature_weight=feature_weight
            ).to(device)
        else:
            self.criterion = DistillationLoss(
                temperature=temperature,
                alpha=alpha
            ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_distill_loss': [],
            'train_recon_loss': [],
            'val_psnr_teacher': [],
            'val_psnr_student': [],
            'val_ssim_teacher': [],
            'val_ssim_student': [],
            'val_lpips_teacher': [],
            'val_lpips_student': [],
            'student_size_mb': []
        }
        
        # LPIPS model (if available)
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device).eval()
        else:
            self.lpips_model = None
    
    def prepare_data(self):
        """Prepare CIFAR-10 dataset."""
        print("\n📁 Preparing CIFAR-10 dataset...")
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])
        
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
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
    
    def rgb_to_lab(self, rgb_tensor):
        """Convert RGB tensor to LAB tensor."""
        rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
        lab_np = np.zeros_like(rgb_np)
        for i in range(rgb_np.shape[0]):
            rgb_normalized = np.clip(rgb_np[i], 0, 1)
            lab_np[i] = color.rgb2lab(rgb_normalized)
        lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
        return lab_tensor.to(self.device)
    
    def lab_to_rgb(self, lab_tensor):
        """Convert LAB tensor to RGB tensor."""
        lab_np = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()
        rgb_np = np.zeros_like(lab_np)
        for i in range(lab_np.shape[0]):
            rgb_np[i] = color.lab2rgb(lab_np[i])
            rgb_np[i] = np.clip(rgb_np[i], 0, 1)
        rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
        return rgb_tensor.to(self.device)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        total_distill = 0.0
        total_recon = 0.0
        num_batches = 0
        
        for batch_idx, (rgb_images, _) in enumerate(self.train_loader):
            rgb_images = rgb_images.to(self.device)
            
            # Convert to LAB
            lab_images = self.rgb_to_lab(rgb_images)
            l_channels = lab_images[:, 0:1, :, :]
            ab_channels = lab_images[:, 1:3, :, :]
            
            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_output = self.teacher(l_channels, return_features=self.use_feature_loss)
            
            # Student forward
            student_output = self.student(l_channels, return_logits=True)
            
            # Compute loss
            if self.use_feature_loss:
                loss_dict = self.criterion(
                    student_output['logits'],
                    teacher_output['logits'],
                    student_output['ab_output'],
                    ab_channels,
                    student_features=student_output.get('features'),
                    teacher_features=teacher_output.get('features')
                )
            else:
                loss_dict = self.criterion(
                    student_output['logits'],
                    teacher_output['logits'],
                    student_output['ab_output'],
                    ab_channels
                )
            
            loss = loss_dict['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_distill += loss_dict['distill_loss'].item()
            total_recon += loss_dict['reconstruction_loss'].item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f} "
                      f"(Distill={loss_dict['distill_loss'].item():.4f}, "
                      f"Recon={loss_dict['reconstruction_loss'].item():.4f})")
        
        avg_loss = total_loss / num_batches
        avg_distill = total_distill / num_batches
        avg_recon = total_recon / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_distill_loss'].append(avg_distill)
        self.history['train_recon_loss'].append(avg_recon)
        
        return avg_loss
    
    def evaluate(self, num_samples=500):
        """Evaluate teacher and student models."""
        self.student.eval()
        self.teacher.model.eval()
        
        teacher_psnr = []
        teacher_ssim = []
        teacher_lpips = []
        student_psnr = []
        student_ssim = []
        student_lpips = []
        val_losses = []  # Track validation loss
        
        count = 0
        with torch.no_grad():
            for rgb_images, _ in self.val_loader:
                if count >= num_samples:
                    break
                
                rgb_images = rgb_images.to(self.device)
                lab_images = self.rgb_to_lab(rgb_images)
                l_channels = lab_images[:, 0:1, :, :]
                ab_channels = lab_images[:, 1:3, :, :]
                
                # Teacher predictions
                teacher_output = self.teacher(l_channels, return_features=self.use_feature_loss)
                teacher_ab = teacher_output['ab_output']
                teacher_lab = torch.cat([l_channels, teacher_ab], dim=1)
                teacher_rgb = self.lab_to_rgb(teacher_lab)
                
                # Student predictions (with logits for loss computation)
                student_output = self.student(l_channels, return_logits=True)
                student_ab = student_output['ab_output']
                student_lab = torch.cat([l_channels, student_ab], dim=1)
                student_rgb = self.lab_to_rgb(student_lab)
                
                # Compute validation loss
                if self.use_feature_loss:
                    loss_dict = self.criterion(
                        student_output['logits'],
                        teacher_output['logits'],
                        student_output['ab_output'],
                        ab_channels,
                        student_features=student_output.get('features'),
                        teacher_features=teacher_output.get('features')
                    )
                else:
                    loss_dict = self.criterion(
                        student_output['logits'],
                        teacher_output['logits'],
                        student_output['ab_output'],
                        ab_channels
                    )
                val_losses.append(loss_dict['total_loss'].item())
                
                # Convert to numpy for metrics
                for i in range(rgb_images.size(0)):
                    if count >= num_samples:
                        break
                    
                    orig = rgb_images[i].permute(1, 2, 0).cpu().numpy()
                    teach = teacher_rgb[i].permute(1, 2, 0).cpu().numpy()
                    stud = student_rgb[i].permute(1, 2, 0).cpu().numpy()
                    
                    # PSNR
                    t_psnr = psnr(orig, teach, data_range=1.0)
                    s_psnr = psnr(orig, stud, data_range=1.0)
                    teacher_psnr.append(t_psnr)
                    student_psnr.append(s_psnr)
                    
                    # SSIM
                    t_ssim = ssim(orig, teach, data_range=1.0, channel_axis=2)
                    s_ssim = ssim(orig, stud, data_range=1.0, channel_axis=2)
                    teacher_ssim.append(t_ssim)
                    student_ssim.append(s_ssim)
                    
                    # LPIPS
                    if self.lpips_model is not None:
                        orig_t = rgb_images[i:i+1]
                        teach_t = teacher_rgb[i:i+1]
                        stud_t = student_rgb[i:i+1]
                        t_lpips = self.lpips_model(orig_t, teach_t).item()
                        s_lpips = self.lpips_model(orig_t, stud_t).item()
                        teacher_lpips.append(t_lpips)
                        student_lpips.append(s_lpips)
                    
                    count += 1
        
        # Calculate student model size
        student_size_mb = sum(p.numel() * 4 for p in self.student.parameters()) / (1024 * 1024)
        
        # Convert numpy types to native Python types for JSON serialization
        results = {
            'teacher_psnr': float(np.mean(teacher_psnr)) if teacher_psnr else 0.0,
            'student_psnr': float(np.mean(student_psnr)) if student_psnr else 0.0,
            'teacher_ssim': float(np.mean(teacher_ssim)) if teacher_ssim else 0.0,
            'student_ssim': float(np.mean(student_ssim)) if student_ssim else 0.0,
            'teacher_lpips': float(np.mean(teacher_lpips)) if teacher_lpips else None,
            'student_lpips': float(np.mean(student_lpips)) if student_lpips else None,
            'student_size_mb': float(student_size_mb),
            'val_loss': float(np.mean(val_losses)) if val_losses else float('inf')
        }
        
        return results
    
    def plot_training_history(self, save_dir='models', save_prefix='student_distill'):
        """Plot and save training history visualization."""
        if len(self.history['train_loss']) == 0:
            print("⚠️  No training history to plot")
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Knowledge Distillation Training History: {save_prefix}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Training Losses
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Total Loss', linewidth=2)
        ax1.plot(epochs, self.history['train_distill_loss'], 'g--', label='Distillation Loss', linewidth=1.5)
        ax1.plot(epochs, self.history['train_recon_loss'], 'r--', label='Reconstruction Loss', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation PSNR (Teacher vs Student)
        ax2 = axes[0, 1]
        if len(self.history['val_psnr_teacher']) > 0:
            ax2.plot(epochs, self.history['val_psnr_teacher'], 'b-', label='Teacher PSNR', linewidth=2)
            ax2.plot(epochs, self.history['val_psnr_student'], 'r-', label='Student PSNR', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('Validation PSNR: Teacher vs Student')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation SSIM (Teacher vs Student)
        ax3 = axes[1, 0]
        if len(self.history['val_ssim_teacher']) > 0:
            ax3.plot(epochs, self.history['val_ssim_teacher'], 'b-', label='Teacher SSIM', linewidth=2)
            ax3.plot(epochs, self.history['val_ssim_student'], 'r-', label='Student SSIM', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('SSIM')
            ax3.set_title('Validation SSIM: Teacher vs Student')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation LPIPS (Teacher vs Student) or Student Size
        ax4 = axes[1, 1]
        if len(self.history.get('val_lpips_teacher', [])) > 0:
            ax4.plot(epochs, self.history['val_lpips_teacher'], 'b-', label='Teacher LPIPS', linewidth=2)
            ax4.plot(epochs, self.history['val_lpips_student'], 'r-', label='Student LPIPS', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('LPIPS (lower is better)')
            ax4.set_title('Validation LPIPS: Teacher vs Student')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        elif len(self.history.get('student_size_mb', [])) > 0:
            ax4.plot(epochs, self.history['student_size_mb'], 'g-', label='Student Model Size', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Size (MB)')
            ax4.set_title('Student Model Size')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{save_prefix}_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training history plot saved to: {plot_path}")
    
    def train(self, epochs=50, save_dir='models', save_prefix='student_distill', patience=5):
        """
        Main training loop.
        
        Args:
            epochs: Maximum number of epochs
            save_dir: Directory to save models
            save_prefix: Prefix for saved files
            patience: Early stopping patience (stop if no improvement for N epochs). Default: 5
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')  # Use validation loss like other training scripts
        patience_counter = 0
        
        print(f"Early stopping patience: {patience} epochs")
        print("="*80)
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*80}")
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            train_time = time.time() - start_time
            
            # Evaluate
            print(f"\n📊 Evaluating...")
            eval_results = self.evaluate(num_samples=500)
            
            # Update history (convert numpy types to native Python types for JSON serialization)
            self.history['val_psnr_teacher'].append(float(eval_results['teacher_psnr']))
            self.history['val_psnr_student'].append(float(eval_results['student_psnr']))
            self.history['val_ssim_teacher'].append(float(eval_results['teacher_ssim']))
            self.history['val_ssim_student'].append(float(eval_results['student_ssim']))
            if eval_results['teacher_lpips'] is not None:
                self.history['val_lpips_teacher'].append(float(eval_results['teacher_lpips']))
                self.history['val_lpips_student'].append(float(eval_results['student_lpips']))
            self.history['student_size_mb'].append(float(eval_results['student_size_mb']))
            
            # Print metrics
            print(f"\n📈 Metrics:")
            print(f"   Teacher - PSNR: {eval_results['teacher_psnr']:.4f}, "
                  f"SSIM: {eval_results['teacher_ssim']:.4f}")
            if eval_results['teacher_lpips'] is not None:
                print(f"            LPIPS: {eval_results['teacher_lpips']:.4f}")
            print(f"   Student - PSNR: {eval_results['student_psnr']:.4f}, "
                  f"SSIM: {eval_results['student_ssim']:.4f}")
            if eval_results['student_lpips'] is not None:
                print(f"            LPIPS: {eval_results['student_lpips']:.4f}")
            print(f"   Student Size: {eval_results['student_size_mb']:.2f} MB")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Time: {train_time:.2f}s")
            
            # Save best model (only save best, not all checkpoints)
            # Use validation loss for consistency with other training scripts
            val_loss = eval_results['val_loss']
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience when we find a better model
                best_path = os.path.join(save_dir, f'{save_prefix}_best_model.pth')
                torch.save(self.student.state_dict(), best_path)
                print(f"   ✅ Saved best model (Val Loss: {best_val_loss:.4f}, PSNR: {eval_results['student_psnr']:.4f})")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement (Patience: {patience_counter}/{patience})")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                    print(f"   Best validation loss: {best_val_loss:.4f}")
                    print(f"   Best model was saved at epoch {epoch - patience}")
                    break
            
            # Update scheduler
            self.scheduler.step(train_loss)
        
        # Save training history
        history_path = os.path.join(save_dir, f'{save_prefix}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✅ Training history saved to: {history_path}")
        
        # Plot and save training history
        self.plot_training_history(save_dir, save_prefix)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train student model via knowledge distillation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--teacher_path', type=str, default=None, 
                       help='Path to teacher model checkpoint (uses pretrained if None)')
    parser.add_argument('--teacher_type', type=str, default='eccv16', 
                       choices=['eccv16', 'siggraph17'], help='Teacher model type')
    parser.add_argument('--teacher_use_ila', action='store_true', 
                       help='Use ILA in teacher (ECCV16 only)')
    parser.add_argument('--student_type', type=str, default='lightweight',
                       choices=['lightweight', 'mobilenet'], help='Student architecture')
    parser.add_argument('--student_reduction', type=int, default=2,
                       help='Channel reduction factor for student')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Weight for distillation loss (vs reconstruction)')
    parser.add_argument('--use_feature_loss', action='store_true',
                       help='Use feature matching loss in addition to logit distillation')
    parser.add_argument('--feature_weight', type=float, default=0.1,
                       help='Weight for feature matching loss')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--save_prefix', type=str, default='student_distill',
                       help='Prefix for saved model files')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (stop if no improvement for N epochs). Default: 5')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Create trainer
    trainer = DistillationTrainer(
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        teacher_path=args.teacher_path,
        teacher_type=args.teacher_type,
        teacher_use_ila=args.teacher_use_ila,
        student_type=args.student_type,
        student_channel_reduction=args.student_reduction,
        temperature=args.temperature,
        alpha=args.alpha,
        use_feature_loss=args.use_feature_loss,
        feature_weight=args.feature_weight
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Train
    trainer.train(
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix,
        patience=args.patience
    )


if __name__ == '__main__':
    main()

