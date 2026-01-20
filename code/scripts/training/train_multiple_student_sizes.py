#!/usr/bin/env python3
"""
Train Multiple Student Sizes with Knowledge Distillation

This script trains student models of different sizes using the best teacher model
(variant_097_random) and compares their performance across different compression ratios.

Similar to the previous milestone, this trains:
- LightweightStudent with different channel reductions (2x, 4x, 8x)
- MobileNetStyleStudent with different width multipliers
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'training'))

from train_distill import DistillationTrainer


def train_student_size(
    teacher_path,
    student_type,
    student_size_param,
    device,
    save_dir,
    epochs=40,
    batch_size=32,
    learning_rate=1e-4,
    temperature=3.0,
    alpha=0.5,
    use_feature_loss=True,
    feature_weight=0.1,
    patience=5
):
    """
    Train a single student model with specified size.
    
    Args:
        teacher_path: Path to teacher checkpoint
        student_type: 'lightweight' or 'mobilenet'
        student_size_param: For 'lightweight': channel_reduction (2, 4, 8)
                           For 'mobilenet': width_multiplier (0.5, 0.25, 0.125)
        device: Device to use
        save_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Distillation temperature
        alpha: Distillation loss weight
        use_feature_loss: Whether to use feature matching
        feature_weight: Feature loss weight
        patience: Early stopping patience
    
    Returns:
        Dictionary with training results and metrics
    """
    print("\n" + "="*80)
    print(f"Training {student_type} student with size param: {student_size_param}")
    print("="*80)
    
    # Determine save prefix and student_channel_reduction
    if student_type == 'lightweight':
        student_reduction = student_size_param
        save_prefix = f"student_lightweight_{student_reduction}x"
        # For lightweight, use reduction directly
        channel_reduction_for_trainer = student_reduction
    else:  # mobilenet
        width_mult = student_size_param
        student_reduction = int(1.0 / width_mult)  # For naming (e.g., 0.5 -> 2x)
        save_prefix = f"student_mobilenet_{student_reduction}x"
        # For mobilenet, train_distill.py expects student_channel_reduction
        # and calculates width_multiplier = 1.0 / student_channel_reduction
        # So we pass the reduction value (e.g., 2 for 0.5 width_mult)
        channel_reduction_for_trainer = student_reduction
    
    # Create trainer
    # Note: Variant checkpoints may have extra keys, but the backbone should load fine
    trainer = DistillationTrainer(
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        teacher_path=teacher_path,
        teacher_type='eccv16',  # Teacher is ECCV16 variant (uses ECCVGenerator backbone)
        teacher_use_ila=False,  # Best teacher (variant_097_random) doesn't use ILA
        student_type=student_type,
        student_channel_reduction=channel_reduction_for_trainer,
        temperature=temperature,
        alpha=alpha,
        use_feature_loss=use_feature_loss,
        feature_weight=feature_weight
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Train
    trainer.train(
        epochs=epochs,
        save_dir=save_dir,
        save_prefix=save_prefix,
        patience=patience
    )
    
    # Get final metrics
    final_eval = trainer.evaluate(num_samples=1000)
    
    # Count parameters
    teacher_params = sum(p.numel() for p in trainer.teacher.parameters())
    student_params = sum(p.numel() for p in trainer.student.parameters())
    compression_ratio = teacher_params / student_params if student_params > 0 else 0
    
    # Get model size in MB
    student_size_mb = sum(p.numel() * 4 for p in trainer.student.parameters()) / (1024 * 1024)
    teacher_size_mb = sum(p.numel() * 4 for p in trainer.teacher.parameters()) / (1024 * 1024)
    
    results = {
        'student_type': student_type,
        'student_size_param': student_size_param,
        'save_prefix': save_prefix,
        'compression_ratio': compression_ratio,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'teacher_size_mb': teacher_size_mb,
        'student_size_mb': student_size_mb,
        'final_metrics': final_eval,
        'training_history': trainer.history
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train multiple student sizes with knowledge distillation'
    )
    parser.add_argument('--teacher_path', type=str, 
                       default='experiments/variant_097_random/checkpoints/final.pth',
                       help='Path to teacher model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu/cuda). Auto-detect if not specified')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of epochs per student')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Distillation loss weight')
    parser.add_argument('--use_feature_loss', action='store_true', default=True,
                       help='Use feature matching loss')
    parser.add_argument('--no_feature_loss', dest='use_feature_loss', action='store_false',
                       help='Disable feature matching loss')
    parser.add_argument('--feature_weight', type=float, default=0.1,
                       help='Feature loss weight')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='experiments/distilled_students',
                       help='Directory to save all student models')
    parser.add_argument('--lightweight_sizes', type=int, nargs='+', 
                       default=[2, 4, 8],
                       help='Channel reduction factors for LightweightStudent (e.g., 2 4 8)')
    parser.add_argument('--mobilenet_sizes', type=float, nargs='+',
                       default=[0.5, 0.25, 0.125],
                       help='Width multipliers for MobileNetStyleStudent (e.g., 0.5 0.25 0.125)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip students that already have results')
    
    args = parser.parse_args()
    
    # Device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Check teacher path
    if not os.path.exists(args.teacher_path):
        print(f"❌ Error: Teacher checkpoint not found at {args.teacher_path}")
        print("   Please specify correct path with --teacher_path")
        return
    
    print(f"📚 Using teacher: {args.teacher_path}\n")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Results storage
    all_results = []
    results_file = os.path.join(args.save_dir, 'all_students_results.json')
    
    # Load existing results if any
    if os.path.exists(results_file) and args.skip_existing:
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
            existing_prefixes = {r['save_prefix'] for r in existing_results}
    else:
        existing_results = []
        existing_prefixes = set()
    
    print("="*80)
    print("TRAINING MULTIPLE STUDENT SIZES")
    print("="*80)
    print(f"\nLightweightStudent sizes: {args.lightweight_sizes}x")
    print(f"MobileNetStyleStudent sizes: {[f'{int(1/x)}x' for x in args.mobilenet_sizes]}")
    print(f"Total students to train: {len(args.lightweight_sizes) + len(args.mobilenet_sizes)}\n")
    
    # Train LightweightStudent models
    for reduction in args.lightweight_sizes:
        save_prefix = f"student_lightweight_{reduction}x"
        if save_prefix in existing_prefixes and args.skip_existing:
            print(f"\n⏭️  Skipping {save_prefix} (already exists)")
            continue
        
        try:
            results = train_student_size(
                teacher_path=args.teacher_path,
                student_type='lightweight',
                student_size_param=reduction,
                device=device,
                save_dir=args.save_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                temperature=args.temperature,
                alpha=args.alpha,
                use_feature_loss=args.use_feature_loss,
                feature_weight=args.feature_weight,
                patience=args.patience
            )
            all_results.append(results)
            
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\n✅ Completed {save_prefix}")
        except Exception as e:
            print(f"\n❌ Error training {save_prefix}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Train MobileNetStyleStudent models
    for width_mult in args.mobilenet_sizes:
        reduction = int(1.0 / width_mult)
        save_prefix = f"student_mobilenet_{reduction}x"
        if save_prefix in existing_prefixes and args.skip_existing:
            print(f"\n⏭️  Skipping {save_prefix} (already exists)")
            continue
        
        try:
            results = train_student_size(
                teacher_path=args.teacher_path,
                student_type='mobilenet',
                student_size_param=width_mult,
                device=device,
                save_dir=args.save_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                temperature=args.temperature,
                alpha=args.alpha,
                use_feature_loss=args.use_feature_loss,
                feature_weight=args.feature_weight,
                patience=args.patience
            )
            all_results.append(results)
            
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\n✅ Completed {save_prefix}")
        except Exception as e:
            print(f"\n❌ Error training {save_prefix}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    if all_results:
        # Sort by compression ratio (descending)
        all_results.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        print(f"\n📊 Trained {len(all_results)} student models:\n")
        print(f"{'Model':<30} {'Params':<15} {'Size (MB)':<12} {'Compression':<12} {'PSNR':<8} {'SSIM':<8}")
        print("-" * 95)
        
        for r in all_results:
            metrics = r['final_metrics']
            print(f"{r['save_prefix']:<30} "
                  f"{r['student_params']/1e6:>6.2f}M{'':<8} "
                  f"{r['student_size_mb']:>6.2f}{'':<5} "
                  f"{r['compression_ratio']:>6.1f}x{'':<5} "
                  f"{metrics['student_psnr']:>6.2f} "
                  f"{metrics['student_ssim']:>6.4f}")
        
        # Save final results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ All results saved to: {results_file}")
        
        # Create comparison summary
        summary_file = os.path.join(args.save_dir, 'students_comparison_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("KNOWLEDGE DISTILLATION - STUDENT SIZE COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Teacher: {args.teacher_path}\n")
            f.write(f"Teacher Params: {all_results[0]['teacher_params']/1e6:.2f}M\n")
            f.write(f"Teacher Size: {all_results[0]['teacher_size_mb']:.2f} MB\n\n")
            f.write(f"Total Students Trained: {len(all_results)}\n\n")
            f.write("-"*80 + "\n\n")
            
            for r in all_results:
                f.write(f"Student: {r['save_prefix']}\n")
                f.write(f"  Type: {r['student_type']}\n")
                f.write(f"  Parameters: {r['student_params']/1e6:.2f}M\n")
                f.write(f"  Size: {r['student_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {r['compression_ratio']:.2f}x\n")
                metrics = r['final_metrics']
                f.write(f"  PSNR: {metrics['student_psnr']:.4f} dB\n")
                f.write(f"  SSIM: {metrics['student_ssim']:.4f}\n")
                if metrics.get('student_lpips'):
                    f.write(f"  LPIPS: {metrics['student_lpips']:.4f}\n")
                f.write("\n")
        
        print(f"✅ Comparison summary saved to: {summary_file}")
    else:
        print("\n⚠️  No students were trained (all may have been skipped)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
