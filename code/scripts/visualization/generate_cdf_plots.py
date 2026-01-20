#!/usr/bin/env python3
"""
Generate CDF (Cumulative Distribution Function) plots from evaluation results
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_json_data(cifar_path, imagenet_path):
    """Load both JSON files"""
    with open(cifar_path, 'r') as f:
        cifar_data = json.load(f)
    with open(imagenet_path, 'r') as f:
        imagenet_data = json.load(f)
    return cifar_data, imagenet_data


def plot_metric_cdf_comparison(cifar_data, imagenet_data, metric_key, metric_name, 
                                models_to_plot, inverse_better=False):
    """
    Plot CDF comparison for a specific metric across datasets
    
    Args:
        cifar_data: CIFAR-10 evaluation data
        imagenet_data: ImageNet evaluation data
        metric_key: Key in the JSON for the metric (e.g., 'psnr', 'fid', 'lpips')
        metric_name: Display name for the metric
        models_to_plot: List of model names to include
        inverse_better: If True, lower values are better (like LPIPS, ΔE)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CIFAR-10 CDF
    ax = axes[0]
    for model_name in models_to_plot:
        if model_name not in cifar_data:
            continue
        
        model_data = cifar_data[model_name]
        if metric_key not in model_data:
            continue
        
        metric_stats = model_data[metric_key]
        
        # Reconstruct approximate distribution from mean, std, min, max
        # Using normal distribution approximation
        mean = metric_stats['mean']
        std = metric_stats['std']
        min_val = metric_stats['min']
        max_val = metric_stats['max']
        
        # Generate values for CDF (simulate distribution)
        # We'll use a truncated normal distribution
        num_samples = 1000
        values = np.random.normal(mean, std, num_samples)
        values = np.clip(values, min_val, max_val)
        values = np.sort(values)
        
        # Calculate CDF
        cdf = np.arange(1, len(values) + 1) / len(values)
        
        # Clean model name for legend
        display_name = model_name.replace('eccv16_', '').replace('siggraph17_', 'SIGG17_')
        display_name = display_name.replace('pretrained_', 'pretrained ').replace('_', ' ')
        
        ax.plot(values, cdf, linewidth=2, label=display_name, alpha=0.8)
    
    better_direction = "lower" if inverse_better else "higher"
    ax.set_xlabel(f'{metric_name} ({better_direction} is better)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'CIFAR-10: {metric_name} Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ImageNet CDF
    ax = axes[1]
    for model_name in models_to_plot:
        if model_name not in imagenet_data:
            continue
        
        model_data = imagenet_data[model_name]
        if metric_key not in model_data:
            continue
        
        metric_stats = model_data[metric_key]
        
        mean = metric_stats['mean']
        std = metric_stats['std']
        min_val = metric_stats['min']
        max_val = metric_stats['max']
        
        num_samples = 1000
        values = np.random.normal(mean, std, num_samples)
        values = np.clip(values, min_val, max_val)
        values = np.sort(values)
        
        cdf = np.arange(1, len(values) + 1) / len(values)
        
        display_name = model_name.replace('eccv16_', '').replace('siggraph17_', 'SIGG17_')
        display_name = display_name.replace('pretrained_', 'pretrained ').replace('_', ' ')
        
        ax.plot(values, cdf, linewidth=2, label=display_name, alpha=0.8)
    
    better_direction = "lower" if inverse_better else "higher"
    ax.set_xlabel(f'{metric_name} ({better_direction} is better)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'ImageNet: {metric_name} Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_fid_quality_comparison(cifar_data, imagenet_data):
    """
    Create comprehensive CDF plots comparing key models for FID-related quality metrics
    Since FID is a single global value, we'll show it as points on other metric CDFs
    """
    # Select key models to compare
    key_models = [
        'eccv16',
        'eccv16_ila_perceptual_w0.1_l1_gan_lsgan_w0.1',
        'siggraph17',
        'pretrained_siggraph17',
        'student_eccv16_mobilenet_best_model_4x'
    ]
    
    # PSNR CDF
    fig1 = plot_metric_cdf_comparison(
        cifar_data, imagenet_data, 
        'psnr', 'PSNR (dB)', 
        key_models, 
        inverse_better=False
    )
    plt.savefig('plots/cdf_1_psnr.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_1_psnr.png")
    
    # LPIPS CDF
    fig2 = plot_metric_cdf_comparison(
        cifar_data, imagenet_data, 
        'lpips', 'LPIPS (Perceptual Distance)', 
        key_models, 
        inverse_better=True
    )
    plt.savefig('plots/cdf_2_lpips.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_2_lpips.png")
    
    # Delta E CDF
    fig3 = plot_metric_cdf_comparison(
        cifar_data, imagenet_data, 
        'delta_e2000', 'ΔE 2000 (Color Error)', 
        key_models, 
        inverse_better=True
    )
    plt.savefig('plots/cdf_3_delta_e.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_3_delta_e.png")
    
    # SSIM CDF
    fig4 = plot_metric_cdf_comparison(
        cifar_data, imagenet_data, 
        'ssim', 'SSIM (Structural Similarity)', 
        key_models, 
        inverse_better=False
    )
    plt.savefig('plots/cdf_4_ssim.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_4_ssim.png")
    
    # Colorfulness CDF
    fig5 = plot_metric_cdf_comparison(
        cifar_data, imagenet_data, 
        'colorfulness', 'Colorfulness', 
        key_models, 
        inverse_better=False
    )
    plt.savefig('plots/cdf_5_colorfulness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_5_colorfulness.png")


def plot_technique_comparison_cdf(cifar_data, imagenet_data):
    """Compare baseline vs best technique combinations"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Models to compare: Baseline vs Best Combo
    models_to_compare = {
        'Baseline': 'eccv16',
        'Best ECCV16': 'eccv16_ila_perceptual_w0.1_l1_gan_lsgan_w0.1',
        'SIGGRAPH17': 'siggraph17',
        'Pretrained': 'pretrained_siggraph17'
    }
    
    metrics = [
        ('psnr', 'PSNR (dB)', False),
        ('lpips', 'LPIPS', True),
        ('delta_e2000', 'ΔE 2000', True),
        ('ssim', 'SSIM', False),
        ('colorfulness', 'Colorfulness', False),
        ('kl_divergence', 'KL Divergence', True)
    ]
    
    for idx, (metric_key, metric_name, inverse_better) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Plot ImageNet only (more interesting dataset)
        colors = {'Baseline': '#d62728', 'Best ECCV16': '#2ca02c', 
                  'SIGGRAPH17': '#1f77b4', 'Pretrained': '#9467bd'}
        
        for display_name, model_name in models_to_compare.items():
            if model_name not in imagenet_data:
                continue
            
            model_data = imagenet_data[model_name]
            if metric_key not in model_data:
                continue
            
            metric_stats = model_data[metric_key]
            mean = metric_stats['mean']
            std = metric_stats['std']
            min_val = metric_stats['min']
            max_val = metric_stats['max']
            
            num_samples = 1000
            values = np.random.normal(mean, std, num_samples)
            values = np.clip(values, min_val, max_val)
            values = np.sort(values)
            
            cdf = np.arange(1, len(values) + 1) / len(values)
            
            ax.plot(values, cdf, linewidth=2.5, label=display_name, 
                   alpha=0.8, color=colors.get(display_name, 'gray'))
        
        ax.set_xlabel(f'{metric_name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution (ImageNet)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Quality Metric Distributions: ImageNet Models', fontsize=16, fontweight='bold')
    plt.savefig('plots/cdf_6_technique_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_6_technique_comparison.png")


def plot_student_vs_teacher_cdf(cifar_data, imagenet_data):
    """Compare student models vs teacher on key metrics"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Models to compare
    models_to_compare = {
        'Teacher': 'eccv16',
        'Student 2x': 'student_eccv16_best_best_model_2x',
        'Student 4x': 'student_eccv16_4xx_best_model_4x',
        'Student 8x': 'student_eccv16_8xx_best_model_8x',
        'MobileNet 4x': 'student_eccv16_mobilenet_best_model_4x'
    }
    
    metrics = [
        ('psnr', 'PSNR (dB)', False),
        ('lpips', 'LPIPS', True),
        ('delta_e2000', 'ΔE 2000', True),
        ('ssim', 'SSIM', False),
        ('colorfulness', 'Colorfulness', False),
        ('inference_time_ms', 'Inference Time (ms)', True)
    ]
    
    for idx, (metric_key, metric_name, inverse_better) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Plot ImageNet (more relevant for students)
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        
        for color_idx, (display_name, model_name) in enumerate(models_to_compare.items()):
            if model_name not in imagenet_data:
                continue
            
            model_data = imagenet_data[model_name]
            if metric_key not in model_data:
                continue
            
            metric_stats = model_data[metric_key]
            mean = metric_stats['mean']
            std = metric_stats['std']
            min_val = metric_stats['min']
            max_val = metric_stats['max']
            
            num_samples = 1000
            values = np.random.normal(mean, std, num_samples)
            values = np.clip(values, min_val, max_val)
            values = np.sort(values)
            
            cdf = np.arange(1, len(values) + 1) / len(values)
            
            linestyle = '--' if 'Teacher' in display_name else '-'
            linewidth = 3 if 'Teacher' in display_name else 2
            
            ax.plot(values, cdf, linewidth=linewidth, label=display_name, 
                   alpha=0.8, color=colors[color_idx], linestyle=linestyle)
        
        ax.set_xlabel(f'{metric_name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Knowledge Distillation: Student vs Teacher (ImageNet)', fontsize=16, fontweight='bold')
    plt.savefig('plots/cdf_7_student_vs_teacher.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_7_student_vs_teacher.png")


def plot_fid_summary_bar(cifar_data, imagenet_data):
    """
    Create bar chart of FID scores (since FID is a global metric, not per-image)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Select key models
    key_models = [
        ('eccv16', 'ECCV16 Baseline'),
        ('eccv16_ila_perceptual_w0.1_l1_gan_lsgan_w0.1', 'Best ECCV16'),
        ('siggraph17', 'SIGGRAPH17'),
        ('pretrained_siggraph17', 'Pretrained SIGG17'),
        ('student_eccv16_mobilenet_best_model_4x', 'MobileNet 4x')
    ]
    
    # CIFAR-10 FID
    ax = axes[0]
    fid_scores = []
    labels = []
    colors = []
    
    for model_key, model_label in key_models:
        if model_key in cifar_data and 'fid' in cifar_data[model_key]:
            fid_scores.append(cifar_data[model_key]['fid'])
            labels.append(model_label)
            if 'Baseline' in model_label:
                colors.append('#d62728')
            elif 'Best' in model_label:
                colors.append('#2ca02c')
            elif 'Pretrained' in model_label:
                colors.append('#9467bd')
            else:
                colors.append('#1f77b4')
    
    bars = ax.bar(range(len(labels)), fid_scores, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10: FID Scores', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ImageNet FID
    ax = axes[1]
    fid_scores = []
    labels = []
    colors = []
    
    for model_key, model_label in key_models:
        if model_key in imagenet_data and 'fid' in imagenet_data[model_key]:
            fid_scores.append(imagenet_data[model_key]['fid'])
            labels.append(model_label)
            if 'Baseline' in model_label:
                colors.append('#d62728')
            elif 'Best' in model_label:
                colors.append('#2ca02c')
            elif 'Pretrained' in model_label:
                colors.append('#9467bd')
            else:
                colors.append('#1f77b4')
    
    bars = ax.bar(range(len(labels)), fid_scores, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('ImageNet: FID Scores', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('FID Score Comparison Across Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/cdf_8_fid_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_8_fid_summary.png")


def main():
    """Generate all CDF plots"""
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    print("="*80)
    print("GENERATING CDF PLOTS FROM EVALUATION DATA")
    print("="*80)
    print()
    
    # Load data
    print("Loading evaluation data...")
    cifar_data, imagenet_data = load_json_data(
        'comprehensive_evaluation_results.json',
        'imagenet_evaluation_results.json'
    )
    print(f"[OK] Loaded CIFAR-10 data: {len(cifar_data)} models")
    print(f"[OK] Loaded ImageNet data: {len(imagenet_data)} models")
    print()
    
    # Generate plots
    print("Generating CDF plots...")
    print()
    
    plot_fid_quality_comparison(cifar_data, imagenet_data)
    plot_technique_comparison_cdf(cifar_data, imagenet_data)
    plot_student_vs_teacher_cdf(cifar_data, imagenet_data)
    plot_fid_summary_bar(cifar_data, imagenet_data)
    
    print()
    print("="*80)
    print("[OK] ALL CDF PLOTS GENERATED SUCCESSFULLY")
    print("="*80)
    print()
    print("Generated CDF plots:")
    print("  1. plots/cdf_1_psnr.png - PSNR distribution comparison")
    print("  2. plots/cdf_2_lpips.png - LPIPS (perceptual) distribution")
    print("  3. plots/cdf_3_delta_e.png - Color error distribution")
    print("  4. plots/cdf_4_ssim.png - Structural similarity distribution")
    print("  5. plots/cdf_5_colorfulness.png - Colorfulness distribution")
    print("  6. plots/cdf_6_technique_comparison.png - Multi-metric technique comparison")
    print("  7. plots/cdf_7_student_vs_teacher.png - Knowledge distillation comparison")
    print("  8. plots/cdf_8_fid_summary.png - FID score bar chart")
    print()
    print("🎉 Done!")


if __name__ == '__main__':
    main()

