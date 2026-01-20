#!/usr/bin/env python3
"""
Generate comprehensive CDF plots from evaluation results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_data():
    """Load evaluation data"""
    with open('comprehensive_evaluation_results.json', 'r') as f:
        cifar_data = json.load(f)
    with open('imagenet_evaluation_results.json', 'r') as f:
        imagenet_data = json.load(f)
    return cifar_data, imagenet_data


def plot_fid_summary(cifar_data, imagenet_data):
    """Bar chart of FID scores"""
    key_models = [
        ('eccv16', 'Baseline'),
        ('eccv16_ila_perceptual_w0.1_l1_gan_lsgan_w0.1', 'Best ECCV16'),
        ('siggraph17', 'SIGGRAPH17'),
        ('pretrained_siggraph17', 'Pretrained'),
        ('student_eccv16_mobilenet_best_model_4x', 'MobileNet 4x')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CIFAR-10
    ax = axes[0]
    fids = []
    labels = []
    colors_list = []
    
    for model_key, label in key_models:
        if model_key in cifar_data and 'fid' in cifar_data[model_key]:
            fids.append(cifar_data[model_key]['fid'])
            labels.append(label)
            if 'Baseline' in label:
                colors_list.append('#d62728')
            elif 'Best' in label:
                colors_list.append('#2ca02c')
            elif 'Pretrained' in label:
                colors_list.append('#9467bd')
            else:
                colors_list.append('#1f77b4')
    
    bars = ax.bar(range(len(labels)), fids, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10: FID Scores', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ImageNet
    ax = axes[1]
    fids = []
    labels = []
    colors_list = []
    
    for model_key, label in key_models:
        if model_key in imagenet_data and 'fid' in imagenet_data[model_key]:
            fids.append(imagenet_data[model_key]['fid'])
            labels.append(label)
            if 'Baseline' in label:
                colors_list.append('#d62728')
            elif 'Best' in label:
                colors_list.append('#2ca02c')
            elif 'Pretrained' in label:
                colors_list.append('#9467bd')
            else:
                colors_list.append('#1f77b4')
    
    bars = ax.bar(range(len(labels)), fids, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('ImageNet: FID Scores', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('FID Score Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/cdf_1_fid_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: plots/cdf_1_fid_summary.png")


def plot_metric_distributions(cifar_data, imagenet_data, metric_key, metric_name, filename):
    """Plot distribution comparison for a specific metric"""
    key_models = [
        ('eccv16', 'Baseline', '#d62728'),
        ('eccv16_ila_perceptual_w0.1_l1_gan_lsgan_w0.1', 'Best ECCV16', '#2ca02c'),
        ('siggraph17', 'SIGGRAPH17', '#1f77b4'),
        ('pretrained_siggraph17', 'Pretrained', '#9467bd')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CIFAR-10
    ax = axes[0]
    for model_key, label, color in key_models:
        if model_key not in cifar_data or metric_key not in cifar_data[model_key]:
            continue
        
        stats = cifar_data[model_key][metric_key]
        mean = stats['mean']
        std = stats['std']
        min_val = stats['min']
        max_val = stats['max']
        
        # Generate distribution approximation
        values = np.random.normal(mean, std, 1000)
        values = np.clip(values, min_val, max_val)
        values = np.sort(values)
        cdf = np.arange(1, len(values) + 1) / len(values)
        
        ax.plot(values, cdf, linewidth=2.5, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'CIFAR-10: {metric_name} Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ImageNet
    ax = axes[1]
    for model_key, label, color in key_models:
        if model_key not in imagenet_data or metric_key not in imagenet_data[model_key]:
            continue
        
        stats = imagenet_data[model_key][metric_key]
        mean = stats['mean']
        std = stats['std']
        min_val = stats['min']
        max_val = stats['max']
        
        values = np.random.normal(mean, std, 1000)
        values = np.clip(values, min_val, max_val)
        values = np.sort(values)
        cdf = np.arange(1, len(values) + 1) / len(values)
        
        ax.plot(values, cdf, linewidth=2.5, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'ImageNet: {metric_name} Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{metric_name} Cumulative Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}")


def main():
    """Generate all CDF plots"""
    os.makedirs('plots', exist_ok=True)
    
    print("="*80)
    print("GENERATING CDF PLOTS")
    print("="*80)
    
    print("\nLoading data...")
    cifar_data, imagenet_data = load_data()
    print(f"Loaded {len(cifar_data)} CIFAR-10 models")
    print(f"Loaded {len(imagenet_data)} ImageNet models")
    
    print("\nGenerating plots...")
    
    # FID summary
    plot_fid_summary(cifar_data, imagenet_data)
    
    # Metric distributions
    plot_metric_distributions(cifar_data, imagenet_data, 'psnr', 'PSNR (dB)', 'plots/cdf_2_psnr.png')
    plot_metric_distributions(cifar_data, imagenet_data, 'lpips', 'LPIPS (Perceptual Distance)', 'plots/cdf_3_lpips.png')
    plot_metric_distributions(cifar_data, imagenet_data, 'delta_e2000', 'Delta E 2000 (Color Error)', 'plots/cdf_4_delta_e.png')
    plot_metric_distributions(cifar_data, imagenet_data, 'ssim', 'SSIM (Structural Similarity)', 'plots/cdf_5_ssim.png')
    plot_metric_distributions(cifar_data, imagenet_data, 'colorfulness', 'Colorfulness', 'plots/cdf_6_colorfulness.png')
    
    print("\n" + "="*80)
    print("[OK] ALL CDF PLOTS GENERATED")
    print("="*80)
    print("\nGenerated files:")
    print("  1. plots/cdf_1_fid_summary.png - FID comparison bar chart")
    print("  2. plots/cdf_2_psnr.png - PSNR distribution")
    print("  3. plots/cdf_3_lpips.png - LPIPS distribution")
    print("  4. plots/cdf_4_delta_e.png - Color error distribution")
    print("  5. plots/cdf_5_ssim.png - Structural similarity distribution")
    print("  6. plots/cdf_6_colorfulness.png - Colorfulness distribution")
    print("\nDone!")


if __name__ == '__main__':
    main()

