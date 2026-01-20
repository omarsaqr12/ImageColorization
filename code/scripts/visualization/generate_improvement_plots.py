#!/usr/bin/env python3
"""
Generate plots showing model improvements across different techniques
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# ============================================================================
# DATA FROM PRESENTATION
# ============================================================================

# CIFAR-10 Results
cifar10_data = {
    'ECCV16 Baseline': {'FID': 41.88, 'LPIPS': 0.055, 'ΔE': 13.07, 'PSNR': 24.98, 'Colorfulness': 0.049, 'SSIM': 0.934},
    'ECCV16 + Perceptual': {'FID': 44.13, 'LPIPS': 0.056, 'ΔE': 13.04, 'PSNR': 25.00, 'Colorfulness': 0.046, 'SSIM': 0.935},
    'ECCV16 + ILA': {'FID': 46.56, 'LPIPS': 0.058, 'ΔE': 13.12, 'PSNR': 24.95, 'Colorfulness': 0.042, 'SSIM': 0.934},
    'ECCV16 + Perc+ILA': {'FID': 43.43, 'LPIPS': 0.056, 'ΔE': 13.02, 'PSNR': 25.01, 'Colorfulness': 0.046, 'SSIM': 0.935},
    'ECCV16 + Perc+ILA+GAN': {'FID': 41.61, 'LPIPS': 0.055, 'ΔE': 13.01, 'PSNR': 25.04, 'Colorfulness': 0.047, 'SSIM': 0.935},
    'SIGGRAPH17': {'FID': 26.08, 'LPIPS': 0.043, 'ΔE': 12.53, 'PSNR': 25.25, 'Colorfulness': 0.065, 'SSIM': 0.940},
    'SIGGRAPH17 + Perc+GAN': {'FID': 22.77, 'LPIPS': 0.041, 'ΔE': 12.49, 'PSNR': 25.31, 'Colorfulness': 0.063, 'SSIM': 0.942},
}

# ImageNet Results
imagenet_data = {
    'ECCV16 Baseline': {'FID': 41.47, 'LPIPS': 0.217, 'ΔE': 21.12, 'PSNR': 19.68, 'Colorfulness': 0.143, 'SSIM': 0.836},
    'ECCV16 + Perceptual': {'FID': 39.58, 'LPIPS': 0.203, 'ΔE': 24.27, 'PSNR': 18.85, 'Colorfulness': 0.123, 'SSIM': 0.832},
    'ECCV16 + ILA': {'FID': 25.94, 'LPIPS': 0.179, 'ΔE': 20.30, 'PSNR': 21.21, 'Colorfulness': 0.130, 'SSIM': 0.850},
    'ECCV16 + ILA+Perc+GAN': {'FID': 21.26, 'LPIPS': 0.165, 'ΔE': 20.73, 'PSNR': 21.17, 'Colorfulness': 0.081, 'SSIM': 0.850},
    'SIGGRAPH17': {'FID': 22.15, 'LPIPS': 0.142, 'ΔE': 16.62, 'PSNR': 22.19, 'Colorfulness': 0.091, 'SSIM': 0.860},
    'SIGGRAPH17 Pretrained': {'FID': 12.77, 'LPIPS': 0.111, 'ΔE': 12.99, 'PSNR': 24.33, 'Colorfulness': 0.110, 'SSIM': 0.880},
}

# Student Models (Knowledge Distillation)
cifar10_students = {
    'Teacher (Baseline)': {'FID': 41.88, 'LPIPS': 0.055, 'Params': 32.2, 'Size_MB': 123, 'Speed_ms': 0.97},
    'Best ECCV16 Combo': {'FID': 41.61, 'LPIPS': 0.055, 'Params': 33.1, 'Size_MB': 126, 'Speed_ms': 1.32},
    'Student 8x': {'FID': 38.41, 'LPIPS': 0.054, 'Params': 0.41, 'Size_MB': 1.6, 'Speed_ms': 0.57},
    'MobileNet 4x': {'FID': 42.76, 'LPIPS': 0.056, 'Params': 0.07, 'Size_MB': 0.26, 'Speed_ms': 0.65},
}

imagenet_students = {
    'Teacher (Baseline)': {'FID': 41.47, 'LPIPS': 0.217, 'Params': 32.2, 'Size_MB': 123, 'Speed_ms': 3.30},
    'Best ECCV16 Combo': {'FID': 21.26, 'LPIPS': 0.165, 'Params': 33.1, 'Size_MB': 126, 'Speed_ms': 4.43},
    'Student 2x': {'FID': 30.13, 'LPIPS': 0.161, 'Params': 6.3, 'Size_MB': 24, 'Speed_ms': 1.16},
    'Student 4x': {'FID': 31.46, 'LPIPS': 0.159, 'Params': 1.6, 'Size_MB': 6.1, 'Speed_ms': 0.77},
    'MobileNet 4x': {'FID': 29.40, 'LPIPS': 0.152, 'Params': 0.07, 'Size_MB': 0.26, 'Speed_ms': 0.84},
}


def plot_technique_progression():
    """Plot how different techniques improve FID scores"""
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    
    # CIFAR-10
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Baseline', '+ Perceptual', '+ ILA', '+ Perc+ILA', '+ Perc+ILA+GAN', 'SIGGRAPH17', 'SIGG17\n+Perc+GAN']
    fid_scores = [41.88, 44.13, 46.56, 43.43, 41.61, 26.08, 22.77]
    colors = ['#d62728' if fid > 41.88 else '#2ca02c' for fid in fid_scores]
    colors[-2:] = ['#1f77b4', '#1f77b4']  # Blue for SIGGRAPH17
    
    bars = ax1.bar(range(len(models)), fid_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=41.88, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('CIFAR-10: Technique Progression', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ImageNet
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Baseline', '+ Perceptual', '+ ILA', '+ ILA+Perc+GAN', 'SIGGRAPH17', 'SIGG17\nPretrained']
    fid_scores = [41.47, 39.58, 25.94, 21.26, 22.15, 12.77]
    colors = ['#2ca02c' if fid < 41.47 else '#d62728' for fid in fid_scores]
    colors[-2:] = ['#1f77b4', '#1f77b4']  # Blue for SIGGRAPH17
    
    bars = ax2.bar(range(len(models)), fid_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=41.47, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax2.set_title('ImageNet: Technique Progression', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Improvement Across Techniques', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('plots/1_technique_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/1_technique_progression.png")


def plot_dataset_comparison():
    """Compare technique effectiveness across datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # FID comparison
    ax = axes[0, 0]
    techniques = ['Baseline', '+ Perceptual', '+ ILA', 'Best Combo', 'SIGGRAPH17']
    cifar_fid = [41.88, 44.13, 46.56, 41.61, 26.08]
    imagenet_fid = [41.47, 39.58, 25.94, 21.26, 22.15]
    
    x = np.arange(len(techniques))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cifar_fid, width, label='CIFAR-10', color='#ff7f0e', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, imagenet_fid, width, label='ImageNet', color='#1f77b4', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('FID: Technique Effectiveness by Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # LPIPS comparison
    ax = axes[0, 1]
    cifar_lpips = [0.055, 0.056, 0.058, 0.055, 0.043]
    imagenet_lpips = [0.217, 0.203, 0.179, 0.165, 0.142]
    
    bars1 = ax.bar(x - width/2, cifar_lpips, width, label='CIFAR-10', color='#ff7f0e', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, imagenet_lpips, width, label='ImageNet', color='#1f77b4', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('LPIPS (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('LPIPS: Perceptual Quality by Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Color Error (Delta E) comparison
    ax = axes[1, 0]
    cifar_de = [13.07, 13.04, 13.12, 13.01, 12.53]
    imagenet_de = [21.12, 24.27, 20.30, 20.73, 16.62]
    
    bars1 = ax.bar(x - width/2, cifar_de, width, label='CIFAR-10', color='#ff7f0e', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, imagenet_de, width, label='ImageNet', color='#1f77b4', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔE Color Error (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('ΔE: Color Accuracy by Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Colorfulness comparison
    ax = axes[1, 1]
    cifar_color = [0.049, 0.046, 0.042, 0.047, 0.065]
    imagenet_color = [0.143, 0.123, 0.130, 0.081, 0.091]
    
    bars1 = ax.bar(x - width/2, cifar_color, width, label='CIFAR-10', color='#ff7f0e', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, imagenet_color, width, label='ImageNet', color='#1f77b4', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Colorfulness (higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('Colorfulness: Output Vibrancy by Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Cross-Dataset Technique Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/2_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/2_dataset_comparison.png")


def plot_improvement_percentages():
    """Plot percentage improvements over baseline"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CIFAR-10
    ax = axes[0]
    techniques = ['Perceptual', 'ILA', 'Perc+ILA', 'Perc+ILA+GAN', 'SIGGRAPH17', 'SIGG17\n+Perc+GAN']
    fid_improvements = [
        (41.88 - 44.13) / 41.88 * 100,  # -5.4%
        (41.88 - 46.56) / 41.88 * 100,  # -11.2%
        (41.88 - 43.43) / 41.88 * 100,  # -3.7%
        (41.88 - 41.61) / 41.88 * 100,  # +0.6%
        (41.88 - 26.08) / 41.88 * 100,  # +37.7%
        (41.88 - 22.77) / 41.88 * 100,  # +45.6%
    ]
    
    colors = ['#d62728' if imp < 0 else '#2ca02c' for imp in fid_improvements]
    bars = ax.barh(techniques, fid_improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('FID Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10: Relative Improvements', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, fid_improvements)):
        color = 'white' if abs(val) > 15 else 'black'
        ax.text(val/2 if abs(val) > 15 else val + 1, i, f'{val:+.1f}%',
               ha='center' if abs(val) > 15 else 'left', va='center',
               fontsize=10, fontweight='bold', color=color)
    
    # ImageNet
    ax = axes[1]
    techniques = ['Perceptual', 'ILA', 'ILA+Perc+GAN', 'SIGGRAPH17', 'SIGG17\nPretrained']
    fid_improvements = [
        (41.47 - 39.58) / 41.47 * 100,  # +4.6%
        (41.47 - 25.94) / 41.47 * 100,  # +37.4%
        (41.47 - 21.26) / 41.47 * 100,  # +48.7%
        (41.47 - 22.15) / 41.47 * 100,  # +46.6%
        (41.47 - 12.77) / 41.47 * 100,  # +69.2%
    ]
    
    colors = ['#2ca02c' for _ in fid_improvements]
    bars = ax.barh(techniques, fid_improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('FID Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('ImageNet: Relative Improvements', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, fid_improvements)):
        color = 'white' if val > 20 else 'black'
        ax.text(val/2 if val > 20 else val + 1, i, f'{val:+.1f}%',
               ha='center' if val > 20 else 'left', va='center',
               fontsize=10, fontweight='bold', color=color)
    
    plt.suptitle('Percentage Improvement Over Baseline (ECCV16)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/3_improvement_percentages.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/3_improvement_percentages.png")


def plot_distillation_tradeoffs():
    """Plot knowledge distillation results showing quality vs efficiency tradeoffs"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # CIFAR-10: FID vs Model Size
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(cifar10_students.keys())
    sizes = [cifar10_students[m]['Size_MB'] for m in models]
    fids = [cifar10_students[m]['FID'] for m in models]
    colors = ['red', 'orange', 'green', 'blue']
    
    for i, (model, size, fid, c) in enumerate(zip(models, sizes, fids, colors)):
        ax1.scatter(size, fid, s=400, alpha=0.7, c=c, edgecolors='black', linewidth=2, zorder=3)
        ax1.annotate(model, (size, fid), fontsize=9, ha='center', va='bottom', 
                    xytext=(0, 8), textcoords='offset points', fontweight='bold')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Size (MB, log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('CIFAR-10: Quality vs Model Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Smaller is better
    
    # ImageNet: FID vs Model Size
    ax2 = fig.add_subplot(gs[0, 1])
    models = list(imagenet_students.keys())
    sizes = [imagenet_students[m]['Size_MB'] for m in models]
    fids = [imagenet_students[m]['FID'] for m in models]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'blue']
    
    for i, (model, size, fid, c) in enumerate(zip(models, sizes, fids, colors)):
        ax2.scatter(size, fid, s=400, alpha=0.7, c=c, edgecolors='black', linewidth=2, zorder=3)
        ax2.annotate(model, (size, fid), fontsize=9, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', fontweight='bold')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Model Size (MB, log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax2.set_title('ImageNet: Quality vs Model Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # Smaller is better
    
    # CIFAR-10: FID vs Speed
    ax3 = fig.add_subplot(gs[1, 0])
    models = list(cifar10_students.keys())
    speeds = [cifar10_students[m]['Speed_ms'] for m in models]
    fids = [cifar10_students[m]['FID'] for m in models]
    colors = ['red', 'orange', 'green', 'blue']
    
    for i, (model, speed, fid, c) in enumerate(zip(models, speeds, fids, colors)):
        ax3.scatter(speed, fid, s=400, alpha=0.7, c=c, edgecolors='black', linewidth=2, zorder=3)
        ax3.annotate(model, (speed, fid), fontsize=9, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', fontweight='bold')
    
    ax3.set_xlabel('Inference Speed (ms, lower is faster)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax3.set_title('CIFAR-10: Quality vs Speed', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()  # Faster is better
    
    # ImageNet: FID vs Speed
    ax4 = fig.add_subplot(gs[1, 1])
    models = list(imagenet_students.keys())
    speeds = [imagenet_students[m]['Speed_ms'] for m in models]
    fids = [imagenet_students[m]['FID'] for m in models]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'blue']
    
    for i, (model, speed, fid, c) in enumerate(zip(models, speeds, fids, colors)):
        ax4.scatter(speed, fid, s=400, alpha=0.7, c=c, edgecolors='black', linewidth=2, zorder=3)
        ax4.annotate(model, (speed, fid), fontsize=9, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', fontweight='bold')
    
    ax4.set_xlabel('Inference Speed (ms, lower is faster)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax4.set_title('ImageNet: Quality vs Speed', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()  # Faster is better
    
    plt.suptitle('Knowledge Distillation: Quality vs Efficiency Trade-offs', fontsize=16, fontweight='bold')
    plt.savefig('plots/4_distillation_tradeoffs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/4_distillation_tradeoffs.png")


def plot_compression_ratios():
    """Plot compression ratios and quality retention"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CIFAR-10
    ax = axes[0]
    models = ['Teacher\n(Baseline)', 'Best Combo', 'Student 8x', 'MobileNet 4x']
    compression_ratios = [1, 0.98, 80, 473]  # relative to teacher
    fid_scores = [41.88, 41.61, 38.41, 42.76]
    
    x = np.arange(len(models))
    
    # Create bar chart
    bars = ax.bar(x, fid_scores, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10: Compression vs Quality', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', alpha=0.3)
    
    # Color bars based on improvement
    colors = ['gray', 'gray', '#2ca02c', '#d62728']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add compression ratio labels
    for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
        if ratio > 1:
            label = f'{ratio}× smaller'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # ImageNet
    ax = axes[1]
    models = ['Teacher\n(Baseline)', 'Best Combo', 'Student 2x', 'Student 4x', 'MobileNet 4x']
    compression_ratios = [1, 0.96, 5.1, 20.2, 473]  # relative to teacher
    fid_scores = [41.47, 21.26, 30.13, 31.46, 29.40]
    
    x = np.arange(len(models))
    
    # Create bar chart
    bars = ax.bar(x, fid_scores, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('ImageNet: Compression vs Quality', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Color bars
    colors = ['gray', '#9467bd', '#2ca02c', '#2ca02c', '#2ca02c']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add compression ratio labels
    for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
        if ratio > 1:
            label = f'{ratio:.0f}× smaller' if ratio >= 10 else f'{ratio:.1f}× smaller'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Model Compression: Size Reduction vs Quality', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/5_compression_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/5_compression_ratios.png")


def plot_multi_metric_radar():
    """Create radar charts comparing models across multiple metrics"""
    from math import pi
    
    fig = plt.figure(figsize=(16, 8))
    
    # Prepare data (normalize all metrics to 0-1 scale where higher is better)
    def normalize_metric(value, min_val, max_val, inverse=False):
        """Normalize to 0-1, optionally inverse for 'lower is better' metrics"""
        normalized = (value - min_val) / (max_val - min_val)
        return 1 - normalized if inverse else normalized
    
    # CIFAR-10 radar
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    
    categories = ['FID\n(inv)', 'LPIPS\n(inv)', 'ΔE\n(inv)', 'PSNR', 'Colorfulness', 'SSIM']
    num_vars = len(categories)
    
    # Normalize metrics for comparison
    models_to_plot = ['ECCV16 Baseline', 'ECCV16 + Perc+ILA+GAN', 'SIGGRAPH17', 'SIGGRAPH17 + Perc+GAN']
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    ax1.set_theta_offset(pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=10)
    
    colors_radar = ['red', 'orange', 'blue', 'green']
    
    for model, color in zip(models_to_plot, colors_radar):
        data = cifar10_data[model]
        values = [
            normalize_metric(data['FID'], 20, 50, inverse=True),
            normalize_metric(data['LPIPS'], 0.04, 0.06, inverse=True),
            normalize_metric(data['ΔE'], 12.4, 13.2, inverse=True),
            normalize_metric(data['PSNR'], 24.8, 25.5, inverse=False),
            normalize_metric(data['Colorfulness'], 0.04, 0.07, inverse=False),
            normalize_metric(data['SSIM'], 0.93, 0.95, inverse=False),
        ]
        values += values[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=model.replace('ECCV16 ', '').replace('SIGGRAPH17', 'SIGG17'), color=color)
        ax1.fill(angles, values, alpha=0.15, color=color)
    
    ax1.set_ylim(0, 1)
    ax1.set_title('CIFAR-10: Multi-Metric Comparison\n(normalized, outer=better)', 
                  size=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax1.grid(True)
    
    # ImageNet radar
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    
    models_to_plot = ['ECCV16 Baseline', 'ECCV16 + ILA+Perc+GAN', 'SIGGRAPH17', 'SIGGRAPH17 Pretrained']
    
    ax2.set_theta_offset(pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=10)
    
    for model, color in zip(models_to_plot, colors_radar):
        data = imagenet_data[model]
        values = [
            normalize_metric(data['FID'], 12, 42, inverse=True),
            normalize_metric(data['LPIPS'], 0.11, 0.22, inverse=True),
            normalize_metric(data['ΔE'], 12, 25, inverse=True),
            normalize_metric(data['PSNR'], 18.5, 24.5, inverse=False),
            normalize_metric(data['Colorfulness'], 0.08, 0.15, inverse=False),
            normalize_metric(data['SSIM'], 0.83, 0.88, inverse=False),
        ]
        values += values[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, label=model.replace('ECCV16 ', '').replace('SIGGRAPH17', 'SIGG17'), color=color)
        ax2.fill(angles, values, alpha=0.15, color=color)
    
    ax2.set_ylim(0, 1)
    ax2.set_title('ImageNet: Multi-Metric Comparison\n(normalized, outer=better)', 
                  size=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax2.grid(True)
    
    plt.suptitle('Multi-Metric Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/6_multi_metric_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/6_multi_metric_radar.png")


def plot_pareto_frontier():
    """Plot Pareto frontier for quality vs efficiency tradeoff"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ImageNet: FID vs Params (main tradeoff visualization)
    ax = axes[0]
    
    models_data = []
    for name, data in imagenet_students.items():
        models_data.append({
            'name': name,
            'params': data['Params'],
            'fid': data['FID'],
            'size_mb': data['Size_MB']
        })
    
    # Sort by params
    models_data = sorted(models_data, key=lambda x: x['params'])
    
    params = [m['params'] for m in models_data]
    fids = [m['fid'] for m in models_data]
    names = [m['name'] for m in models_data]
    
    # Color code: green for students, gray for teachers
    colors = ['red' if 'Teacher' in n or 'Combo' in n else 'green' for n in names]
    
    # Plot points
    for i, (p, f, n, c) in enumerate(zip(params, fids, names, colors)):
        ax.scatter(p, f, s=500, alpha=0.7, c=c, edgecolors='black', linewidth=2, zorder=3)
        
        # Position labels
        if 'Best' in n:
            xytext = (10, -15)
        elif 'Teacher' in n:
            xytext = (10, 10)
        elif 'Student 2x' in n:
            xytext = (-15, -20)
        elif 'Student 4x' in n:
            xytext = (10, -15)
        else:
            xytext = (10, 5)
            
        ax.annotate(n, (p, f), fontsize=9, fontweight='bold',
                   xytext=xytext, textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))
    
    # Draw Pareto frontier (connecting dominant points)
    pareto_points = [
        ('MobileNet 4x', 0.07, 29.40),
        ('Student 4x', 1.6, 31.46),
        ('Student 2x', 6.3, 30.13),
        ('Best ECCV16 Combo', 33.1, 21.26),
    ]
    pareto_x = [p[1] for p in pareto_points]
    pareto_y = [p[2] for p in pareto_points]
    ax.plot(pareto_x, pareto_y, 'b--', linewidth=2, alpha=0.5, label='Pareto Frontier', zorder=1)
    
    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters (millions, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('ImageNet: Pareto Frontier (Quality vs Size)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ImageNet: Compression achievements
    ax = axes[1]
    
    # Calculate improvements over baseline teacher
    baseline_fid = 41.47
    
    improvements = []
    for m in models_data:
        if m['name'] != 'Teacher (Baseline)':
            fid_improvement = (baseline_fid - m['fid']) / baseline_fid * 100
            compression = 123 / m['size_mb']  # relative to teacher 123 MB
            improvements.append({
                'name': m['name'],
                'fid_improvement': fid_improvement,
                'compression': compression
            })
    
    # Sort by compression
    improvements = sorted(improvements, key=lambda x: x['compression'])
    
    names = [i['name'] for i in improvements]
    x = np.arange(len(names))
    
    # Create dual y-axis plot
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    ax_twin = ax.twinx()
    
    bars1 = ax.bar(x - 0.2, [i['fid_improvement'] for i in improvements], 
                   0.4, label='FID Improvement (%)', color=color1, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax_twin.bar(x + 0.2, [i['compression'] for i in improvements], 
                        0.4, label='Compression Ratio (×)', color=color2, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Improvement over Baseline (%)', fontsize=11, fontweight='bold', color=color1)
    ax_twin.set_ylabel('Compression Ratio (× smaller)', fontsize=11, fontweight='bold', color=color2)
    ax.set_title('ImageNet: Students Beat Teachers with Compression', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelcolor=color1)
    ax_twin.tick_params(axis='y', labelcolor=color2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, [i['fid_improvement'] for i in improvements])):
        ax.text(bar.get_x() + bar.get_width()/2., val,
               f'{val:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (bar, val) in enumerate(zip(bars2, [i['compression'] for i in improvements])):
        ax_twin.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.0f}×',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.suptitle('Quality-Efficiency Pareto Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/7_pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/7_pareto_frontier.png")


def main():
    """Generate all plots"""
    import os
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    print("="*80)
    print("GENERATING MODEL IMPROVEMENT PLOTS")
    print("="*80)
    print()
    
    # Generate all plots
    plot_technique_progression()
    plot_dataset_comparison()
    plot_improvement_percentages()
    plot_distillation_tradeoffs()
    plot_compression_ratios()
    plot_multi_metric_radar()
    plot_pareto_frontier()
    
    print()
    print("="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*80)
    print()
    print("📁 Plots saved in: plots/")
    print()
    print("Generated plots:")
    print("  1. plots/1_technique_progression.png - FID improvements across techniques")
    print("  2. plots/2_dataset_comparison.png - Cross-dataset technique comparison")
    print("  3. plots/3_improvement_percentages.png - Relative improvements over baseline")
    print("  4. plots/4_distillation_tradeoffs.png - Quality vs efficiency trade-offs")
    print("  5. plots/5_compression_ratios.png - Compression vs quality retention")
    print("  6. plots/6_multi_metric_radar.png - Multi-metric radar charts")
    print("  7. plots/7_pareto_frontier.png - Pareto frontier analysis")
    print()
    print("🎉 Done!")


if __name__ == '__main__':
    main()

