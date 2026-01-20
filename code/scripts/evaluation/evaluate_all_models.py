#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates and compares ALL available models at once
"""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import json
import glob

# Try to import tabulate for nice tables, fallback to simple formatting
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17

def rgb_to_lab(rgb_tensor, device):
    """Convert RGB tensor to LAB tensor"""
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
    lab_np = np.zeros_like(rgb_np)
    for i in range(rgb_np.shape[0]):
        rgb_normalized = np.clip(rgb_np[i], 0, 1)
        lab_np[i] = color.rgb2lab(rgb_normalized)
    lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
    return lab_tensor.to(device)

def lab_to_rgb(lab_tensor):
    """Convert LAB tensor to RGB tensor"""
    lab_np = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()
    rgb_np = np.zeros_like(lab_np)
    for i in range(lab_np.shape[0]):
        rgb_np[i] = color.lab2rgb(lab_np[i])
        rgb_np[i] = np.clip(rgb_np[i], 0, 1)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
    return rgb_tensor

def calculate_metrics(original, colorized):
    """Calculate evaluation metrics"""
    original = np.clip(original, 0, 1)
    colorized = np.clip(colorized, 0, 1)
    psnr_value = psnr(original, colorized, data_range=1.0)
    ssim_value = ssim(original, colorized, data_range=1.0, channel_axis=2, multichannel=True)
    original_lab = color.rgb2lab(original)
    colorized_lab = color.rgb2lab(colorized)
    color_diff = np.mean(np.sqrt(np.sum((original_lab - colorized_lab)**2, axis=2)))
    return {'psnr': psnr_value, 'ssim': ssim_value, 'color_diff': color_diff}

def load_all_models(device):
    """Load all available models"""
    models = {}
    
    print("="*80)
    print("Loading Models")
    print("="*80)
    
    # Load pre-trained models
    try:
        models['pretrained_eccv16'] = eccv16(pretrained=True).eval().to(device)
        models['pretrained_siggraph17'] = siggraph17(pretrained=True).eval().to(device)
        print("✅ Pre-trained models loaded")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained models: {e}")
    
    # Find all model files
    model_files = []
    patterns = [
        'eccv16_best_model.pth',
        'eccv16_ila_best_model.pth',
        'eccv16*perceptual*best_model.pth',
        'eccv16*gan*best_model.pth',  # Include GAN models
        'siggraph17_best_model.pth',
        'siggraph17*perceptual*best_model.pth',
        'siggraph17*gan*best_model.pth'  # Include GAN models
    ]
    
    for pattern in patterns:
        model_files.extend(glob.glob(pattern))
    
    # Load each model
    for filepath in sorted(set(model_files)):
        if not os.path.isfile(filepath):
            continue
            
        try:
            # Determine model type and ILA usage
            is_eccv16 = 'eccv16' in filepath
            use_ila = 'ila' in filepath and is_eccv16
            
            # Skip discriminator files (they end with _discriminator_best.pth)
            if 'discriminator' in filepath:
                print(f"⏭️  Skipping discriminator: {filepath}")
                continue
            
            if is_eccv16:
                model = eccv16(pretrained=False, use_ila=use_ila).eval().to(device)
            else:
                model = siggraph17(pretrained=False).eval().to(device)
            
            model.load_state_dict(torch.load(filepath, map_location=device))
            
            # Create clean key name
            key = filepath.replace('_best_model.pth', '').replace('.pth', '')
            models[key] = model
            print(f"✅ Loaded: {filepath}")
        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
    
    print(f"\n📊 Total models loaded: {len(models)}")
    print("="*80)
    return models

def evaluate_model(model, model_key, test_loader, device, num_samples=1000):
    """Evaluate a single model"""
    metrics = {'psnr': [], 'ssim': [], 'color_diff': []}
    count = 0
    
    for batch_idx, (rgb_image, _) in enumerate(test_loader):
        if count >= num_samples:
            break
        
        try:
            rgb_image = rgb_image.to(device)
            original = rgb_image[0].permute(1, 2, 0).cpu().numpy()
            
            # Colorize
            lab_image = rgb_to_lab(rgb_image, device)
            l_channel = lab_image[:, 0:1, :, :]
            
            with torch.no_grad():
                predicted_ab = model(l_channel)
            
            predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
            predicted_rgb = lab_to_rgb(predicted_lab)
            colorized = predicted_rgb[0].permute(1, 2, 0).cpu().numpy()
            
            # Calculate metrics
            metric_values = calculate_metrics(original, colorized)
            metrics['psnr'].append(metric_values['psnr'])
            metrics['ssim'].append(metric_values['ssim'])
            metrics['color_diff'].append(metric_values['color_diff'])
            
            count += 1
            if count % 100 == 0:
                print(f"   {model_key}: {count}/{num_samples}")
        except Exception as e:
            print(f"   ⚠️  Error in {model_key}: {e}")
            continue
    
    return {
        'psnr': {'mean': float(np.mean(metrics['psnr'])), 'std': float(np.std(metrics['psnr']))},
        'ssim': {'mean': float(np.mean(metrics['ssim'])), 'std': float(np.std(metrics['ssim']))},
        'color_diff': {'mean': float(np.mean(metrics['color_diff'])), 'std': float(np.std(metrics['color_diff']))},
        'num_samples': count
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate and compare all models')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples (default: 1000)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda, auto-detects if not specified)')
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Load all models
    models = load_all_models(device)
    
    if len(models) == 0:
        print("❌ No models found! Please train models first.")
        return
    
    # Prepare test data
    print(f"\n📁 Preparing CIFAR-10 test data ({args.num_samples} samples)...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./cifar-10-python', train=False, download=True, transform=transform
    )
    if args.num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:args.num_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"✅ Test samples: {len(test_dataset)}\n")
    
    # Evaluate all models
    print("="*80)
    print("EVALUATION IN PROGRESS")
    print("="*80)
    results = {}
    
    for model_key in sorted(models.keys()):
        print(f"\n🔍 Evaluating: {model_key}")
        results[model_key] = evaluate_model(
            models[model_key], model_key, test_loader, device, args.num_samples
        )
    
    # Generate comparison report
    print("\n" + "="*80)
    print("EVALUATION RESULTS - COMPARISON TABLE")
    print("="*80)
    
    # Group by model type
    eccv16_results = {k: v for k, v in results.items() if 'eccv16' in k}
    siggraph17_results = {k: v for k, v in results.items() if 'siggraph17' in k}
    
    for model_type, model_results in [('ECCV16', eccv16_results), ('SIGGRAPH17', siggraph17_results)]:
        if not model_results:
            continue
        
        print(f"\n{model_type} Models:")
        print("-" * 80)
        
        # Prepare table data
        table_data = []
        for model_key in sorted(model_results.keys()):
            r = model_results[model_key]
            table_data.append([
                model_key,
                f"{r['psnr']['mean']:.4f} ± {r['psnr']['std']:.4f}",
                f"{r['ssim']['mean']:.4f} ± {r['ssim']['std']:.4f}",
                f"{r['color_diff']['mean']:.4f} ± {r['color_diff']['std']:.4f}"
            ])
        
        headers = ['Model', 'PSNR (↑)', 'SSIM (↑)', 'Color Diff (↓)']
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            # Simple table format without tabulate
            col_widths = [40, 20, 20, 20]
            print(' | '.join(h.ljust(w) for h, w in zip(headers, col_widths)))
            print('-' * sum(col_widths) + '-' * (len(headers) * 3))
            for row in table_data:
                print(' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
        print()
    
    # Save results
    output_file = 'all_models_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print(f"✅ Results saved to: {output_file}")
    print("="*80)
    
    # Summary statistics
    print("\n📊 SUMMARY:")
    print("-" * 80)
    if eccv16_results:
        best_psnr_eccv16 = max(eccv16_results.items(), key=lambda x: x[1]['psnr']['mean'])
        best_ssim_eccv16 = max(eccv16_results.items(), key=lambda x: x[1]['ssim']['mean'])
        best_color_eccv16 = min(eccv16_results.items(), key=lambda x: x[1]['color_diff']['mean'])
        print(f"ECCV16 Best PSNR: {best_psnr_eccv16[0]} ({best_psnr_eccv16[1]['psnr']['mean']:.4f})")
        print(f"ECCV16 Best SSIM: {best_ssim_eccv16[0]} ({best_ssim_eccv16[1]['ssim']['mean']:.4f})")
        print(f"ECCV16 Best Color: {best_color_eccv16[0]} ({best_color_eccv16[1]['color_diff']['mean']:.4f})")
    
    if siggraph17_results:
        best_psnr_sig = max(siggraph17_results.items(), key=lambda x: x[1]['psnr']['mean'])
        best_ssim_sig = max(siggraph17_results.items(), key=lambda x: x[1]['ssim']['mean'])
        best_color_sig = min(siggraph17_results.items(), key=lambda x: x[1]['color_diff']['mean'])
        print(f"\nSIGGRAPH17 Best PSNR: {best_psnr_sig[0]} ({best_psnr_sig[1]['psnr']['mean']:.4f})")
        print(f"SIGGRAPH17 Best SSIM: {best_ssim_sig[0]} ({best_ssim_sig[1]['ssim']['mean']:.4f})")
        print(f"SIGGRAPH17 Best Color: {best_color_sig[0]} ({best_color_sig[1]['color_diff']['mean']:.4f})")
    
    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()

