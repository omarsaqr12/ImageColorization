#!/usr/bin/env python3
"""
Comprehensive Colorization with Top 20 Models

This script:
1. Loads the TOP 20 most important models (baselines + best variants + students)
2. Colorizes 500 images from ImageNet and 500 from CIFAR-10
3. Creates comparison visualizations
4. All models loaded to GPU for fast inference
"""

import os
import sys
import json
import warnings
import gc
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import glob
import random
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Conversion from CIE-LAB.*')
warnings.filterwarnings('ignore', message='.*negative Z values.*')

# Add paths
sys.path.append('colorization')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'training'))
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens
from eccv16_variants import build_eccv16_variant
from student import LightweightStudent, MobileNetStyleStudent


def rgb_to_lab_tensor(rgb_tensor, device):
    """Convert RGB tensor [B, 3, H, W] to LAB tensor"""
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
    lab_np = np.zeros_like(rgb_np)
    for i in range(rgb_np.shape[0]):
        rgb_normalized = np.clip(rgb_np[i], 0, 1)
        lab_np[i] = color.rgb2lab(rgb_normalized)
    lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
    return lab_tensor.to(device)


def lab_to_rgb_tensor(lab_tensor):
    """Convert LAB tensor [B, 3, H, W] to RGB tensor"""
    lab_np = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()
    rgb_np = np.zeros_like(lab_np)
    for i in range(lab_np.shape[0]):
        rgb_np[i] = color.lab2rgb(lab_np[i])
        rgb_np[i] = np.clip(rgb_np[i], 0, 1)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
    return rgb_tensor


def find_variant_directory(experiments_root, variant_id, init_mode):
    """Find the variant directory with various naming patterns"""
    # Try different naming patterns
    patterns = [
        f"variant_{variant_id}_{init_mode}",  # variant_variant_070_random
        f"{variant_id}_{init_mode}",           # variant_070_random
        f"variant_{variant_id}",               # variant_variant_070
        variant_id,                             # variant_070
    ]
    
    for pattern in patterns:
        variant_dir = os.path.join(experiments_root, pattern)
        if os.path.exists(variant_dir):
            return variant_dir
    
    # Also search by scanning the directory
    if os.path.exists(experiments_root):
        for item in os.listdir(experiments_root):
            item_path = os.path.join(experiments_root, item)
            if os.path.isdir(item_path):
                # Check if this directory matches the variant
                if variant_id in item and init_mode in item:
                    return item_path
    
    return None


def get_top_20_models_info(experiments_root='experiments', students_dir='experiments/distilled_students'):
    """
    Get info for the top 20 most important models:
    - 2 baseline models (eccv16, siggraph17)
    - 6 student models (all)
    - 12 best variant models (best from each metric)
    """
    models_to_load = {}
    
    print("="*80)
    print("Selecting Top 20 Most Important Models")
    print("="*80)
    
    # 1. Baseline models (2)
    models_to_load['baseline_eccv16'] = {
        'type': 'baseline',
        'baseline_type': 'eccv16',
        'use_pretrained_preprocessing': True,
        'display_name': 'Baseline ECCV16'
    }
    models_to_load['baseline_siggraph17'] = {
        'type': 'baseline',
        'baseline_type': 'siggraph17',
        'use_pretrained_preprocessing': True,
        'display_name': 'Baseline SIGGRAPH17'
    }
    print("✅ Added 2 baseline models")
    
    # 2. Student models (up to 6)
    if os.path.exists(students_dir):
        student_patterns = [
            os.path.join(students_dir, 'student_*_best_model.pth'),
            os.path.join(students_dir, 'student_*.pth')
        ]
        model_files = []
        for pattern in student_patterns:
            model_files.extend(glob.glob(pattern))
        
        student_count = 0
        for filepath in sorted(set(model_files)):
            if not os.path.isfile(filepath):
                continue
            filename = os.path.basename(filepath)
            
            if 'lightweight' in filename:
                parts = filename.replace('_best_model.pth', '').replace('.pth', '').split('_')
                reduction = int(parts[-1].replace('x', ''))
                model_key = f"student_lightweight_{reduction}x"
                models_to_load[model_key] = {
                    'type': 'student',
                    'student_type': 'lightweight',
                    'reduction': reduction,
                    'checkpoint': filepath,
                    'display_name': f'Student Light {reduction}x'
                }
                student_count += 1
            elif 'mobilenet' in filename:
                parts = filename.replace('_best_model.pth', '').replace('.pth', '').split('_')
                reduction = int(parts[-1].replace('x', ''))
                model_key = f"student_mobilenet_{reduction}x"
                models_to_load[model_key] = {
                    'type': 'student',
                    'student_type': 'mobilenet',
                    'reduction': reduction,
                    'checkpoint': filepath,
                    'display_name': f'Student Mobile {reduction}x'
                }
                student_count += 1
        print(f"✅ Added {student_count} student models")
    
    # 3. Best variant models - get BEST from each metric
    best_models_file = 'best_models_by_metric.json'
    variants_to_load = {}  # Use dict to track display names
    
    # Always include the best overall model (KD Teacher) first
    variants_to_load[('variant_097', 'random')] = '⭐ Best Overall (Teacher)'
    
    if os.path.exists(best_models_file):
        with open(best_models_file, 'r') as f:
            best_data = json.load(f)
        
        # Define which metrics to use and their display names
        metrics_to_load = [
            ('imagenet_psnr', 'Best ImgNet PSNR'),
            ('imagenet_ssim', 'Best ImgNet SSIM'),
            ('imagenet_lpips', 'Best ImgNet LPIPS'),
            ('imagenet_deltaE', 'Best ImgNet ΔE'),
            ('cifar10_psnr', 'Best CIFAR PSNR'),
            ('cifar10_ssim', 'Best CIFAR SSIM'),
            ('cifar10_lpips', 'Best CIFAR LPIPS'),
            ('cifar10_deltaE', 'Best CIFAR ΔE'),
        ]
        
        for metric_key, display_prefix in metrics_to_load:
            if metric_key in best_data and best_data[metric_key]:
                # Get top 1-2 models from each metric
                for i, model in enumerate(best_data[metric_key][:2]):
                    variant_id = model.get('variant_id', '')
                    init_mode = model.get('init_mode', '')
                    if variant_id and init_mode:
                        key = (variant_id, init_mode)
                        if key not in variants_to_load:
                            rank = "" if i == 0 else " #2"
                            variants_to_load[key] = f"{display_prefix}{rank}"
        
        print(f"📋 Found {len(variants_to_load)} unique best variants (including Teacher)")
    else:
        print(f"⚠️  {best_models_file} not found")
        print(f"📋 Will load Teacher model (variant_097_random)")
    
    # Load the best variants
    variant_count = 0
    remaining_slots = 20 - len(models_to_load)
    
    for (variant_id, init_mode), display_name in list(variants_to_load.items())[:remaining_slots]:
        # Find the variant directory
        variant_dir = find_variant_directory(experiments_root, variant_id, init_mode)
        
        if not variant_dir:
            print(f"⚠️  Directory not found for {variant_id}_{init_mode}")
            continue
        
        # Check for checkpoint
        checkpoints_dir = os.path.join(variant_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            print(f"⚠️  No checkpoints dir for {variant_id}")
            continue
        
        checkpoint_path = os.path.join(checkpoints_dir, 'final.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoints_dir, 'warmup.pth')
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  No checkpoint for {variant_id}")
                continue
        
        # Load config
        config_path = os.path.join(variant_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"⚠️  No config for {variant_id}")
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_key = f"{variant_id}_{init_mode}"
            models_to_load[model_key] = {
                'type': 'variant',
                'config': config,
                'checkpoint': checkpoint_path,
                'display_name': display_name
            }
            variant_count += 1
            print(f"   📌 {model_key}: {display_name}")
        except Exception as e:
            print(f"⚠️  Failed to load config for {variant_id}: {e}")
            continue
    
    print(f"✅ Added {variant_count} best variant models")
    print(f"\n📊 Total: {len(models_to_load)} models selected")
    
    return models_to_load


def load_all_models_to_gpu(models_info, device):
    """Load all selected models to GPU"""
    models = {}
    
    print("\n" + "="*80)
    print(f"Loading All {len(models_info)} Models to GPU")
    print("="*80)
    
    for model_key, info in models_info.items():
        try:
            model_type = info.get('type', 'variant')
            
            if model_type == 'baseline':
                baseline_type = info.get('baseline_type', 'eccv16')
                if baseline_type == 'eccv16':
                    model = eccv16(pretrained=True).eval().to(device)
                else:
                    model = siggraph17(pretrained=True).eval().to(device)
            
            elif model_type == 'student':
                student_type = info.get('student_type', 'lightweight')
                reduction = info.get('reduction', 2)
                checkpoint = info.get('checkpoint')
                
                if student_type == 'lightweight':
                    model = LightweightStudent(channel_reduction=reduction).eval()
                else:
                    width_mult = 1.0 / reduction
                    model = MobileNetStyleStudent(width_multiplier=width_mult).eval()
                
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                model = model.to(device)
            
            else:  # variant
                config = info.get('config', {})
                checkpoint = info.get('checkpoint')
                
                model = build_eccv16_variant(**config).eval()
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                model = model.to(device)
            
            models[model_key] = {
                'model': model,
                'use_pretrained_preprocessing': info.get('use_pretrained_preprocessing', False),
                'display_name': info.get('display_name', model_key)
            }
            print(f"✅ Loaded: {model_key}")
            
        except Exception as e:
            print(f"⚠️  Failed to load {model_key}: {e}")
            continue
    
    print(f"\n📊 Successfully loaded {len(models)} models to GPU")
    
    # Show GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"💾 GPU Memory: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")
    
    return models


def colorize_with_model(rgb_image, model_info, device):
    """Colorize image with a pre-loaded model (fast - no disk I/O)"""
    model = model_info['model']
    use_pretrained_preprocessing = model_info.get('use_pretrained_preprocessing', False)
    
    if use_pretrained_preprocessing:
        # Use pretrained preprocessing (for baseline models)
        if rgb_image.dim() == 4:
            rgb_image = rgb_image[0]
        
        rgb_np = rgb_image.permute(1, 2, 0).cpu().numpy()
        rgb_np = np.clip(rgb_np, 0, 1)
        rgb_np_uint8 = (rgb_np * 255).astype(np.uint8)
        
        (tens_l_orig, tens_l_rs) = preprocess_img(rgb_np_uint8, HW=(256, 256))
        tens_l_rs = tens_l_rs.to(device)
        
        with torch.no_grad():
            colorized_ab = model(tens_l_rs).cpu()
        
        colorized = postprocess_tens(tens_l_orig, colorized_ab)
        return colorized
    else:
        # Use LAB preprocessing (for trained models)
        if rgb_image.dim() == 3:
            rgb_image = rgb_image.unsqueeze(0)
        
        rgb_image = rgb_image.to(device)
        lab_image = rgb_to_lab_tensor(rgb_image, device)
        l_channel = lab_image[:, 0:1, :, :]
        
        with torch.no_grad():
            output = model(l_channel)
            if isinstance(output, dict):
                predicted_ab = output.get('ab_output', output.get('ab', None))
                if predicted_ab is None:
                    for v in output.values():
                        if isinstance(v, torch.Tensor) and v.shape[1] == 2:
                            predicted_ab = v
                            break
                    if predicted_ab is None:
                        predicted_ab = list(output.values())[0]
            else:
                predicted_ab = output
        
        predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
        predicted_rgb = lab_to_rgb_tensor(predicted_lab)
        
        predicted_rgb_np = predicted_rgb[0].permute(1, 2, 0).cpu().numpy()
        predicted_rgb_np = np.clip(predicted_rgb_np, 0, 1)
        return predicted_rgb_np


def create_comparison_image(
    original_rgb, grayscale, colorized_results, output_path, image_idx
):
    """Create comparison visualization for top 20 models"""
    num_models = len(colorized_results)
    
    # Layout: 5 columns, enough rows for all models + original + grayscale
    cols = 5
    rows = (num_models + 2 + cols - 1) // cols
    
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_rgb)
    ax.set_title('Original', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Grayscale
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(grayscale, cmap='gray')
    ax.set_title('Grayscale', fontsize=10)
    ax.axis('off')
    
    # All models
    for idx, (model_name, (colorized, display_name)) in enumerate(colorized_results.items(), start=2):
        row = idx // cols
        col = idx % cols
        if row < rows:
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(colorized)
            title = display_name if len(display_name) <= 20 else display_name[:17] + '...'
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    
    plt.suptitle(f'Image {image_idx + 1} - Top 20 Models', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_dataset(dataset_name, all_models, device, num_images=500, output_dir=None):
    """Process a dataset with all models pre-loaded in GPU"""
    
    print("\n" + "="*80)
    print(f"Processing {dataset_name.upper()}")
    print("="*80)
    
    # Setup dataset
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10_data' if os.path.exists('./data/cifar10_data') else './cifar-10-python',
            train=False,
            download=True,
            transform=transform
        )
        if num_images < len(dataset):
            indices = torch.randperm(len(dataset))[:num_images]
            dataset = torch.utils.data.Subset(dataset, indices)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
    elif dataset_name.lower() == 'imagenet':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'evaluation'))
        from variant_evaluation import ImageNetSubsetDataset
        
        imagenet_root = os.environ.get(
            "IMAGENET_VAL_ROOT",
            "/home/mohab/Downloads/work/a/imagenet-object-localization-challenge/ILSVRC"
        )
        
        if not os.path.exists(imagenet_root):
            print(f"❌ ImageNet root not found: {imagenet_root}")
            print("   Set IMAGENET_VAL_ROOT environment variable")
            return
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        dataset = ImageNetSubsetDataset(
            root_dir=imagenet_root,
            transform=transform,
            max_samples=num_images
        )
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return
    
    if output_dir is None:
        output_dir = f"results/colorized_images_{dataset_name.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Processing {len(dataset)} images with {len(all_models)} models")
    print(f"💡 All models in GPU - fast inference!\n")
    
    for image_idx, data in enumerate(tqdm(test_loader, desc=f"Colorizing {dataset_name}")):
        if image_idx >= num_images:
            break
        
        if dataset_name.lower() == 'cifar10':
            rgb_image, _ = data
        else:
            if isinstance(data, (list, tuple)) and len(data) >= 1:
                rgb_image = data[0]
            else:
                rgb_image = data
        
        original_rgb = rgb_image[0].permute(1, 2, 0).cpu().numpy()
        original_rgb = np.clip(original_rgb, 0, 1)
        grayscale = np.mean(original_rgb, axis=2)
        
        colorized_results = {}
        
        for model_name, model_info in all_models.items():
            try:
                colorized_img = colorize_with_model(rgb_image, model_info, device)
                display_name = model_info.get('display_name', model_name)
                colorized_results[model_name] = (colorized_img, display_name)
            except Exception as e:
                print(f"\n⚠️  Error with {model_name}: {e}")
                continue
        
        if colorized_results:
            output_path = os.path.join(output_dir, f"image_{image_idx + 1:05d}.png")
            create_comparison_image(original_rgb, grayscale, colorized_results, output_path, image_idx)
    
    print(f"\n✅ Completed {dataset_name}: {len(dataset)} images processed")
    print(f"   Results saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Colorize images with top 20 most important models'
    )
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu/cuda). Auto-detect if not specified')
    parser.add_argument('--num_images', type=int, default=500,
                       help='Number of images per dataset (default: 500)')
    parser.add_argument('--cifar10_dir', type=str, 
                       default='results/colorized_images_cifar10',
                       help='Output directory for CIFAR-10 images')
    parser.add_argument('--imagenet_dir', type=str,
                       default='results/colorized_images_imagenet',
                       help='Output directory for ImageNet images')
    parser.add_argument('--experiments_root', type=str, default='experiments',
                       help='Root directory for variant experiments')
    parser.add_argument('--students_dir', type=str,
                       default='experiments/distilled_students',
                       help='Directory for student models')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device == 'cuda' and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💾 GPU Total Memory: {gpu_mem:.2f} GiB")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Get top 20 models info
    models_info = get_top_20_models_info(args.experiments_root, args.students_dir)
    
    # Load all to GPU
    all_models = load_all_models_to_gpu(models_info, device)
    
    if len(all_models) == 0:
        print("❌ No models loaded! Check your paths.")
        return
    
    # Process CIFAR-10
    process_dataset(
        'cifar10',
        all_models,
        device,
        num_images=args.num_images,
        output_dir=args.cifar10_dir
    )
    
    # Process ImageNet
    process_dataset(
        'imagenet',
        all_models,
        device,
        num_images=args.num_images,
        output_dir=args.imagenet_dir
    )
    
    print("\n" + "="*80)
    print("COLORIZATION COMPLETE")
    print("="*80)
    print(f"\n✅ CIFAR-10 results: {args.cifar10_dir}")
    print(f"✅ ImageNet results: {args.imagenet_dir}")
    print(f"\n📊 Processed with {len(all_models)} top models")


if __name__ == '__main__':
    main()
