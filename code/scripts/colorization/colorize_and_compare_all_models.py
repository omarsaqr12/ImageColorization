#!/usr/bin/env python3
"""
Colorize and Compare All Models
Colorizes 1000 images using all available models and creates comparison visualizations.
"""

import os
import sys
import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import glob
import json

# Suppress LAB to RGB conversion warnings (common when colorizing)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Conversion from CIE-LAB.*')

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17

# Import student models
sys.path.append('src/training')
from student import LightweightStudent, MobileNetStyleStudent


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
    device = lab_tensor.device
    lab_np = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()
    rgb_np = np.zeros_like(lab_np)
    for i in range(lab_np.shape[0]):
        rgb_np[i] = color.lab2rgb(lab_np[i])
        rgb_np[i] = np.clip(rgb_np[i], 0, 1)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
    return rgb_tensor.to(device)


def load_teacher_models(device, models_dir='models'):
    """Load all teacher models"""
    models = {}
    
    print("="*80)
    print("Loading Teacher Models")
    print("="*80)
    
    # Load pretrained models
    try:
        models['pretrained_eccv16'] = {
            'model': eccv16(pretrained=True).eval().to(device),
            'type': 'eccv16',
            'use_ila': False,
            'use_pretrained_preprocessing': True
        }
        models['pretrained_siggraph17'] = {
            'model': siggraph17(pretrained=True).eval().to(device),
            'type': 'siggraph17',
            'use_ila': False,
            'use_pretrained_preprocessing': True
        }
        print("✅ Pre-trained models loaded")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained models: {e}")
    
    # Find all teacher model files
    teacher_patterns = [
        'eccv16_best_model.pth',
        'eccv16_ila_best_model.pth',
        'eccv16*perceptual*best_model.pth',
        'eccv16*gan*best_model.pth',
        'siggraph17_best_model.pth',
        'siggraph17*perceptual*best_model.pth',
        'siggraph17*gan*best_model.pth'
    ]
    
    model_files = []
    for pattern in teacher_patterns:
        model_files.extend(glob.glob(os.path.join(models_dir, pattern)))
    
    # Load each teacher model
    for filepath in sorted(set(model_files)):
        if not os.path.isfile(filepath):
            continue
        
        # Skip discriminators
        if 'discriminator' in filepath:
            continue
        
        try:
            filename = os.path.basename(filepath)
            is_eccv16 = 'eccv16' in filename
            use_ila = 'ila' in filename and is_eccv16
            
            if is_eccv16:
                model = eccv16(pretrained=False, use_ila=use_ila).eval().to(device)
            else:
                model = siggraph17(pretrained=False).eval().to(device)
            
            model.load_state_dict(torch.load(filepath, map_location=device))
            
            # Create clean key name
            key = filename.replace('_best_model.pth', '').replace('.pth', '')
            models[key] = {
                'model': model,
                'type': 'eccv16' if is_eccv16 else 'siggraph17',
                'use_ila': use_ila,
                'use_pretrained_preprocessing': False  # Trained models use LAB preprocessing
            }
            print(f"✅ Loaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
    
    print(f"\n📊 Total teacher models loaded: {len(models)}")
    return models


def load_student_models(device, models_dir='models'):
    """Load all student models"""
    models = {}
    
    print("\n" + "="*80)
    print("Loading Student Models")
    print("="*80)
    
    # Find all student model files
    student_patterns = [
        os.path.join(models_dir, 'student_*_best_model*.pth'),
        os.path.join(models_dir, 'student_*.pth')
    ]
    
    model_files = []
    for pattern in student_patterns:
        model_files.extend(glob.glob(pattern))
    
    # Load each student model
    for filepath in sorted(set(model_files)):
        if not os.path.isfile(filepath):
            continue
        
        filename = os.path.basename(filepath)
        
        # Skip if it's not a student model or is a checkpoint
        if 'epoch' in filename or 'history' in filename:
            continue
        
        try:
            # Determine student type and reduction from filename
            is_mobilenet = 'mobilenet' in filename.lower() or 'mob' in filename.lower()
            
            # Extract reduction from filename patterns
            reduction = 2  # default
            if '_4x' in filename or '_4_' in filename or filename.endswith('_4.pth'):
                reduction = 4
            elif '_8x' in filename or '_8_' in filename or filename.endswith('_8.pth'):
                reduction = 8
            elif '4e' in filename or '4x' in filename:
                reduction = 4
            elif '8e' in filename or '8x' in filename:
                reduction = 8
            
            # Create student model
            if is_mobilenet:
                student = MobileNetStyleStudent(width_multiplier=1.0/reduction)
            else:
                student = LightweightStudent(channel_reduction=reduction)
            
            # Load weights
            state_dict = torch.load(filepath, map_location=device)
            student.load_state_dict(state_dict)
            student = student.eval().to(device)
            
            # Create clean key name
            key = filename.replace('_best_model.pth', '').replace('.pth', '')
            # Remove trailing numbers that might be reduction indicators
            if key.endswith('_2') or key.endswith('_4') or key.endswith('_8'):
                key = key[:-2]
            
            models[key] = {
                'model': student,
                'type': 'student',
                'architecture': 'mobilenet' if is_mobilenet else 'lightweight',
                'reduction': reduction,
                'use_pretrained_preprocessing': False  # Students use LAB preprocessing
            }
            print(f"✅ Loaded: {filename} (reduction: {reduction}x)")
        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Total student models loaded: {len(models)}")
    return models


def colorize_with_model(rgb_image, model_info, device):
    """Colorize an image using a specific model"""
    model = model_info['model']
    use_pretrained_preprocessing = model_info.get('use_pretrained_preprocessing', False)
    
    if use_pretrained_preprocessing:
        # Use original preprocessing for pre-trained models
        from colorizers.util import preprocess_img, postprocess_tens
        
        if isinstance(rgb_image, torch.Tensor):
            rgb_image_pil = transforms.ToPILImage()(rgb_image.squeeze(0))
        else:
            rgb_image_pil = rgb_image
        
        if rgb_image_pil.mode != 'RGB':
            rgb_image_pil = rgb_image_pil.convert('RGB')
        
        (tens_l_orig, tens_l_rs) = preprocess_img(np.array(rgb_image_pil), HW=(256, 256))
        
        if device != 'cpu':
            tens_l_rs = tens_l_rs.to(device)
        
        with torch.no_grad():
            colorized_ab = model(tens_l_rs).cpu()
        
        colorized_image = postprocess_tens(tens_l_orig, colorized_ab)
        return colorized_image
    else:
        # Use LAB preprocessing for trained models
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.to(device)
        else:
            rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(0).to(device)
        
        # Convert RGB to LAB
        lab_image = rgb_to_lab(rgb_image, device)
        l_channel = lab_image[:, 0:1, :, :]
        
        # Colorize
        with torch.no_grad():
            predicted_ab = model(l_channel)
        
        # Reconstruct RGB
        predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
        predicted_rgb = lab_to_rgb(predicted_lab)
        
        # Convert to numpy
        predicted_rgb_np = predicted_rgb[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(predicted_rgb_np, 0, 1)


def create_comparison_image(original_rgb, grayscale, colorized_results, output_path, image_idx):
    """Create a comparison image showing original, grayscale, and all colorized versions"""
    num_models = len(colorized_results)
    
    # Calculate grid dimensions
    cols = min(4, num_models + 2)  # Original + Grayscale + models
    rows = (num_models + 2 + cols - 1) // cols  # Ceiling division
    
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_rgb)
    ax.set_title('Original', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Grayscale image
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(grayscale, cmap='gray')
    ax.set_title('Grayscale Input', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Colorized versions
    for idx, (model_name, colorized_img) in enumerate(colorized_results.items()):
        row = (idx + 2) // cols
        col = (idx + 2) % cols
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(colorized_img)
        # Truncate long model names
        title = model_name if len(model_name) <= 30 else model_name[:27] + '...'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    total_plots = rows * cols
    used_plots = 2 + num_models
    for idx in range(used_plots, total_plots):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    plt.suptitle(f'Image {image_idx + 1} - Model Comparison', fontsize=12, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Colorize and compare images using all models')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to colorize (default: 1000)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda, auto-detects if not specified)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--output_dir', type=str, default='results/colorization_comparisons', 
                       help='Output directory for comparison images')
    parser.add_argument('--save_every', type=int, default=1, 
                       help='Save comparison image every N images (default: 1, saves all images)')
    parser.add_argument('--skip_saving', action='store_true', 
                       help='Skip saving images (only process them)')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all models
    teacher_models = load_teacher_models(device, args.models_dir)
    student_models = load_student_models(device, args.models_dir)
    
    all_models = {**teacher_models, **student_models}
    
    if len(all_models) == 0:
        print("❌ No models found! Please check models directory.")
        return
    
    print(f"\n📊 Total models to use: {len(all_models)}")
    print("Models:", list(all_models.keys()))
    
    # Prepare test data
    print(f"\n📁 Preparing CIFAR-10 test data ({args.num_images} images)...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./cifar-10-python', train=False, download=True, transform=transform
    )
    if args.num_images < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:args.num_images]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"✅ Test samples: {len(test_dataset)}\n")
    
    # Process images
    print("="*80)
    print("COLORIZING IMAGES")
    print("="*80)
    
    results_summary = {
        'num_images': args.num_images,
        'num_models': len(all_models),
        'models': list(all_models.keys()),
        'saved_images': []
    }
    
    for image_idx, (rgb_image, _) in enumerate(tqdm(test_loader, desc="Processing images")):
        if image_idx >= args.num_images:
            break
        
        rgb_image = rgb_image.to(device)
        original_rgb = rgb_image[0].permute(1, 2, 0).cpu().numpy()
        original_rgb = np.clip(original_rgb, 0, 1)
        
        # Create grayscale version
        grayscale = np.mean(original_rgb, axis=2)
        
        # Colorize with all models
        colorized_results = {}
        
        for model_name, model_info in all_models.items():
            try:
                colorized_img = colorize_with_model(rgb_image, model_info, device)
                colorized_results[model_name] = colorized_img
            except Exception as e:
                print(f"\n⚠️  Error colorizing with {model_name} for image {image_idx + 1}: {e}")
                continue
        
        # Save comparison image (save all by default unless skip_saving is set)
        should_save = not args.skip_saving and ((image_idx + 1) % args.save_every == 0)
        
        if should_save and len(colorized_results) > 0:
            output_path = os.path.join(args.output_dir, f'comparison_{image_idx + 1:05d}.png')
            create_comparison_image(original_rgb, grayscale, colorized_results, output_path, image_idx)
            results_summary['saved_images'].append(image_idx + 1)
        
        # Print progress every 100 images
        if (image_idx + 1) % 100 == 0:
            print(f"\n✅ Processed {image_idx + 1}/{args.num_images} images")
            if should_save:
                print(f"   Saved {len(results_summary['saved_images'])} comparison images so far")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
    print(f"✅ Processed {args.num_images} images")
    print(f"✅ Saved {len(results_summary['saved_images'])} comparison images to: {args.output_dir}")
    print(f"✅ Summary saved to: {summary_path}")
    print("\n🎉 Colorization and comparison completed!")


if __name__ == '__main__':
    main()

