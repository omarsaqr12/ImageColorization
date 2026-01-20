#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates ALL models (teachers and students) with full metrics:
- PSNR, SSIM, LPIPS
- Inference speed
- Number of parameters
- Model size
"""

import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import json
import glob
import cv2
from PIL import Image

# Try to import tabulate for nice tables
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

# Try to import FID and KID
FID_AVAILABLE = False
KID_AVAILABLE = False
USE_PHOTOSYNTHESIS = False
try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    try:
        from photosynthesis_metrics import FID, KID
        FID_AVAILABLE = True
        KID_AVAILABLE = True
        USE_PHOTOSYNTHESIS = True
    except ImportError:
        print("Warning: FID/KID not available. Install with: pip install pytorch-fid or pip install photosynthesis-metrics")

# Try to import FSIM
FSIM_AVAILABLE = False
fsim_func = None
try:
    import sewar
    # Check what's available in sewar.full_ref
    try:
        from sewar import full_ref
        # List available functions for debugging
        available_funcs = [attr for attr in dir(full_ref) if not attr.startswith('_')]
        
        # Try common FSIM function names
        if hasattr(full_ref, 'fsim'):
            fsim_func = full_ref.fsim
            FSIM_AVAILABLE = True
        elif hasattr(full_ref, 'fsimc'):
            fsim_func = full_ref.fsimc
            FSIM_AVAILABLE = True
        elif hasattr(full_ref, 'FSIM'):
            fsim_func = full_ref.FSIM
            FSIM_AVAILABLE = True
        else:
            # FSIM might not be available in this version of sewar
            print(f"Warning: FSIM function not found in sewar.full_ref. Available functions: {', '.join(available_funcs[:10])}")
    except ImportError as e:
        print(f"Warning: Could not import sewar.full_ref: {e}")
    
    if FSIM_AVAILABLE:
        print("✅ FSIM metric available")
    else:
        print("⚠️  FSIM not available - continuing without FSIM metric")
except (ImportError, ModuleNotFoundError) as e:
    FSIM_AVAILABLE = False
    print(f"Warning: sewar package not available. Install with: pip install sewar")

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


def calculate_delta_e2000(original_lab, colorized_lab):
    """Calculate Delta E2000 color difference"""
    # Delta E2000 is complex, using simplified Delta E for now
    # Full implementation would require colorio or similar library
    delta_e = np.sqrt(np.sum((original_lab - colorized_lab)**2, axis=2))
    return np.mean(delta_e)


def calculate_colorfulness_index(image):
    """Calculate colorfulness index using Hasler and Süsstrunk method"""
    # Convert to RG and YB opponent color space
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    
    # RG and YB channels
    rg = r - g
    yb = 0.5 * (r + g) - b
    
    # Calculate mean and std
    rg_mean = np.mean(rg)
    rg_std = np.std(rg)
    yb_mean = np.mean(yb)
    yb_std = np.std(yb)
    
    # Colorfulness formula
    colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
    return colorfulness


def calculate_color_histogram_kl_divergence(original, colorized, bins=32):
    """Calculate KL divergence between color histograms"""
    # Flatten images
    orig_flat = original.reshape(-1, 3)
    colorized_flat = colorized.reshape(-1, 3)
    
    # Create histograms for each channel
    kl_divs = []
    for channel in range(3):
        orig_hist, _ = np.histogram(orig_flat[:, channel], bins=bins, range=(0, 1), density=True)
        colorized_hist, _ = np.histogram(colorized_flat[:, channel], bins=bins, range=(0, 1), density=True)
        
        # Normalize to probabilities
        orig_hist = orig_hist + 1e-10  # Avoid zeros
        colorized_hist = colorized_hist + 1e-10
        orig_hist = orig_hist / orig_hist.sum()
        colorized_hist = colorized_hist / colorized_hist.sum()
        
        # Calculate KL divergence
        kl_div = entropy(orig_hist, colorized_hist)
        kl_divs.append(kl_div)
    
    return np.mean(kl_divs)


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    return param_size / (1024 * 1024)


def count_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


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
            'use_ila': False
        }
        models['pretrained_siggraph17'] = {
            'model': siggraph17(pretrained=True).eval().to(device),
            'type': 'siggraph17',
            'use_ila': False
        }
        print("✅ Pre-trained models loaded")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained models: {e}")
    
    # Find all teacher model files - use more comprehensive patterns
    teacher_patterns = [
        os.path.join(models_dir, 'eccv16*_best_model.pth'),
        os.path.join(models_dir, 'eccv16*.pth'),
        os.path.join(models_dir, 'siggraph17*_best_model.pth'),
        os.path.join(models_dir, 'siggraph17*.pth')
    ]
    
    model_files = []
    for pattern in teacher_patterns:
        found = glob.glob(pattern)
        model_files.extend(found)
    
    # Remove duplicates and filter out non-teacher files
    model_files = sorted(set(model_files))
    # Filter out student models, discriminators, and other non-teacher files
    original_count = len(model_files)
    
    # Track what's being filtered and why
    filtered_out = []
    for f in model_files:
        filename = os.path.basename(f).lower()
        reason = None
        if 'student' in filename:
            reason = "student model"
        elif 'discriminator' in filename:
            reason = "discriminator"
        elif 'epoch' in filename:
            reason = "epoch checkpoint"
        elif 'history' in filename:
            reason = "history file"
        
        if reason:
            filtered_out.append((os.path.basename(f), reason))
    
    model_files = [f for f in model_files if 'student' not in os.path.basename(f).lower() 
                   and 'discriminator' not in os.path.basename(f).lower()
                   and 'epoch' not in os.path.basename(f).lower()
                   and 'history' not in os.path.basename(f).lower()]
    
    print(f"📁 Found {len(model_files)} teacher model files (filtered {len(filtered_out)} files from {original_count} total)")
    if filtered_out:
        print("   Filtered out:")
        for filename, reason in filtered_out:
            print(f"     - {filename} ({reason})")
    if len(model_files) > 0:
        print("   Files to load:")
        for f in model_files:
            print(f"     - {os.path.basename(f)}")
    
    # Load each teacher model
    for filepath in model_files:
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
                'use_ila': use_ila
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
        found = glob.glob(pattern)
        model_files.extend(found)
    
    # Remove duplicates
    model_files = sorted(set(model_files))
    
    print(f"📁 Found {len(model_files)} student model files")
    if len(model_files) > 0:
        print("   Files found:")
        for f in model_files:
            print(f"     - {os.path.basename(f)}")
    
    # Load each student model
    for filepath in model_files:
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
            
            # Create clean key name - preserve reduction to avoid collisions
            # Remove common suffixes first
            key = filename.replace('_best_model.pth', '').replace('.pth', '')
            
            # Check if reduction is already in the key (as _N or _Nx)
            has_reduction_in_name = (f'_{reduction}' in key and key.endswith(f'_{reduction}')) or \
                                   (f'_{reduction}x' in key) or \
                                   (f'_{reduction}_' in key)
            
            # If reduction is not clearly in the key, add it to distinguish different reductions
            if not has_reduction_in_name:
                key = f"{key}_{reduction}x"
            else:
                # Normalize: ensure it ends with _Nx format
                if key.endswith(f'_{reduction}'):
                    key = key.replace(f'_{reduction}', f'_{reduction}x')
                elif f'_{reduction}_' in key:
                    # Reduction is in the middle, keep as is but ensure format
                    pass
            
            models[key] = {
                'model': student,
                'type': 'student',
                'architecture': 'mobilenet' if is_mobilenet else 'lightweight',
                'reduction': reduction
            }
            print(f"✅ Loaded: {filename} → key: {key} (reduction: {reduction}x)")
        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Total student models loaded: {len(models)}")
    return models


def evaluate_model(model_info, test_loader, device, num_samples=5000, lpips_model=None):
    """Evaluate a single model"""
    model = model_info['model']
    model_type = model_info['type']
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'delta_e2000': [],
        'lpips': [],
        'colorfulness': [],
        'kl_divergence': [],
        'fsim': [],
        'inference_times': []
    }
    
    # Store images for FID/KID computation
    original_images_list = []
    predicted_images_list = []
    
    count = 0
    
    # Warmup
    with torch.no_grad():
        dummy_l = torch.randn(1, 1, 32, 32).to(device)
        try:
            if model_type == 'student':
                _ = model(dummy_l)
            else:
                _ = model(dummy_l)
        except:
            pass
    
    for batch_idx, (rgb_images, _) in enumerate(test_loader):
        if count >= num_samples:
            break
        
        rgb_images = rgb_images.to(device)
        lab_images = rgb_to_lab(rgb_images, device)
        l_channels = lab_images[:, 0:1, :, :]
        
        # Process each image in batch individually for accurate timing
        for i in range(rgb_images.size(0)):
            if count >= num_samples:
                break
            
            l_single = l_channels[i:i+1]
            
            # Measure inference time
            torch.cuda.synchronize() if device != 'cpu' else None
            start_time = time.time()
            
            with torch.no_grad():
                if model_type == 'student':
                    predicted_ab_single = model(l_single)
                else:
                    # Teacher model
                    predicted_ab_single = model(l_single)
            
            torch.cuda.synchronize() if device != 'cpu' else None
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Reconstruct RGB
            predicted_lab = torch.cat([l_single, predicted_ab_single], dim=1)
            predicted_rgb_single = lab_to_rgb(predicted_lab)
            
            # Convert to numpy for metrics
            orig = rgb_images[i].permute(1, 2, 0).cpu().numpy()
            pred = predicted_rgb_single[0].permute(1, 2, 0).cpu().numpy()
            
            # PSNR
            psnr_value = psnr(orig, pred, data_range=1.0)
            metrics['psnr'].append(psnr_value)
            
            # SSIM
            ssim_value = ssim(orig, pred, data_range=1.0, channel_axis=2)
            metrics['ssim'].append(ssim_value)
            
            # Delta E2000 (color difference in LAB space)
            orig_lab = color.rgb2lab(orig)
            pred_lab = color.rgb2lab(pred)
            delta_e = calculate_delta_e2000(orig_lab, pred_lab)
            metrics['delta_e2000'].append(delta_e)
            
            # LPIPS
            if lpips_model is not None:
                orig_t = rgb_images[i:i+1].to(device)
                pred_t = predicted_rgb_single.to(device)
                lpips_value = lpips_model(orig_t, pred_t).item()
                metrics['lpips'].append(lpips_value)
            
            # Colorfulness
            colorfulness = calculate_colorfulness_index(pred)
            metrics['colorfulness'].append(colorfulness)
            
            # KL Divergence (color histogram)
            kl_div = calculate_color_histogram_kl_divergence(orig, pred)
            metrics['kl_divergence'].append(kl_div)
            
            # FSIM (Feature Similarity Index)
            if FSIM_AVAILABLE and fsim_func is not None:
                try:
                    # FSIM expects images in [0, 255] range
                    orig_uint8 = (np.clip(orig, 0, 1) * 255).astype(np.uint8)
                    pred_uint8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
                    # Convert to grayscale for FSIM
                    orig_gray = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2GRAY) if len(orig_uint8.shape) == 3 else orig_uint8
                    pred_gray = cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2GRAY) if len(pred_uint8.shape) == 3 else pred_uint8
                    fsim_value = fsim_func(orig_gray, pred_gray)
                    metrics['fsim'].append(fsim_value)
                except Exception as e:
                    # If FSIM fails, skip it for this image
                    pass
            
            # Store images for FID/KID (convert to PIL Image format)
            if FID_AVAILABLE or KID_AVAILABLE:
                orig_pil = Image.fromarray((np.clip(orig, 0, 1) * 255).astype(np.uint8))
                pred_pil = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8))
                original_images_list.append(orig_pil)
                predicted_images_list.append(pred_pil)
            
            metrics['inference_times'].append(inference_time)
            count += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"   Processed {count}/{num_samples} samples")
    
    # Calculate model statistics
    num_params = count_parameters(model)
    model_size_mb = calculate_model_size(model)
    
    # Calculate FID and KID if available
    fid_score = None
    kid_score = None
    if (FID_AVAILABLE or KID_AVAILABLE) and len(original_images_list) > 0:
        try:
            import tempfile
            import os
            import shutil
            
            # Create temporary directories for FID/KID computation
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_dir = os.path.join(tmpdir, 'original')
                pred_dir = os.path.join(tmpdir, 'predicted')
                os.makedirs(orig_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)
                
                # Save images
                for i, img in enumerate(original_images_list):
                    img.save(os.path.join(orig_dir, f'{i:05d}.png'))
                for i, img in enumerate(predicted_images_list):
                    img.save(os.path.join(pred_dir, f'{i:05d}.png'))
                
                # Calculate FID
                if FID_AVAILABLE:
                    try:
                        if USE_PHOTOSYNTHESIS:
                            fid_metric = FID()
                            fid_score = fid_metric.compute(orig_dir, pred_dir)
                        else:
                            # Use pytorch_fid module
                            from pytorch_fid import fid_score as fid_module
                            fid_score = fid_module.calculate_fid_given_paths([orig_dir, pred_dir], 
                                                                              batch_size=50, 
                                                                              device=device, 
                                                                              dims=2048)
                    except Exception as e:
                        print(f"Warning: FID calculation failed: {e}")
                        fid_score = None
                
                # Calculate KID
                if KID_AVAILABLE and USE_PHOTOSYNTHESIS:
                    try:
                        kid_metric = KID()
                        kid_score = kid_metric.compute(orig_dir, pred_dir)
                    except Exception as e:
                        print(f"Warning: KID calculation failed: {e}")
                        kid_score = None
        except Exception as e:
            print(f"Warning: FID/KID computation failed: {e}")
    
    # Calculate summary statistics
    results = {
        'psnr': {
            'mean': float(np.mean(metrics['psnr'])),
            'std': float(np.std(metrics['psnr'])),
            'min': float(np.min(metrics['psnr'])),
            'max': float(np.max(metrics['psnr']))
        },
        'ssim': {
            'mean': float(np.mean(metrics['ssim'])),
            'std': float(np.std(metrics['ssim'])),
            'min': float(np.min(metrics['ssim'])),
            'max': float(np.max(metrics['ssim']))
        },
        'delta_e2000': {
            'mean': float(np.mean(metrics['delta_e2000'])),
            'std': float(np.std(metrics['delta_e2000'])),
            'min': float(np.min(metrics['delta_e2000'])),
            'max': float(np.max(metrics['delta_e2000']))
        },
        'colorfulness': {
            'mean': float(np.mean(metrics['colorfulness'])),
            'std': float(np.std(metrics['colorfulness'])),
            'min': float(np.min(metrics['colorfulness'])),
            'max': float(np.max(metrics['colorfulness']))
        },
        'kl_divergence': {
            'mean': float(np.mean(metrics['kl_divergence'])),
            'std': float(np.std(metrics['kl_divergence'])),
            'min': float(np.min(metrics['kl_divergence'])),
            'max': float(np.max(metrics['kl_divergence']))
        },
        'inference_time_ms': {
            'mean': float(np.mean(metrics['inference_times'])),
            'std': float(np.std(metrics['inference_times'])),
            'min': float(np.min(metrics['inference_times'])),
            'max': float(np.max(metrics['inference_times']))
        },
        'num_parameters': int(num_params),
        'num_parameters_millions': float(num_params / 1e6),
        'model_size_mb': float(model_size_mb),
        'num_samples': count
    }
    
    if metrics['lpips']:
        results['lpips'] = {
            'mean': float(np.mean(metrics['lpips'])),
            'std': float(np.std(metrics['lpips'])),
            'min': float(np.min(metrics['lpips'])),
            'max': float(np.max(metrics['lpips']))
        }
    
    if metrics['fsim']:
        results['fsim'] = {
            'mean': float(np.mean(metrics['fsim'])),
            'std': float(np.std(metrics['fsim'])),
            'min': float(np.min(metrics['fsim'])),
            'max': float(np.max(metrics['fsim']))
        }
    
    if fid_score is not None:
        results['fid'] = float(fid_score)
    
    if kid_score is not None:
        results['kid'] = float(kid_score)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of all models')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to evaluate (default: 5000)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda, auto-detects if not specified)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--output_file', type=str, default='comprehensive_evaluation_results.json', 
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Initialize LPIPS if available
    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
        print("✅ LPIPS model loaded")
    else:
        print("⚠️  LPIPS not available - skipping LPIPS metrics")
    
    # Load all models
    teacher_models = load_teacher_models(device, args.models_dir)
    student_models = load_student_models(device, args.models_dir)
    
    all_models = {**teacher_models, **student_models}
    
    if len(all_models) == 0:
        print("❌ No models found! Please check models directory.")
        return
    
    print(f"\n📊 Total models loaded: {len(all_models)}")
    print("Models to evaluate:")
    for key in sorted(all_models.keys()):
        model_type = all_models[key].get('type', 'unknown')
        print(f"  - {key} ({model_type})")
    
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
    print(f"Total models to evaluate: {len(all_models)}\n")
    results = {}
    failed_models = []
    
    for idx, model_key in enumerate(sorted(all_models.keys()), 1):
        print(f"\n[{idx}/{len(all_models)}] 🔍 Evaluating: {model_key}")
        try:
            model_results = evaluate_model(
                all_models[model_key], test_loader, device, args.num_samples, lpips_model
            )
            results[model_key] = model_results
            print(f"   ✅ PSNR: {model_results['psnr']['mean']:.4f}, "
                  f"SSIM: {model_results['ssim']['mean']:.4f}, "
                  f"Params: {model_results['num_parameters_millions']:.2f}M, "
                  f"Time: {model_results['inference_time_ms']['mean']:.2f}ms")
        except Exception as e:
            print(f"   ❌ Error evaluating {model_key}: {e}")
            failed_models.append((model_key, str(e)))
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"✅ Successfully evaluated: {len(results)}/{len(all_models)} models")
    if failed_models:
        print(f"❌ Failed to evaluate: {len(failed_models)} models")
        for model_key, error in failed_models:
            print(f"   - {model_key}: {error[:100]}...")
    print("="*80)
    
    # Generate comparison tables
    print("\n" + "="*80)
    print("EVALUATION RESULTS - COMPREHENSIVE COMPARISON")
    print("="*80)
    
    # Prepare table data - Main metrics table
    print("\n📊 MAIN METRICS TABLE")
    print("="*80)
    table_data = []
    for model_key in sorted(results.keys()):
        r = results[model_key]
        row = [
            model_key,
            f"{r['psnr']['mean']:.4f} ± {r['psnr']['std']:.4f}",
            f"{r['ssim']['mean']:.4f} ± {r['ssim']['std']:.4f}",
            f"{r['delta_e2000']['mean']:.4f} ± {r['delta_e2000']['std']:.4f}",
            f"{r['lpips']['mean']:.4f} ± {r['lpips']['std']:.4f}" if 'lpips' in r else "N/A",
            f"{r['fsim']['mean']:.4f} ± {r['fsim']['std']:.4f}" if 'fsim' in r else "N/A",
            f"{r['fid']:.2f}" if 'fid' in r else "N/A",
            f"{r['kid']:.4f}" if 'kid' in r else "N/A",
            f"{r['inference_time_ms']['mean']:.2f} ± {r['inference_time_ms']['std']:.2f}",
            f"{r['num_parameters_millions']:.2f}M",
            f"{r['model_size_mb']:.2f} MB"
        ]
        table_data.append(row)
    
    headers = ['Model', 'PSNR (↑)', 'SSIM (↑)', 'ΔE2000 (↓)', 'LPIPS (↓)', 'FSIM (↑)', 'FID (↓)', 'KID (↓)', 'Inference (ms)', 'Params', 'Size (MB)']
    
    if HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.4f'))
    else:
        # Simple table format
        col_widths = [40, 18, 18, 18, 18, 12, 12]
        print(' | '.join(h.ljust(w) for h, w in zip(headers, col_widths)))
        print('-' * sum(col_widths) + '-' * (len(headers) * 3))
        for row in table_data:
            print(' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    
    # Save results
    output_path = args.output_file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Additional metrics table
    print("\n📊 PERCEPTUAL & COLOR METRICS TABLE")
    print("="*80)
    table_data2 = []
    for model_key in sorted(results.keys()):
        r = results[model_key]
        row = [
            model_key,
            f"{r['colorfulness']['mean']:.2f} ± {r['colorfulness']['std']:.2f}",
            f"{r['kl_divergence']['mean']:.4f} ± {r['kl_divergence']['std']:.4f}",
            f"{r['lpips']['mean']:.4f} ± {r['lpips']['std']:.4f}" if 'lpips' in r else "N/A",
            f"{r['fsim']['mean']:.4f} ± {r['fsim']['std']:.4f}" if 'fsim' in r else "N/A",
            f"{r['fid']:.2f}" if 'fid' in r else "N/A",
            f"{r['kid']:.4f}" if 'kid' in r else "N/A"
        ]
        table_data2.append(row)
    
    headers2 = ['Model', 'Colorfulness (↑)', 'KL Divergence (↓)', 'LPIPS (↓)', 'FSIM (↑)', 'FID (↓)', 'KID (↓)']
    
    if HAS_TABULATE:
        print(tabulate(table_data2, headers=headers2, tablefmt='grid', floatfmt='.4f'))
    else:
        col_widths = [40, 20, 20, 18]
        print(' | '.join(h.ljust(w) for h, w in zip(headers2, col_widths)))
        print('-' * sum(col_widths) + '-' * (len(headers2) * 3))
        for row in table_data2:
            print(' | '.join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Find best models
    if results:
        best_psnr = max(results.items(), key=lambda x: x[1]['psnr']['mean'])
        best_ssim = max(results.items(), key=lambda x: x[1]['ssim']['mean'])
        best_delta_e = min(results.items(), key=lambda x: x[1]['delta_e2000']['mean'])
        best_colorfulness = max(results.items(), key=lambda x: x[1]['colorfulness']['mean'])
        best_kl = min(results.items(), key=lambda x: x[1]['kl_divergence']['mean'])
        fastest = min(results.items(), key=lambda x: x[1]['inference_time_ms']['mean'])
        smallest = min(results.items(), key=lambda x: x[1]['num_parameters'])
        
        print(f"\n🏆 Best PSNR: {best_psnr[0]} ({best_psnr[1]['psnr']['mean']:.4f} dB)")
        print(f"🏆 Best SSIM: {best_ssim[0]} ({best_ssim[1]['ssim']['mean']:.4f})")
        print(f"🏆 Best ΔE2000: {best_delta_e[0]} ({best_delta_e[1]['delta_e2000']['mean']:.4f})")
        print(f"🏆 Best Colorfulness: {best_colorfulness[0]} ({best_colorfulness[1]['colorfulness']['mean']:.2f})")
        print(f"🏆 Best KL Divergence: {best_kl[0]} ({best_kl[1]['kl_divergence']['mean']:.4f})")
        if LPIPS_AVAILABLE:
            best_lpips = min([(k, v) for k, v in results.items() if 'lpips' in v], 
                           key=lambda x: x[1]['lpips']['mean'])
            print(f"🏆 Best LPIPS: {best_lpips[0]} ({best_lpips[1]['lpips']['mean']:.4f})")
        if FSIM_AVAILABLE:
            best_fsim = max([(k, v) for k, v in results.items() if 'fsim' in v], 
                           key=lambda x: x[1]['fsim']['mean'])
            print(f"🏆 Best FSIM: {best_fsim[0]} ({best_fsim[1]['fsim']['mean']:.4f})")
        if FID_AVAILABLE:
            best_fid = min([(k, v) for k, v in results.items() if 'fid' in v], 
                          key=lambda x: x[1]['fid'])
            print(f"🏆 Best FID: {best_fid[0]} ({best_fid[1]['fid']:.2f})")
        if KID_AVAILABLE:
            best_kid = min([(k, v) for k, v in results.items() if 'kid' in v], 
                          key=lambda x: x[1]['kid'])
            print(f"🏆 Best KID: {best_kid[0]} ({best_kid[1]['kid']:.4f})")
        print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['inference_time_ms']['mean']:.2f} ms)")
        print(f"📦 Smallest: {smallest[0]} ({smallest[1]['num_parameters_millions']:.2f}M params, {smallest[1]['model_size_mb']:.2f} MB)")
    
    print("\n🎉 Evaluation completed!")


if __name__ == '__main__':
    main()

