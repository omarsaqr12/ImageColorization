#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Colorization Models on CIFAR-10
Evaluates ECCV 2016 and SIGGRAPH 2017 models using multiple metrics
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import json
from collections import defaultdict

# Add colorization directory to path
sys.path.append('colorization')

try:
    from colorizers import *
except ImportError:
    print("Error: Could not import colorizers. Make sure the colorization directory is present.")
    sys.exit(1)

# Additional imports for advanced metrics
try:
    from pytorch_fid import fid_score
    import lpips
except ImportError:
    print("Warning: FID and LPIPS not available. Install with: pip install pytorch-fid lpips")
    fid_score = None
    lpips = None

class ColorizationEvaluator:
    def __init__(self, device='cpu', use_ila=False, ila_reduction=4, ila_use_dw_conv=True):
        """
        Initialize colorization evaluator.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
            use_ila: If True, use ILA blocks in ECCV16 model. Default: False
            ila_reduction: Channel reduction factor for ILA. Default: 4
            ila_use_dw_conv: Whether ILA uses depthwise convolution. Default: True
        """
        self.device = device
        self.results = defaultdict(dict)
        self.use_ila = use_ila
        
        # Load models
        print("Loading colorization models...")
        
        # ECCV16 with optional ILA - check for trained model first
        model_path = 'eccv16_ila_best_model.pth' if use_ila else 'eccv16_best_model.pth'
        
        if os.path.exists(model_path):
            print(f"✓ Found trained model: {model_path}")
            print(f"  Loading trained ECCV16 model ({'with ILA' if use_ila else 'baseline'})...")
            self.colorizer_eccv16 = eccv16(
                pretrained=False, 
                use_ila=use_ila, 
                ila_reduction=ila_reduction, 
                ila_use_dw_conv=ila_use_dw_conv
            ).eval().to(device)
            self.colorizer_eccv16.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  ✓ Loaded trained model from: {model_path}")
        else:
            print(f"  Trained model not found: {model_path}")
            if use_ila:
                print(f"  ⚠ Using random initialization (ILA models need training)")
            else:
                print(f"  → Using pretrained weights from URL")
            self.colorizer_eccv16 = eccv16(
                pretrained=not use_ila,  # Only use pretrained if not ILA
                use_ila=use_ila, 
                ila_reduction=ila_reduction, 
                ila_use_dw_conv=ila_use_dw_conv
            ).eval().to(device)
        
        # SIGGRAPH17 - always use pretrained (or check for trained if exists)
        if os.path.exists('siggraph17_best_model.pth'):
            print(f"✓ Found trained SIGGRAPH17 model")
            self.colorizer_siggraph17 = siggraph17(pretrained=False).eval().to(device)
            self.colorizer_siggraph17.load_state_dict(torch.load('siggraph17_best_model.pth', map_location=device))
            print(f"  ✓ Loaded trained SIGGRAPH17 model")
        else:
            self.colorizer_siggraph17 = siggraph17(pretrained=True).eval().to(device)
            print(f"  → Using pretrained SIGGRAPH17 weights from URL")
        
        if use_ila:
            print(f"ILA enabled in ECCV16: reduction={ila_reduction}, use_dw_conv={ila_use_dw_conv}")
        else:
            print("ILA disabled (baseline ECCV16)")
        
        # Initialize LPIPS if available
        if lpips is not None:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        print("Models loaded successfully!")
    
    def download_cifar10(self, data_dir='./cifar10_data'):
        """Download and prepare CIFAR-10 dataset"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        print("Downloading CIFAR-10 dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        
        return trainset, testset
    
    def rgb_to_lab(self, rgb_image):
        """Convert RGB image to LAB color space"""
        # Convert to numpy
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.numpy()
        
        # Ensure RGB format
        if rgb_image.shape[0] == 3:  # CHW format
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        # Convert to LAB
        lab_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        return lab_image
    
    def lab_to_rgb(self, lab_image):
        """Convert LAB image to RGB color space"""
        rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
        return rgb_image
    
    def calculate_psnr(self, original, colorized):
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(original, colorized, data_range=1.0)
    
    def calculate_ssim(self, original, colorized):
        """Calculate Structural Similarity Index"""
        # Determine appropriate window size based on image dimensions
        min_dim = min(original.shape[0], original.shape[1])
        win_size = min(7, min_dim) if min_dim >= 7 else min_dim
        # Ensure win_size is odd and at least 3
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        
        # Use channel_axis for multichannel images
        return ssim(original, colorized, multichannel=True, data_range=1.0, win_size=win_size, channel_axis=2)
    
    def calculate_delta_e2000(self, original_lab, colorized_lab):
        """Calculate ΔE2000 color difference"""
        # Simplified ΔE2000 calculation
        diff = np.sqrt(np.sum((original_lab - colorized_lab) ** 2, axis=2))
        return np.mean(diff)
    
    def calculate_colorfulness_index(self, image):
        """Calculate colorfulness index"""
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Convert to RGB if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Calculate colorfulness using Hasler and Süsstrunk method
        rg = image[:, :, 0] - image[:, :, 1]
        yb = (image[:, :, 0] + image[:, :, 1]) / 2 - image[:, :, 2]
        
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)
        
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        
        colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
        return colorfulness
    
    def calculate_color_histogram_kl_divergence(self, original, colorized):
        """Calculate KL divergence between color histograms"""
        def get_color_histogram(image):
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            
            if image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to HSV for better color representation
            hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Calculate histogram
            hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = hist.flatten()
            hist = hist / np.sum(hist)  # Normalize
            
            return hist
        
        hist_orig = get_color_histogram(original)
        hist_colorized = get_color_histogram(colorized)
        
        # Calculate KL divergence
        kl_div = entropy(hist_orig + 1e-10, hist_colorized + 1e-10)
        return kl_div
    
    def calculate_lpips(self, original, colorized):
        """Calculate LPIPS perceptual similarity"""
        if lpips is None:
            return None
        
        # Ensure tensors are in correct format
        if not isinstance(original, torch.Tensor):
            original = torch.tensor(original).permute(2, 0, 1).unsqueeze(0)
        if not isinstance(colorized, torch.Tensor):
            colorized = torch.tensor(colorized).permute(2, 0, 1).unsqueeze(0)
        
        original = original.to(self.device)
        colorized = colorized.to(self.device)
        
        with torch.no_grad():
            lpips_score = self.lpips_model(original, colorized)
        
        return lpips_score.item()
    
    def colorize_image(self, grayscale_image, model_name='eccv16'):
        """Colorize a grayscale image using specified model"""
        # Convert to PIL Image for processing
        if isinstance(grayscale_image, torch.Tensor):
            grayscale_image = transforms.ToPILImage()(grayscale_image)
        
        # Convert to RGB (grayscale to 3-channel)
        if grayscale_image.mode != 'RGB':
            grayscale_image = grayscale_image.convert('RGB')
        
        # Preprocess
        (tens_l_orig, tens_l_rs) = preprocess_img(np.array(grayscale_image), HW=(256,256))
        
        if self.device != 'cpu':
            tens_l_rs = tens_l_rs.to(self.device)
        
        # Colorize
        if model_name == 'eccv16':
            with torch.no_grad():
                colorized_ab = self.colorizer_eccv16(tens_l_rs).cpu()
        else:  # siggraph17
            with torch.no_grad():
                colorized_ab = self.colorizer_siggraph17(tens_l_rs).cpu()
        
        # Post-process
        colorized_image = postprocess_tens(tens_l_orig, colorized_ab)
        
        return colorized_image
    
    def evaluate_model(self, model_name, test_loader, num_samples=1000):
        """Evaluate a specific model on CIFAR-10 test set"""
        print(f"Evaluating {model_name} model...")
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'delta_e2000': [],
            'lpips': [],
            'colorfulness': [],
            'kl_divergence': [],
            'inference_times': []
        }
        
        count = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            if count >= num_samples:
                break
            
            for i in range(len(images)):
                if count >= num_samples:
                    break
                
                original_image = images[i]
                
                # Convert to grayscale
                grayscale_image = transforms.Grayscale(num_output_channels=1)(original_image)
                grayscale_image = transforms.Grayscale(num_output_channels=3)(grayscale_image)
                
                # Measure inference time
                start_time = time.time()
                colorized_image = self.colorize_image(grayscale_image, model_name)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Convert images to numpy for metric calculation
                original_np = original_image.numpy().transpose(1, 2, 0)
                colorized_np = np.array(colorized_image)
                
                # Calculate metrics
                psnr_score = self.calculate_psnr(original_np, colorized_np)
                ssim_score = self.calculate_ssim(original_np, colorized_np)
                
                # Color difference
                original_lab = self.rgb_to_lab(original_np)
                colorized_lab = self.rgb_to_lab(colorized_np)
                delta_e = self.calculate_delta_e2000(original_lab, colorized_lab)
                
                # Perceptual metrics
                lpips_score = self.calculate_lpips(original_np, colorized_np)
                
                # Colorfulness
                colorfulness = self.calculate_colorfulness_index(colorized_np)
                
                # KL divergence
                kl_div = self.calculate_color_histogram_kl_divergence(original_np, colorized_np)
                
                # Store metrics
                metrics['psnr'].append(psnr_score)
                metrics['ssim'].append(ssim_score)
                metrics['delta_e2000'].append(delta_e)
                if lpips_score is not None:
                    metrics['lpips'].append(lpips_score)
                metrics['colorfulness'].append(colorfulness)
                metrics['kl_divergence'].append(kl_div)
                metrics['inference_times'].append(inference_time)
                
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count}/{num_samples} samples")
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics.items():
            if values:  # Check if list is not empty
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary, metrics
    
    def calculate_model_efficiency(self):
        """Calculate model efficiency metrics"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def estimate_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
        
        efficiency_metrics = {}
        
        for model_name, model in [('eccv16', self.colorizer_eccv16), ('siggraph17', self.colorizer_siggraph17)]:
            efficiency_metrics[model_name] = {
                'parameters_millions': count_parameters(model) / 1e6,
                'model_size_mb': estimate_model_size(model),
                'device': str(next(model.parameters()).device)
            }
        
        return efficiency_metrics
    
    def run_comprehensive_evaluation(self, num_samples=1000):
        """Run comprehensive evaluation on both models"""
        print("Starting comprehensive evaluation...")
        
        # Download CIFAR-10
        trainset, testset = self.download_cifar10()
        test_loader = DataLoader(testset, batch_size=32, shuffle=False)
        
        # Evaluate both models
        results = {}
        
        for model_name in ['eccv16', 'siggraph17']:
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name.upper()} Model")
            print(f"{'='*50}")
            
            summary, detailed_metrics = self.evaluate_model(model_name, test_loader, num_samples)
            results[model_name] = {
                'summary': summary,
                'detailed_metrics': detailed_metrics
            }
        
        # Calculate efficiency metrics
        efficiency_metrics = self.calculate_model_efficiency()
        
        # Generate report
        self.generate_report(results, efficiency_metrics)
        
        return results, efficiency_metrics
    
    def convert_to_json_serializable(self, obj):
        """Convert NumPy types to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def generate_report(self, results, efficiency_metrics):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE COLORIZATION MODEL EVALUATION REPORT")
        print("="*80)
        
        # Reconstruction Metrics
        print("\n📊 RECONSTRUCTION METRICS")
        print("-" * 50)
        print(f"{'Metric':<15} {'ECCV16':<15} {'SIGGRAPH17':<15}")
        print("-" * 50)
        
        for metric in ['psnr', 'ssim', 'delta_e2000']:
            eccv16_val = results['eccv16']['summary'][metric]['mean']
            siggraph17_val = results['siggraph17']['summary'][metric]['mean']
            print(f"{metric.upper():<15} {eccv16_val:<15.3f} {siggraph17_val:<15.3f}")
        
        # Perceptual Metrics
        print("\n🎨 PERCEPTUAL METRICS")
        print("-" * 50)
        print(f"{'Metric':<15} {'ECCV16':<15} {'SIGGRAPH17':<15}")
        print("-" * 50)
        
        for metric in ['lpips', 'colorfulness']:
            if metric in results['eccv16']['summary']:
                eccv16_val = results['eccv16']['summary'][metric]['mean']
                siggraph17_val = results['siggraph17']['summary'][metric]['mean']
                print(f"{metric.upper():<15} {eccv16_val:<15.3f} {siggraph17_val:<15.3f}")
        
        # Semantic Correctness
        print("\n🧠 SEMANTIC CORRECTNESS")
        print("-" * 50)
        print(f"{'Metric':<15} {'ECCV16':<15} {'SIGGRAPH17':<15}")
        print("-" * 50)
        
        kl_div_eccv16 = results['eccv16']['summary']['kl_divergence']['mean']
        kl_div_siggraph17 = results['siggraph17']['summary']['kl_divergence']['mean']
        print(f"{'KL_DIVERGENCE':<15} {kl_div_eccv16:<15.3f} {kl_div_siggraph17:<15.3f}")
        
        # Efficiency Metrics
        print("\n⚡ EFFICIENCY METRICS")
        print("-" * 50)
        print(f"{'Metric':<20} {'ECCV16':<15} {'SIGGRAPH17':<15}")
        print("-" * 50)
        
        for model_name in ['eccv16', 'siggraph17']:
            params = efficiency_metrics[model_name]['parameters_millions']
            size = efficiency_metrics[model_name]['model_size_mb']
            avg_time = np.mean(results[model_name]['summary']['inference_times']['mean'])
            
            print(f"{'Parameters (M)':<20} {params:<15.2f} {efficiency_metrics['siggraph17']['parameters_millions']:<15.2f}")
            print(f"{'Model Size (MB)':<20} {size:<15.2f} {efficiency_metrics['siggraph17']['model_size_mb']:<15.2f}")
            print(f"{'Avg Latency (ms)':<20} {avg_time:<15.2f} {np.mean(results['siggraph17']['summary']['inference_times']['mean']):<15.2f}")
            break  # Only print once
        
        # Save detailed results to JSON
        detailed_results = {
            'reconstruction_metrics': {
                'eccv16': {k: v for k, v in results['eccv16']['summary'].items() if k in ['psnr', 'ssim', 'delta_e2000']},
                'siggraph17': {k: v for k, v in results['siggraph17']['summary'].items() if k in ['psnr', 'ssim', 'delta_e2000']}
            },
            'perceptual_metrics': {
                'eccv16': {k: v for k, v in results['eccv16']['summary'].items() if k in ['lpips', 'colorfulness']},
                'siggraph17': {k: v for k, v in results['siggraph17']['summary'].items() if k in ['lpips', 'colorfulness']}
            },
            'semantic_metrics': {
                'eccv16': {k: v for k, v in results['eccv16']['summary'].items() if k in ['kl_divergence']},
                'siggraph17': {k: v for k, v in results['siggraph17']['summary'].items() if k in ['kl_divergence']}
            },
            'efficiency_metrics': efficiency_metrics,
            'inference_times': {
                'eccv16': results['eccv16']['summary']['inference_times'],
                'siggraph17': results['siggraph17']['summary']['inference_times']
            }
        }
        
        # Convert to JSON-serializable format
        detailed_results = self.convert_to_json_serializable(detailed_results)
        
        with open('colorization_evaluation_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: colorization_evaluation_results.json")
        print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate colorization models with optional ILA')
    parser.add_argument('--use_ila', action='store_true', 
                       help='Enable ILA blocks in ECCV16 model')
    parser.add_argument('--ila_reduction', type=int, default=4,
                       help='Channel reduction factor for ILA (default: 4)')
    parser.add_argument('--ila_use_dw_conv', action='store_true', default=True,
                       help='Use depthwise convolution in ILA (default: True)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda). Auto-detects if not specified')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Initialize evaluator
    evaluator = ColorizationEvaluator(
        device=device,
        use_ila=args.use_ila,
        ila_reduction=args.ila_reduction,
        ila_use_dw_conv=args.ila_use_dw_conv
    )
    
    # Run comprehensive evaluation
    results, efficiency_metrics = evaluator.run_comprehensive_evaluation(num_samples=1000)
    
    print("\n✅ Evaluation completed successfully!")
    print("Check 'colorization_evaluation_results.json' for detailed metrics.")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Device: {device}")
    print(f"\nILA Configuration:")
    if args.use_ila:
        print(f"  ✓ ILA ENABLED in ECCV16")
        print(f"  - Reduction factor: {args.ila_reduction}")
        print(f"  - Depthwise conv: {args.ila_use_dw_conv}")
    else:
        print(f"  ✗ ILA DISABLED (baseline ECCV16)")
    print(f"\nModels evaluated:")
    print(f"  - ECCV16 ({'with ILA' if args.use_ila else 'baseline'})")
    print(f"  - SIGGRAPH17")
    if 'eccv16' in results and 'summary' in results['eccv16']:
        if 'psnr' in results['eccv16']['summary']:
            eccv16_psnr = results['eccv16']['summary']['psnr']['mean']
            siggraph17_psnr = results['siggraph17']['summary']['psnr']['mean'] if 'siggraph17' in results else 'N/A'
            print(f"\nQuick Results:")
            print(f"  - ECCV16 PSNR: {eccv16_psnr:.3f} dB")
            if siggraph17_psnr != 'N/A':
                print(f"  - SIGGRAPH17 PSNR: {siggraph17_psnr:.3f} dB")
    print("="*60)

if __name__ == "__main__":
    main()

