#!/usr/bin/env python3
"""
Evaluate Colorization Models Trained with Perceptual Loss
Evaluates models trained with perceptual loss and compares with baseline models
"""

import os
import sys
import time
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
from skimage import color
import json
import glob
import argparse

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens


class PerceptualModelEvaluator:
    def __init__(self, device='cpu', use_ila=False):
        self.device = device
        self.use_ila = use_ila
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all available trained models"""
        print("Loading models...")
        
        # Load pre-trained models for comparison
        self.models['pretrained_eccv16'] = eccv16(pretrained=True).eval().to(self.device)
        self.models['pretrained_siggraph17'] = siggraph17(pretrained=True).eval().to(self.device)
        print("✅ Loaded pre-trained models")
        
        # Find all trained model files
        model_files = {
            'eccv16': [],
            'siggraph17': []
        }
        
        # Look for baseline models
        if os.path.exists('eccv16_best_model.pth'):
            model_files['eccv16'].append(('eccv16_best_model.pth', 'baseline'))
        if os.path.exists('siggraph17_best_model.pth'):
            model_files['siggraph17'].append(('siggraph17_best_model.pth', 'baseline'))
        
        # Look for ILA models
        if os.path.exists('eccv16_ila_best_model.pth'):
            model_files['eccv16'].append(('eccv16_ila_best_model.pth', 'ila'))
        
        # Look for perceptual loss models
        for model_name in ['eccv16', 'siggraph17']:
            pattern = f'{model_name}*perceptual*best_model.pth'
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                # Extract config from filename
                if 'perceptual' in filename:
                    model_files[model_name].append((filepath, 'perceptual'))
        
        # Load found models
        for model_name in ['eccv16', 'siggraph17']:
            for filepath, model_type in model_files[model_name]:
                try:
                    if model_name == 'eccv16':
                        # Check if ILA is needed based on filename
                        use_ila_for_model = 'ila' in filepath or (model_type == 'ila')
                        model = eccv16(pretrained=False, use_ila=use_ila_for_model).eval().to(self.device)
                    else:
                        model = siggraph17(pretrained=False).eval().to(self.device)
                    
                    model.load_state_dict(torch.load(filepath, map_location=self.device))
                    # Create a cleaner key name
                    filename = os.path.basename(filepath).replace('_best_model.pth', '')
                    key = f"{model_name}_{filename}"
                    self.models[key] = model
                    print(f"✅ Loaded {model_name} ({model_type}): {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"❌ Failed to load {filepath}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\nTotal models loaded: {len(self.models)}")
        for key in self.models.keys():
            print(f"  - {key}")
    
    def prepare_data(self, num_samples=1000):
        """Prepare CIFAR-10 test data"""
        print(f"Preparing CIFAR-10 test data ({num_samples} samples)...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./cifar-10-python', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Use subset if specified
        if num_samples < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:num_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Test samples: {len(test_dataset)}")
        return test_loader
    
    def rgb_to_lab(self, rgb_tensor):
        """Convert RGB tensor to LAB tensor"""
        rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
        lab_np = np.zeros_like(rgb_np)
        
        for i in range(rgb_np.shape[0]):
            rgb_normalized = np.clip(rgb_np[i], 0, 1)
            lab_np[i] = color.rgb2lab(rgb_normalized)
        
        lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
        return lab_tensor.to(self.device)
    
    def lab_to_rgb(self, lab_tensor):
        """Convert LAB tensor to RGB tensor"""
        lab_np = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()
        rgb_np = np.zeros_like(lab_np)
        
        for i in range(lab_np.shape[0]):
            rgb_np[i] = color.lab2rgb(lab_np[i])
            rgb_np[i] = np.clip(rgb_np[i], 0, 1)
        
        rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
        return rgb_tensor.to(self.device)
    
    def colorize_with_model(self, rgb_image, model, use_pretrained_preprocessing=False):
        """Colorize image using specified model"""
        if use_pretrained_preprocessing:
            # Use original preprocessing for pre-trained models
            if isinstance(rgb_image, torch.Tensor):
                rgb_image = transforms.ToPILImage()(rgb_image.squeeze(0))
            
            if rgb_image.mode != 'RGB':
                rgb_image = rgb_image.convert('RGB')
            
            (tens_l_orig, tens_l_rs) = preprocess_img(np.array(rgb_image), HW=(256,256))
            
            if self.device != 'cpu':
                tens_l_rs = tens_l_rs.to(self.device)
            
            with torch.no_grad():
                colorized_ab = model(tens_l_rs).cpu()
            
            colorized_image = postprocess_tens(tens_l_orig, colorized_ab)
            return colorized_image
        else:
            # Use standard preprocessing for trained models
            if isinstance(rgb_image, torch.Tensor):
                rgb_image = rgb_image.to(self.device)
            else:
                rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(0).to(self.device)
            
            # Convert RGB to LAB
            lab_image = self.rgb_to_lab(rgb_image)
            l_channel = lab_image[:, 0:1, :, :]
            
            # Colorize
            with torch.no_grad():
                predicted_ab = model(l_channel)
            
            # Reconstruct RGB
            predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
            predicted_rgb = self.lab_to_rgb(predicted_lab)
            
            # Convert to numpy
            predicted_rgb_np = predicted_rgb[0].permute(1, 2, 0).cpu().numpy()
            return predicted_rgb_np
    
    def calculate_metrics(self, original, colorized):
        """Calculate evaluation metrics"""
        # Ensure images are in [0, 1] range
        original = np.clip(original, 0, 1)
        colorized = np.clip(colorized, 0, 1)
        
        # Calculate PSNR
        psnr_value = psnr(original, colorized, data_range=1.0)
        
        # Calculate SSIM
        ssim_value = ssim(original, colorized, data_range=1.0, channel_axis=2, multichannel=True)
        
        # Calculate color difference (L2 distance in LAB space)
        original_lab = color.rgb2lab(original)
        colorized_lab = color.rgb2lab(colorized)
        color_diff = np.mean(np.sqrt(np.sum((original_lab - colorized_lab)**2, axis=2)))
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'color_diff': color_diff
        }
    
    def evaluate_model(self, model_key, test_loader, num_samples=1000):
        """Evaluate a specific model"""
        model = self.models[model_key]
        use_pretrained = 'pretrained' in model_key
        
        print(f"\nEvaluating {model_key}...")
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'color_diff': [],
            'inference_times': []
        }
        
        count = 0
        
        for batch_idx, (rgb_image, _) in enumerate(test_loader):
            if count >= num_samples:
                break
            
            try:
                # Measure inference time
                start_time = time.time()
                colorized = self.colorize_with_model(rgb_image, model, use_pretrained_preprocessing=use_pretrained)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Get original image
                if isinstance(rgb_image, torch.Tensor):
                    original = rgb_image[0].permute(1, 2, 0).cpu().numpy()
                else:
                    original = np.array(rgb_image) / 255.0
                
                # Ensure colorized is in correct format
                if colorized.shape[2] != 3:
                    colorized = colorized.transpose(1, 2, 0)
                
                # Calculate metrics
                metric_values = self.calculate_metrics(original, colorized)
                
                metrics['psnr'].append(metric_values['psnr'])
                metrics['ssim'].append(metric_values['ssim'])
                metrics['color_diff'].append(metric_values['color_diff'])
                metrics['inference_times'].append(inference_time)
                
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count}/{num_samples} samples")
            
            except Exception as e:
                print(f"Error processing sample {count}: {e}")
                continue
        
        # Calculate summary statistics
        summary = {
            'psnr': {
                'mean': np.mean(metrics['psnr']),
                'std': np.std(metrics['psnr'])
            },
            'ssim': {
                'mean': np.mean(metrics['ssim']),
                'std': np.std(metrics['ssim'])
            },
            'color_diff': {
                'mean': np.mean(metrics['color_diff']),
                'std': np.std(metrics['color_diff'])
            },
            'inference_times': {
                'mean': np.mean(metrics['inference_times']),
                'std': np.std(metrics['inference_times'])
            }
        }
        
        return summary, metrics
    
    def run_comprehensive_evaluation(self, num_samples=1000):
        """Run comprehensive evaluation of all models"""
        print("Starting comprehensive evaluation...")
        
        # Prepare test data
        test_loader = self.prepare_data(num_samples)
        
        results = {}
        
        # Evaluate all models
        for model_key in self.models.keys():
            summary, detailed_metrics = self.evaluate_model(
                model_key, test_loader, num_samples
            )
            
            results[model_key] = {
                'summary': summary,
                'detailed_metrics': detailed_metrics
            }
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate comparison report"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Group results by model type
        eccv16_results = {k: v for k, v in results.items() if 'eccv16' in k}
        siggraph17_results = {k: v for k, v in results.items() if 'siggraph17' in k}
        
        for model_type, model_results in [('ECCV16', eccv16_results), ('SIGGRAPH17', siggraph17_results)]:
            print(f"\n{model_type} Models:")
            print("-" * 80)
            
            for model_key, result in sorted(model_results.items()):
                summary = result['summary']
                print(f"\n{model_key}:")
                print(f"  PSNR:    {summary['psnr']['mean']:.4f} ± {summary['psnr']['std']:.4f}")
                print(f"  SSIM:    {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
                print(f"  Color Diff: {summary['color_diff']['mean']:.4f} ± {summary['color_diff']['std']:.4f}")
                print(f"  Inference: {summary['inference_times']['mean']:.2f} ± {summary['inference_times']['std']:.2f} ms")
        
        # Save results to JSON
        output_file = 'perceptual_models_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate models trained with perceptual loss')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to evaluate (default: 1000)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda). Auto-detects if not specified')
    parser.add_argument('--use_ila', action='store_true',
                       help='Use ILA models if available')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Initialize evaluator
    evaluator = PerceptualModelEvaluator(device=device, use_ila=args.use_ila)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(num_samples=args.num_samples)
    
    print("\n🎉 Evaluation completed!")
    print("Check 'perceptual_models_evaluation_results.json' for detailed results.")


if __name__ == "__main__":
    main()

