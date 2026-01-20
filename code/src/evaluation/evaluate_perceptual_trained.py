#!/usr/bin/env python3
"""
Evaluate Models Trained with Perceptual Loss
Simple script to evaluate models trained with perceptual loss and compare with baselines
"""

import os
import sys
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import json
import glob

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17

class PerceptualTrainedEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all available trained models"""
        print("="*80)
        print("Loading Models")
        print("="*80)
        
        # Load pre-trained models for comparison
        try:
            self.models['pretrained_eccv16'] = eccv16(pretrained=True).eval().to(self.device)
            self.models['pretrained_siggraph17'] = siggraph17(pretrained=True).eval().to(self.device)
            print("✅ Loaded pre-trained models")
        except Exception as e:
            print(f"⚠️  Could not load pre-trained models: {e}")
        
        # Find and load trained models
        model_patterns = {
            'eccv16': [
                'eccv16_best_model.pth',  # Baseline
                'eccv16_ila_best_model.pth',  # ILA
                'eccv16*perceptual*best_model.pth',  # Perceptual loss variants
            ],
            'siggraph17': [
                'siggraph17_best_model.pth',  # Baseline
                'siggraph17*perceptual*best_model.pth',  # Perceptual loss variants
            ]
        }
        
        for model_name in ['eccv16', 'siggraph17']:
            for pattern in model_patterns[model_name]:
                for filepath in glob.glob(pattern):
                    if os.path.isfile(filepath):
                        try:
                            # Determine if ILA is needed (only for eccv16)
                            use_ila = 'ila' in filepath and model_name == 'eccv16'
                            
                            if model_name == 'eccv16':
                                model = eccv16(pretrained=False, use_ila=use_ila).eval().to(self.device)
                            else:
                                model = siggraph17(pretrained=False).eval().to(self.device)
                            
                            model.load_state_dict(torch.load(filepath, map_location=self.device))
                            
                            # Create a clean key name
                            key = filepath.replace('_best_model.pth', '').replace('.pth', '')
                            self.models[key] = model
                            print(f"✅ Loaded: {filepath}")
                        except Exception as e:
                            print(f"❌ Failed to load {filepath}: {e}")
        
        print(f"\n📊 Total models loaded: {len(self.models)}")
        for key in sorted(self.models.keys()):
            print(f"   - {key}")
        print("="*80)
    
    def prepare_data(self, num_samples=1000):
        """Prepare CIFAR-10 test data"""
        print(f"\n📁 Preparing CIFAR-10 test data ({num_samples} samples)...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./cifar-10-python', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        if num_samples < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:num_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"✅ Test samples: {len(test_dataset)}")
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
    
    def colorize_image(self, rgb_image, model):
        """Colorize a grayscale image using the model"""
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
        original = np.clip(original, 0, 1)
        colorized = np.clip(colorized, 0, 1)
        
        # PSNR
        psnr_value = psnr(original, colorized, data_range=1.0)
        
        # SSIM
        ssim_value = ssim(original, colorized, data_range=1.0, channel_axis=2, multichannel=True)
        
        # Color difference in LAB space
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
        print(f"\n🔍 Evaluating: {model_key}")
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'color_diff': []
        }
        
        count = 0
        
        for batch_idx, (rgb_image, _) in enumerate(test_loader):
            if count >= num_samples:
                break
            
            try:
                rgb_image = rgb_image.to(self.device)
                
                # Get original image
                original = rgb_image[0].permute(1, 2, 0).cpu().numpy()
                
                # Colorize
                colorized = self.colorize_image(rgb_image, model)
                
                # Calculate metrics
                metric_values = self.calculate_metrics(original, colorized)
                
                metrics['psnr'].append(metric_values['psnr'])
                metrics['ssim'].append(metric_values['ssim'])
                metrics['color_diff'].append(metric_values['color_diff'])
                
                count += 1
                
                if count % 100 == 0:
                    print(f"   Processed {count}/{num_samples} samples")
            
            except Exception as e:
                print(f"   ⚠️  Error processing sample {count}: {e}")
                continue
        
        # Calculate summary statistics
        summary = {
            'psnr': {
                'mean': float(np.mean(metrics['psnr'])),
                'std': float(np.std(metrics['psnr']))
            },
            'ssim': {
                'mean': float(np.mean(metrics['ssim'])),
                'std': float(np.std(metrics['ssim']))
            },
            'color_diff': {
                'mean': float(np.mean(metrics['color_diff'])),
                'std': float(np.std(metrics['color_diff']))
            }
        }
        
        return summary, metrics
    
    def run_evaluation(self, num_samples=1000):
        """Run evaluation on all loaded models"""
        print("\n" + "="*80)
        print("EVALUATION STARTED")
        print("="*80)
        
        # Prepare test data
        test_loader = self.prepare_data(num_samples)
        
        results = {}
        
        # Evaluate all models
        for model_key in sorted(self.models.keys()):
            summary, detailed_metrics = self.evaluate_model(
                model_key, test_loader, num_samples
            )
            
            results[model_key] = {
                'summary': summary,
                'num_samples': len(detailed_metrics['psnr'])
            }
        
        # Generate report
        self.generate_report(results)
        
        return results
    
    def generate_report(self, results):
        """Generate evaluation report"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Group by model type
        eccv16_results = {k: v for k, v in results.items() if 'eccv16' in k}
        siggraph17_results = {k: v for k, v in results.items() if 'siggraph17' in k}
        
        for model_type, model_results in [('ECCV16', eccv16_results), ('SIGGRAPH17', siggraph17_results)]:
            if not model_results:
                continue
                
            print(f"\n{model_type} Models:")
            print("-" * 80)
            print(f"{'Model':<40} {'PSNR':<12} {'SSIM':<12} {'Color Diff':<12}")
            print("-" * 80)
            
            for model_key in sorted(model_results.keys()):
                summary = model_results[model_key]['summary']
                print(f"{model_key:<40} "
                      f"{summary['psnr']['mean']:.4f}±{summary['psnr']['std']:.4f}  "
                      f"{summary['ssim']['mean']:.4f}±{summary['ssim']['std']:.4f}  "
                      f"{summary['color_diff']['mean']:.4f}±{summary['color_diff']['std']:.4f}")
        
        # Save results
        output_file = 'perceptual_trained_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print(f"✅ Results saved to: {output_file}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate models trained with perceptual loss')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to evaluate (default: 1000)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda). Auto-detects if not specified')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🖥️  Using device: {device}")
    
    # Initialize evaluator
    evaluator = PerceptualTrainedEvaluator(device=device)
    
    if len(evaluator.models) == 0:
        print("\n❌ No models found! Please train models first.")
        print("   Expected files:")
        print("   - eccv16_best_model.pth")
        print("   - eccv16_perceptual_w0.1_l1_best_model.pth")
        print("   - siggraph17_best_model.pth")
        print("   - siggraph17_perceptual_w0.1_l1_best_model.pth")
        return
    
    # Run evaluation
    results = evaluator.run_evaluation(num_samples=args.num_samples)
    
    print("\n🎉 Evaluation completed!")
    print("Check 'perceptual_trained_evaluation_results.json' for detailed results.")


if __name__ == "__main__":
    main()

