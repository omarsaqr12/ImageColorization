#!/usr/bin/env python3
"""
Fixed Evaluate Trained Colorization Models Script
Fixes tensor dimension issues
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

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens

class FixedTrainedModelEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.load_models()
        
    def load_models(self):
        """Load both pre-trained and trained models"""
        print("Loading models...")
        
        # Load pre-trained models
        self.pretrained_eccv16 = eccv16(pretrained=True).eval().to(self.device)
        self.pretrained_siggraph17 = siggraph17(pretrained=True).eval().to(self.device)
        
        # Load trained models
        self.trained_eccv16 = eccv16(pretrained=False).eval().to(self.device)
        self.trained_siggraph17 = siggraph17(pretrained=False).eval().to(self.device)
        
        # Load trained weights if they exist
        if os.path.exists('eccv16_best_model.pth'):
            self.trained_eccv16.load_state_dict(torch.load('eccv16_best_model.pth', map_location=self.device))
            print("✅ Loaded trained ECCV16 model")
        else:
            print("❌ Trained ECCV16 model not found. Please train the model first.")
            
        if os.path.exists('siggraph17_best_model.pth'):
            self.trained_siggraph17.load_state_dict(torch.load('siggraph17_best_model.pth', map_location=self.device))
            print("✅ Loaded trained SIGGRAPH17 model")
        else:
            print("❌ Trained SIGGRAPH17 model not found. Please train the model first.")
        
        print("Models loaded successfully!")
    
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
        # Ensure tensor is 4D (batch, channels, height, width)
        if rgb_tensor.dim() == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)
        
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
    
    def colorize_with_model(self, grayscale_image, model, model_type='trained'):
        """Colorize using either pre-trained or trained model with consistent preprocessing"""
        # Ensure grayscale_image is a 4D tensor
        if grayscale_image.dim() == 3:
            grayscale_image = grayscale_image.unsqueeze(0)
        
        # Convert RGB to LAB (same for both model types)
        lab_image = self.rgb_to_lab(grayscale_image)
        l_channel = lab_image[:, 0:1, :, :]
        
        with torch.no_grad():
            predicted_ab = model(l_channel)
        
        # Combine L and predicted AB channels
        predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
        predicted_rgb = self.lab_to_rgb(predicted_lab)
        
        # Convert to numpy
        predicted_rgb_np = predicted_rgb.permute(0, 2, 3, 1).cpu().numpy()[0]
        return predicted_rgb_np
    
    def calculate_metrics(self, original, colorized):
        """Calculate evaluation metrics"""
        # Ensure images are in correct format
        if isinstance(original, torch.Tensor):
            original = original.numpy().transpose(1, 2, 0)
        
        # Calculate PSNR
        psnr_score = psnr(original, colorized, data_range=1.0)
        
        # Calculate SSIM
        min_dim = min(original.shape[0], original.shape[1])
        win_size = min(7, min_dim) if min_dim >= 7 else min_dim
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        
        ssim_score = ssim(original, colorized, multichannel=True, data_range=1.0, 
                          win_size=win_size, channel_axis=2)
        
        # Calculate color difference (simplified)
        diff = np.sqrt(np.sum((original - colorized) ** 2, axis=2))
        color_diff = np.mean(diff)
        
        return {
            'psnr': psnr_score,
            'ssim': ssim_score,
            'color_diff': color_diff
        }
    
    def evaluate_model(self, model, model_name, model_type, test_loader, num_samples=1000):
        """Evaluate a specific model"""
        print(f"\nEvaluating {model_name} ({model_type})...")
        
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
                
            rgb_image = rgb_image.to(self.device)
            # FIXED: Handle tensor dimensions correctly
            if rgb_image.dim() == 4 and rgb_image.shape[0] == 1:
                original_np = rgb_image.squeeze(0).numpy().transpose(1, 2, 0)
            else:
                original_np = rgb_image.numpy().transpose(1, 2, 0)
            
            # Convert to grayscale
            grayscale_image = transforms.Grayscale(num_output_channels=3)(rgb_image)
            
            # Measure inference time
            start_time = time.time()
            colorized_image = self.colorize_with_model(grayscale_image, model, model_type)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate metrics
            model_metrics = self.calculate_metrics(original_np, colorized_image)
            
            # Store results
            metrics['psnr'].append(model_metrics['psnr'])
            metrics['ssim'].append(model_metrics['ssim'])
            metrics['color_diff'].append(model_metrics['color_diff'])
            metrics['inference_times'].append(inference_time)
            
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count}/{num_samples} samples")
        
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
        print("Starting comprehensive evaluation of trained vs pre-trained models...")
        
        # Prepare test data
        test_loader = self.prepare_data(num_samples)
        
        results = {}
        
        # Evaluate all models
        models_to_evaluate = [
            (self.pretrained_eccv16, 'eccv16', 'pretrained'),
            (self.pretrained_siggraph17, 'siggraph17', 'pretrained'),
            (self.trained_eccv16, 'eccv16', 'trained'),
            (self.trained_siggraph17, 'siggraph17', 'trained')
        ]
        
        for model, model_name, model_type in models_to_evaluate:
            summary, detailed_metrics = self.evaluate_model(
                model, model_name, model_type, test_loader, num_samples
            )
            
            key = f"{model_name}_{model_type}"
            results[key] = {
                'summary': summary,
                'detailed_metrics': detailed_metrics
            }
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("TRAINED vs PRE-TRAINED MODEL COMPARISON REPORT")
        print("="*80)
        
        # Reconstruction Metrics Comparison
        print("\n📊 RECONSTRUCTION METRICS COMPARISON")
        print("-" * 60)
        print(f"{'Model':<20} {'Type':<12} {'PSNR':<8} {'SSIM':<8} {'Color Diff':<12}")
        print("-" * 60)
        
        for model_name in ['eccv16', 'siggraph17']:
            for model_type in ['pretrained', 'trained']:
                key = f"{model_name}_{model_type}"
                if key in results:
                    summary = results[key]['summary']
                    print(f"{model_name.upper():<20} {model_type:<12} "
                          f"{summary['psnr']['mean']:<8.3f} {summary['ssim']['mean']:<8.3f} "
                          f"{summary['color_diff']['mean']:<12.3f}")
        
        # Performance Comparison
        print("\n⚡ PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Model':<20} {'Type':<12} {'Avg Time (ms)':<15}")
        print("-" * 60)
        
        for model_name in ['eccv16', 'siggraph17']:
            for model_type in ['pretrained', 'trained']:
                key = f"{model_name}_{model_type}"
                if key in results:
                    summary = results[key]['summary']
                    print(f"{model_name.upper():<20} {model_type:<12} "
                          f"{summary['inference_times']['mean']:<15.2f}")
        
        # Improvement Analysis
        print("\n📈 IMPROVEMENT ANALYSIS")
        print("-" * 60)
        
        for model_name in ['eccv16', 'siggraph17']:
            pretrained_key = f"{model_name}_pretrained"
            trained_key = f"{model_name}_trained"
            
            if pretrained_key in results and trained_key in results:
                pretrained = results[pretrained_key]['summary']
                trained = results[trained_key]['summary']
                
                psnr_improvement = trained['psnr']['mean'] - pretrained['psnr']['mean']
                ssim_improvement = trained['ssim']['mean'] - pretrained['ssim']['mean']
                color_diff_improvement = pretrained['color_diff']['mean'] - trained['color_diff']['mean']
                
                print(f"\n{model_name.upper()} Model Improvements:")
                print(f"  PSNR: {psnr_improvement:+.3f} ({psnr_improvement/pretrained['psnr']['mean']*100:+.1f}%)")
                print(f"  SSIM: {ssim_improvement:+.3f} ({ssim_improvement/pretrained['ssim']['mean']*100:+.1f}%)")
                print(f"  Color Diff: {color_diff_improvement:+.3f} ({color_diff_improvement/pretrained['color_diff']['mean']*100:+.1f}%)")
        
        # Save detailed results
        self.save_results(results)
        
        print("\n" + "="*80)
    
    def save_results(self, results):
        """Save detailed results to JSON"""
        # Convert to JSON-serializable format
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        results_json = convert_to_json_serializable(results)
        
        with open('trained_vs_pretrained_comparison.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: trained_vs_pretrained_comparison.json")

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize evaluator
    evaluator = FixedTrainedModelEvaluator(device=device)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(num_samples=1000)
    
    print("\n🎉 Evaluation completed!")
    print("Check the following files:")
    print("- trained_vs_pretrained_comparison.json")

if __name__ == "__main__":
    main()
