#!/usr/bin/env python3
"""
ImageNet Pretrained Model Evaluation Script - Fixed Version
Fixes the image loading issue for proper colorization
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import json
import glob
import random
from pathlib import Path

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens

class ImageNetDataset(Dataset):
    """Custom ImageNet dataset loader"""
    
    def __init__(self, root_dir, transform=None, max_samples=None):
        """
        Args:
            root_dir (str): Path to ImageNet dataset directory
            transform (callable, optional): Optional transform to be applied on a sample
            max_samples (int, optional): Maximum number of samples to load
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        # Limit samples if specified
        if max_samples and len(self.image_paths) > max_samples:
            self.image_paths = random.sample(self.image_paths, max_samples)
        
        print(f"Found {len(self.image_paths)} images in ImageNet dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Get class name from directory structure
            class_name = os.path.basename(os.path.dirname(image_path))
            
            return image, class_name, image_path
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, "unknown", image_path

class ImageNetPretrainedEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.load_models()
        
    def load_models(self):
        """Load pretrained colorization models"""
        print("Loading pretrained colorization models...")
        
        # Load pretrained models
        self.pretrained_eccv16 = eccv16(pretrained=True).eval().to(self.device)
        self.pretrained_siggraph17 = siggraph17(pretrained=True).eval().to(self.device)
        
        print("✅ Pretrained models loaded successfully!")
    
    def prepare_imagenet_data(self, imagenet_path, num_samples=1000):
        """Prepare ImageNet dataset"""
        print(f"Preparing ImageNet dataset ({num_samples} samples)...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to model input size
            transforms.ToTensor(),
        ])
        
        # Create dataset
        dataset = ImageNetDataset(
            root_dir=imagenet_path,
            transform=transform,
            max_samples=num_samples
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        print(f"ImageNet samples: {len(dataset)}")
        return dataloader
    
    def colorize_with_pretrained_model(self, image_path, model, model_name):
        """Colorize using pretrained model with proper image loading"""
        try:
            # Load image as numpy array first
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # Now use preprocess_img with the numpy array
            tens_l_orig, tens_l_rs = preprocess_img(img_np, HW=(256, 256))
            
            if self.device == 'cuda':
                tens_l_rs = tens_l_rs.cuda()
            
            # Colorize
            with torch.no_grad():
                predicted_ab = model(tens_l_rs)
            
            # Postprocess
            colorized_img = postprocess_tens(tens_l_orig, predicted_ab.cpu())
            
            return colorized_img
            
        except Exception as e:
            print(f"Error colorizing with {model_name}: {e}")
            return None
    
    def calculate_metrics(self, original, colorized):
        """Calculate evaluation metrics"""
        try:
            # Ensure images are numpy arrays
            if isinstance(original, torch.Tensor):
                original = original.numpy().transpose(1, 2, 0)
            if isinstance(colorized, torch.Tensor):
                colorized = colorized.numpy()
            
            # Ensure both images have the same shape
            if original.shape != colorized.shape:
                # Resize colorized to match original
                colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))
            
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
            
            # Calculate color difference
            diff = np.sqrt(np.sum((original - colorized) ** 2, axis=2))
            color_diff = np.mean(diff)
            
            return {
                'psnr': psnr_score,
                'ssim': ssim_score,
                'color_diff': color_diff
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'color_diff': 1.0
            }
    
    def evaluate_model(self, model, model_name, dataloader, num_samples=1000):
        """Evaluate a specific pretrained model"""
        print(f"\nEvaluating {model_name} on ImageNet...")
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'color_diff': [],
            'inference_times': [],
            'class_names': [],
            'image_paths': []
        }
        
        count = 0
        successful_predictions = 0
        
        for batch_idx, (rgb_image, class_name, image_path) in enumerate(dataloader):
            if count >= num_samples:
                break
            
            try:
                # Get image path (single item from batch)
                if isinstance(image_path, (list, tuple)):
                    image_path = image_path[0]
                if isinstance(class_name, (list, tuple)):
                    class_name = class_name[0]
                
                # Load original image for comparison
                original_img = Image.open(image_path).convert('RGB')
                original_img = original_img.resize((256, 256))
                original_np = np.array(original_img) / 255.0
                
                # Measure inference time
                start_time = time.time()
                colorized_img = self.colorize_with_pretrained_model(image_path, model, model_name)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                if colorized_img is not None:
                    # Calculate metrics
                    model_metrics = self.calculate_metrics(original_np, colorized_img)
                    
                    # Store results
                    metrics['psnr'].append(model_metrics['psnr'])
                    metrics['ssim'].append(model_metrics['ssim'])
                    metrics['color_diff'].append(model_metrics['color_diff'])
                    metrics['inference_times'].append(inference_time)
                    metrics['class_names'].append(class_name)
                    metrics['image_paths'].append(image_path)
                    
                    successful_predictions += 1
                    print(f"  ✅ Successfully processed {os.path.basename(image_path)}")
                else:
                    print(f"  ❌ Failed to colorize image: {os.path.basename(image_path)}")
                
                count += 1
                
                if count % 5 == 0:
                    print(f"Processed {count}/{num_samples} samples (Success: {successful_predictions})")
                    
            except Exception as e:
                print(f"Error processing sample {count}: {e}")
                count += 1
                continue
        
        # Calculate summary statistics
        if successful_predictions > 0:
            summary = {
                'total_samples': count,
                'successful_predictions': successful_predictions,
                'success_rate': successful_predictions / count,
                'psnr': {
                    'mean': np.mean(metrics['psnr']),
                    'std': np.std(metrics['psnr']),
                    'min': np.min(metrics['psnr']),
                    'max': np.max(metrics['psnr'])
                },
                'ssim': {
                    'mean': np.mean(metrics['ssim']),
                    'std': np.std(metrics['ssim']),
                    'min': np.min(metrics['ssim']),
                    'max': np.max(metrics['ssim'])
                },
                'color_diff': {
                    'mean': np.mean(metrics['color_diff']),
                    'std': np.std(metrics['color_diff']),
                    'min': np.min(metrics['color_diff']),
                    'max': np.max(metrics['color_diff'])
                },
                'inference_times': {
                    'mean': np.mean(metrics['inference_times']),
                    'std': np.std(metrics['inference_times']),
                    'min': np.min(metrics['inference_times']),
                    'max': np.max(metrics['inference_times'])
                }
            }
        else:
            summary = {
                'total_samples': count,
                'successful_predictions': 0,
                'success_rate': 0.0,
                'error': 'No successful predictions'
            }
        
        return summary, metrics
    
    def run_imagenet_evaluation(self, imagenet_path, num_samples=1000):
        """Run comprehensive evaluation on ImageNet"""
        print("Starting ImageNet evaluation of pretrained models...")
        
        # Prepare ImageNet data
        dataloader = self.prepare_imagenet_data(imagenet_path, num_samples)
        
        results = {}
        
        # Evaluate both pretrained models
        models_to_evaluate = [
            (self.pretrained_eccv16, 'eccv16'),
            (self.pretrained_siggraph17, 'siggraph17')
        ]
        
        for model, model_name in models_to_evaluate:
            summary, detailed_metrics = self.evaluate_model(
                model, model_name, dataloader, num_samples
            )
            
            results[f"{model_name}_pretrained"] = {
                'summary': summary,
                'detailed_metrics': detailed_metrics
            }
        
        # Generate comparison report
        self.generate_imagenet_report(results)
        
        return results
    
    def generate_imagenet_report(self, results):
        """Generate comprehensive ImageNet evaluation report"""
        print("\n" + "="*80)
        print("IMAGENET PRETRAINED MODEL EVALUATION REPORT")
        print("="*80)
        
        # Performance Metrics Comparison
        print("\n📊 PERFORMANCE METRICS COMPARISON")
        print("-" * 80)
        print(f"{'Model':<20} {'Samples':<10} {'Success Rate':<15} {'PSNR':<8} {'SSIM':<8} {'Color Diff':<12}")
        print("-" * 80)
        
        for model_name in ['eccv16', 'siggraph17']:
            key = f"{model_name}_pretrained"
            if key in results:
                summary = results[key]['summary']
                if 'success_rate' in summary and 'psnr' in summary:
                    print(f"{model_name.upper():<20} {summary['total_samples']:<10} "
                          f"{summary['success_rate']:<15.3f} "
                          f"{summary['psnr']['mean']:<8.3f} {summary['ssim']['mean']:<8.3f} "
                          f"{summary['color_diff']['mean']:<12.3f}")
                else:
                    print(f"{model_name.upper():<20} {summary.get('total_samples', 0):<10} "
                          f"{summary.get('success_rate', 0):<15.3f} "
                          f"N/A{'':<5} N/A{'':<5} N/A{'':<8}")
        
        # Inference Time Comparison
        print("\n⚡ INFERENCE TIME COMPARISON")
        print("-" * 60)
        print(f"{'Model':<20} {'Avg Time (ms)':<15} {'Min (ms)':<10} {'Max (ms)':<10}")
        print("-" * 60)
        
        for model_name in ['eccv16', 'siggraph17']:
            key = f"{model_name}_pretrained"
            if key in results and 'inference_times' in results[key]['summary']:
                summary = results[key]['summary']
                print(f"{model_name.upper():<20} {summary['inference_times']['mean']:<15.2f} "
                      f"{summary['inference_times']['min']:<10.2f} {summary['inference_times']['max']:<10.2f}")
            else:
                print(f"{model_name.upper():<20} N/A{'':<12} N/A{'':<7} N/A{'':<7}")
        
        # Model Comparison
        print("\n🔍 MODEL COMPARISON")
        print("-" * 60)
        
        if 'eccv16_pretrained' in results and 'siggraph17_pretrained' in results:
            eccv16_summary = results['eccv16_pretrained']['summary']
            siggraph17_summary = results['siggraph17_pretrained']['summary']
            
            if 'psnr' in eccv16_summary and 'psnr' in siggraph17_summary:
                print(f"PSNR: ECCV16 ({eccv16_summary['psnr']['mean']:.3f}) vs SIGGRAPH17 ({siggraph17_summary['psnr']['mean']:.3f})")
                print(f"SSIM: ECCV16 ({eccv16_summary['ssim']['mean']:.3f}) vs SIGGRAPH17 ({siggraph17_summary['ssim']['mean']:.3f})")
                print(f"Color Diff: ECCV16 ({eccv16_summary['color_diff']['mean']:.3f}) vs SIGGRAPH17 ({siggraph17_summary['color_diff']['mean']:.3f})")
                print(f"Speed: ECCV16 ({eccv16_summary['inference_times']['mean']:.2f}ms) vs SIGGRAPH17 ({siggraph17_summary['inference_times']['mean']:.2f}ms)")
            else:
                print("Unable to compare models - no successful predictions")
        
        # Save detailed results
        self.save_imagenet_results(results)
        
        print("\n" + "="*80)
    
    def save_imagenet_results(self, results):
        """Save detailed ImageNet results to JSON"""
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
        
        # Add metadata
        results_json['metadata'] = {
            'dataset': 'ImageNet',
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'models_evaluated': ['eccv16_pretrained', 'siggraph17_pretrained']
        }
        
        with open('imagenet_pretrained_evaluation.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: imagenet_pretrained_evaluation.json")

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get ImageNet path from user
    imagenet_path = input("Enter the path to your ImageNet dataset: ").strip()
    
    if not os.path.exists(imagenet_path):
        print(f"❌ ImageNet path does not exist: {imagenet_path}")
        return
    
    # Get number of samples
    try:
        num_samples = int(input("Enter number of samples to evaluate (default 1000): ") or "1000")
    except ValueError:
        num_samples = 1000
    
    # Initialize evaluator
    evaluator = ImageNetPretrainedEvaluator(device=device)
    
    # Run evaluation
    results = evaluator.run_imagenet_evaluation(imagenet_path, num_samples)
    
    print("\n🎉 ImageNet evaluation completed!")
    print("Check the following files:")
    print("- imagenet_pretrained_evaluation.json")

if __name__ == "__main__":
    main()
