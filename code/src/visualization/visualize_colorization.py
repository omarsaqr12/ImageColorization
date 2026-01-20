#!/usr/bin/env python3
"""
Visual Comparison of Colorization Models on CIFAR-10
Saves colorized versions of 10 random images for visual inspection
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens

class ColorizationVisualizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.load_models()
        
    def load_models(self):
        """Load colorization models"""
        print("Loading colorization models...")
        self.colorizer_eccv16 = eccv16(pretrained=True).eval()
        self.colorizer_siggraph17 = siggraph17(pretrained=True).eval()
        
        if self.device != 'cpu':
            self.colorizer_eccv16 = self.colorizer_eccv16.to(self.device)
            self.colorizer_siggraph17 = self.colorizer_siggraph17.to(self.device)
        
        print("Models loaded successfully!")
    
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
    
    def load_cifar10_samples(self, num_samples=10):
        """Load random samples from CIFAR-10 test set"""
        print(f"Loading {num_samples} random CIFAR-10 samples...")
        
        # Load CIFAR-10 test set
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./cifar-10-python', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Get random indices
        total_samples = len(test_dataset)
        random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        samples = []
        for idx in random_indices:
            image, label = test_dataset[idx]
            class_name = test_dataset.classes[label]
            samples.append({
                'image': image,
                'label': label,
                'class_name': class_name,
                'index': idx
            })
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def create_visualization(self, samples, output_dir='colorization_results'):
        """Create side-by-side visualization of colorized images"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nCreating visualizations in '{output_dir}' directory...")
        
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}: {sample['class_name']}")
            
            # Get original image
            original_image = sample['image']
            original_np = original_image.numpy().transpose(1, 2, 0)
            
            # Convert to grayscale
            grayscale_image = transforms.Grayscale(num_output_channels=3)(original_image)
            
            # Colorize with both models
            colorized_eccv16 = self.colorize_image(grayscale_image, 'eccv16')
            colorized_siggraph17 = self.colorize_image(grayscale_image, 'siggraph17')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Sample {i+1}: {sample["class_name"]} (Index: {sample["index"]})', 
                        fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(original_np)
            axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Grayscale
            grayscale_np = grayscale_image.numpy().transpose(1, 2, 0)
            axes[0, 1].imshow(grayscale_np, cmap='gray')
            axes[0, 1].set_title('Grayscale Input', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # ECCV16 colorized
            axes[0, 2].imshow(colorized_eccv16)
            axes[0, 2].set_title('ECCV16 Colorized', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            # SIGGRAPH17 colorized
            axes[1, 0].imshow(colorized_siggraph17)
            axes[1, 0].set_title('SIGGRAPH17 Colorized', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Difference visualization (ECCV16 vs Original)
            diff_eccv16 = np.abs(original_np - colorized_eccv16)
            axes[1, 1].imshow(diff_eccv16)
            axes[1, 1].set_title('ECCV16 Difference', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            # Difference visualization (SIGGRAPH17 vs Original)
            diff_siggraph17 = np.abs(original_np - colorized_siggraph17)
            axes[1, 2].imshow(diff_siggraph17)
            axes[1, 2].set_title('SIGGRAPH17 Difference', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            
            # Save individual image
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'sample_{i+1:02d}_{sample["class_name"]}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also save individual colorized images
            eccv16_path = os.path.join(output_dir, f'sample_{i+1:02d}_eccv16.png')
            siggraph17_path = os.path.join(output_dir, f'sample_{i+1:02d}_siggraph17.png')
            
            plt.imsave(eccv16_path, colorized_eccv16)
            plt.imsave(siggraph17_path, colorized_siggraph17)
        
        print(f"\n✅ Visualizations saved to '{output_dir}' directory!")
        print(f"📁 Files created:")
        print(f"   - {len(samples)} comparison images (sample_XX_classname.png)")
        print(f"   - {len(samples)} ECCV16 colorized images (sample_XX_eccv16.png)")
        print(f"   - {len(samples)} SIGGRAPH17 colorized images (sample_XX_siggraph17.png)")
    
    def create_summary_grid(self, samples, output_dir='colorization_results'):
        """Create a summary grid showing all samples"""
        print("Creating summary grid...")
        
        # Create a large grid showing all samples
        fig, axes = plt.subplots(len(samples), 5, figsize=(20, 4*len(samples)))
        fig.suptitle('CIFAR-10 Colorization Comparison Summary', fontsize=20, fontweight='bold')
        
        for i, sample in enumerate(samples):
            # Get original image
            original_image = sample['image']
            original_np = original_image.numpy().transpose(1, 2, 0)
            
            # Convert to grayscale
            grayscale_image = transforms.Grayscale(num_output_channels=3)(original_image)
            
            # Colorize with both models
            colorized_eccv16 = self.colorize_image(grayscale_image, 'eccv16')
            colorized_siggraph17 = self.colorize_image(grayscale_image, 'siggraph17')
            
            # Original
            axes[i, 0].imshow(original_np)
            axes[i, 0].set_title(f'{sample["class_name"]}\nOriginal', fontsize=10)
            axes[i, 0].axis('off')
            
            # Grayscale
            grayscale_np = grayscale_image.numpy().transpose(1, 2, 0)
            axes[i, 1].imshow(grayscale_np, cmap='gray')
            axes[i, 1].set_title('Grayscale', fontsize=10)
            axes[i, 1].axis('off')
            
            # ECCV16
            axes[i, 2].imshow(colorized_eccv16)
            axes[i, 2].set_title('ECCV16', fontsize=10)
            axes[i, 2].axis('off')
            
            # SIGGRAPH17
            axes[i, 3].imshow(colorized_siggraph17)
            axes[i, 3].set_title('SIGGRAPH17', fontsize=10)
            axes[i, 3].axis('off')
            
            # Difference (SIGGRAPH17 - ECCV16)
            diff = np.abs(colorized_siggraph17 - colorized_eccv16)
            axes[i, 4].imshow(diff)
            axes[i, 4].set_title('Difference\n(SIGGRAPH17 - ECCV16)', fontsize=10)
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, 'summary_comparison.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary grid saved to: {summary_path}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize visualizer
    visualizer = ColorizationVisualizer(device=device)
    
    # Load random samples
    samples = visualizer.load_cifar10_samples(num_samples=10)
    
    # Create visualizations
    visualizer.create_visualization(samples)
    visualizer.create_summary_grid(samples)
    
    print("\n" + "="*60)
    print("🎨 COLORIZATION VISUALIZATION COMPLETE!")
    print("="*60)
    print("📁 Check the 'colorization_results' directory for:")
    print("   • Individual comparison images")
    print("   • Summary grid comparison")
    print("   • Individual colorized images from each model")
    print("\n💡 The summary grid shows:")
    print("   • Original CIFAR-10 images")
    print("   • Grayscale inputs")
    print("   • ECCV16 colorized results")
    print("   • SIGGRAPH17 colorized results")
    print("   • Difference between the two models")

if __name__ == "__main__":
    main()
