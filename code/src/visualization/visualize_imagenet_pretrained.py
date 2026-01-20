#!/usr/bin/env python3
"""
ImageNet Colorization Visualization Script
Creates visual comparisons of pretrained models on ImageNet samples
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import json
import glob
from pathlib import Path

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens

class ImageNetVisualizer:
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
    
    def load_imagenet_samples(self, imagenet_path, num_samples=20):
        """Load random samples from ImageNet dataset"""
        print(f"Loading {num_samples} random ImageNet samples...")
        
        # Find all image files
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            image_paths.extend(glob.glob(os.path.join(imagenet_path, '**', ext), recursive=True))
        
        if len(image_paths) == 0:
            print(f"❌ No images found in {imagenet_path}")
            return []
        
        # Get random samples
        if len(image_paths) > num_samples:
            image_paths = random.sample(image_paths, num_samples)
        
        samples = []
        for image_path in image_paths:
            try:
                # Get class name from directory structure
                class_name = os.path.basename(os.path.dirname(image_path))
                samples.append({
                    'image_path': image_path,
                    'class_name': class_name
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(samples)} samples")
        return samples
    
    def colorize_image(self, image_path, model, model_name):
        """Colorize image using pretrained model with proper image loading"""
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
    
    def create_imagenet_comparison(self, samples, output_dir='imagenet_results'):
        """Create comparison visualization for ImageNet samples"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n🎨 Creating ImageNet comparisons in '{output_dir}' directory...")
        
        successful_comparisons = 0
        
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}: {sample['class_name']}")
            
            try:
                image_path = sample['image_path']
                
                # Load original image
                original_img = Image.open(image_path).convert('RGB')
                original_img = original_img.resize((256, 256))
                original_np = np.array(original_img) / 255.0
                
                # Create grayscale version
                grayscale_img = Image.open(image_path).convert('L').convert('RGB')
                grayscale_img = grayscale_img.resize((256, 256))
                grayscale_np = np.array(grayscale_img) / 255.0
                
                # Colorize with both models
                eccv16_result = self.colorize_image(image_path, self.pretrained_eccv16, 'ECCV16')
                siggraph17_result = self.colorize_image(image_path, self.pretrained_siggraph17, 'SIGGRAPH17')
                
                if eccv16_result is not None and siggraph17_result is not None:
                    # Create comparison figure
                    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                    fig.suptitle(f'ImageNet Sample {i+1}: {sample["class_name"]}', 
                                fontsize=16, fontweight='bold')
                    
                    # Original image
                    axes[0, 0].imshow(original_np)
                    axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
                    axes[0, 0].axis('off')
                    
                    # Grayscale input
                    axes[0, 1].imshow(grayscale_np, cmap='gray')
                    axes[0, 1].set_title('Grayscale Input', fontsize=14, fontweight='bold')
                    axes[0, 1].axis('off')
                    
                    # ECCV16 result
                    axes[1, 0].imshow(eccv16_result)
                    axes[1, 0].set_title('ECCV16 Colorization', fontsize=14, fontweight='bold')
                    axes[1, 0].axis('off')
                    
                    # SIGGRAPH17 result
                    axes[1, 1].imshow(siggraph17_result)
                    axes[1, 1].set_title('SIGGRAPH17 Colorization', fontsize=14, fontweight='bold')
                    axes[1, 1].axis('off')
                    
                    # Save comparison
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'imagenet_comparison_{i+1:02d}_{sample["class_name"]}.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Save individual colorized images
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_eccv16.png'), eccv16_result)
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_siggraph17.png'), siggraph17_result)
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_original.png'), original_np)
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_grayscale.png'), grayscale_np)
                    
                    successful_comparisons += 1
                    
                else:
                    print(f"❌ Failed to colorize sample {i+1}")
                    
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {e}")
                continue
        
        print(f"\n✅ Created {successful_comparisons} successful comparisons!")
        return successful_comparisons
    
    def create_class_comparison(self, samples, output_dir='imagenet_results'):
        """Create comparison grouped by ImageNet classes"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n📊 Creating class-based comparisons...")
        
        # Group samples by class
        class_groups = {}
        for sample in samples:
            class_name = sample['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(sample)
        
        # Create comparison for each class (max 3 samples per class)
        for class_name, class_samples in class_groups.items():
            if len(class_samples) == 0:
                continue
                
            # Take up to 3 samples from this class
            selected_samples = class_samples[:3]
            
            fig, axes = plt.subplots(len(selected_samples), 4, figsize=(16, 4 * len(selected_samples)))
            if len(selected_samples) == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'ImageNet Class: {class_name}', fontsize=16, fontweight='bold')
            
            for i, sample in enumerate(selected_samples):
                try:
                    image_path = sample['image_path']
                    
                    # Load original image
                    original_img = Image.open(image_path).convert('RGB')
                    original_img = original_img.resize((256, 256))
                    original_np = np.array(original_img) / 255.0
                    
                    # Colorize with both models
                    eccv16_result = self.colorize_image(image_path, self.pretrained_eccv16, 'ECCV16')
                    siggraph17_result = self.colorize_image(image_path, self.pretrained_siggraph17, 'SIGGRAPH17')
                    
                    if eccv16_result is not None and siggraph17_result is not None:
                        # Original
                        axes[i, 0].imshow(original_np)
                        axes[i, 0].set_title('Original', fontsize=12)
                        axes[i, 0].axis('off')
                        
                        # ECCV16
                        axes[i, 1].imshow(eccv16_result)
                        axes[i, 1].set_title('ECCV16', fontsize=12)
                        axes[i, 1].axis('off')
                        
                        # SIGGRAPH17
                        axes[i, 2].imshow(siggraph17_result)
                        axes[i, 2].set_title('SIGGRAPH17', fontsize=12)
                        axes[i, 2].axis('off')
                        
                        # Side-by-side comparison
                        combined = np.concatenate([eccv16_result, siggraph17_result], axis=1)
                        axes[i, 3].imshow(combined)
                        axes[i, 3].set_title('ECCV16 | SIGGRAPH17', fontsize=12)
                        axes[i, 3].axis('off')
                        
                except Exception as e:
                    print(f"Error processing class sample: {e}")
                    continue
            
            # Save class comparison
            plt.tight_layout()
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = os.path.join(output_dir, f'class_comparison_{safe_class_name}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Created class-based comparisons!")
    
    def create_summary_visualization(self, samples, output_dir='imagenet_results'):
        """Create a summary visualization with best examples"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n📈 Creating summary visualization...")
        
        # Select a subset of samples for summary
        summary_samples = samples[:12]  # Show 12 examples in a 3x4 grid
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('ImageNet Colorization Summary - ECCV16 vs SIGGRAPH17', 
                    fontsize=20, fontweight='bold')
        
        for i, sample in enumerate(summary_samples):
            row = i // 4
            col = i % 4
            
            try:
                image_path = sample['image_path']
                
                # Load original image
                original_img = Image.open(image_path).convert('RGB')
                original_img = original_img.resize((256, 256))
                original_np = np.array(original_img) / 255.0
                
                # Colorize with ECCV16
                eccv16_result = self.colorize_image(image_path, self.pretrained_eccv16, 'ECCV16')
                
                if eccv16_result is not None:
                    # Show ECCV16 result
                    axes[row, col].imshow(eccv16_result)
                    axes[row, col].set_title(f'{sample["class_name"]}\nECCV16', fontsize=10)
                    axes[row, col].axis('off')
                else:
                    axes[row, col].text(0.5, 0.5, 'Failed', ha='center', va='center')
                    axes[row, col].axis('off')
                    
            except Exception as e:
                print(f"Error in summary visualization: {e}")
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row, col].axis('off')
        
        # Save summary
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'imagenet_summary_eccv16.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create SIGGRAPH17 summary
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('ImageNet Colorization Summary - SIGGRAPH17', 
                    fontsize=20, fontweight='bold')
        
        for i, sample in enumerate(summary_samples):
            row = i // 4
            col = i % 4
            
            try:
                image_path = sample['image_path']
                
                # Colorize with SIGGRAPH17
                siggraph17_result = self.colorize_image(image_path, self.pretrained_siggraph17, 'SIGGRAPH17')
                
                if siggraph17_result is not None:
                    # Show SIGGRAPH17 result
                    axes[row, col].imshow(siggraph17_result)
                    axes[row, col].set_title(f'{sample["class_name"]}\nSIGGRAPH17', fontsize=10)
                    axes[row, col].axis('off')
                else:
                    axes[row, col].text(0.5, 0.5, 'Failed', ha='center', va='center')
                    axes[row, col].axis('off')
                    
            except Exception as e:
                print(f"Error in summary visualization: {e}")
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row, col].axis('off')
        
        # Save SIGGRAPH17 summary
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'imagenet_summary_siggraph17.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created summary visualizations!")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
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
        num_samples = int(input("Enter number of samples to visualize (default 20): ") or "20")
    except ValueError:
        num_samples = 20
    
    # Initialize visualizer
    visualizer = ImageNetVisualizer(device=device)
    
    # Load samples
    samples = visualizer.load_imagenet_samples(imagenet_path, num_samples)
    
    if len(samples) == 0:
        print("❌ No samples loaded. Exiting.")
        return
    
    # Create visualizations
    successful_comparisons = visualizer.create_imagenet_comparison(samples)
    visualizer.create_class_comparison(samples)
    visualizer.create_summary_visualization(samples)
    
    print("\n" + "="*70)
    print("🎨 IMAGENET COLORIZATION VISUALIZATION COMPLETE!")
    print("="*70)
    print("📁 Check the 'imagenet_results' directory for:")
    print("   • Individual comparison images (2x2 grids)")
    print("   • Class-based comparisons")
    print("   • Summary visualizations")
    print("   • Individual colorized images")
    print(f"\n✅ Successfully processed {successful_comparisons} samples!")

if __name__ == "__main__":
    main()
