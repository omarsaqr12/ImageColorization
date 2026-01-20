#!/usr/bin/env python3
"""
ImageNet Trained vs Pretrained Visualization Script
Creates visual comparisons of trained and pretrained models on ImageNet samples
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

class ImageNetTrainedVisualizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.load_models()
        
    def load_models(self):
        """Load both pretrained and trained models"""
        print("Loading colorization models...")
        
        # Load pretrained models
        self.pretrained_eccv16 = eccv16(pretrained=True).eval().to(self.device)
        self.pretrained_siggraph17 = siggraph17(pretrained=True).eval().to(self.device)
        
        # Load trained models
        self.trained_eccv16 = eccv16(pretrained=False).eval().to(self.device)
        self.trained_siggraph17 = siggraph17(pretrained=False).eval().to(self.device)
        
        # Load trained weights if they exist
        if os.path.exists('eccv16_imagenet_best_model.pth'):
            self.trained_eccv16.load_state_dict(torch.load('eccv16_imagenet_best_model.pth', map_location=self.device))
            print("✅ Loaded trained ECCV16 model")
        else:
            print("❌ Trained ECCV16 model not found")
            
        if os.path.exists('siggraph17_imagenet_best_model.pth'):
            self.trained_siggraph17.load_state_dict(torch.load('siggraph17_imagenet_best_model.pth', map_location=self.device))
            print("✅ Loaded trained SIGGRAPH17 model")
        else:
            print("❌ Trained SIGGRAPH17 model not found")
        
        print("✅ All models loaded successfully!")
    
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
        """Colorize image using specified model with proper image loading"""
        try:
            # Load image as numpy array first
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # Use preprocess_img with the numpy array
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
    
    def create_trained_vs_pretrained_comparison(self, samples, output_dir='imagenet_trained_results'):
        """Create comparison visualization for trained vs pretrained models"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n🎨 Creating trained vs pretrained comparisons in '{output_dir}' directory...")
        
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
                
                # Colorize with all models
                pretrained_eccv16 = self.colorize_image(image_path, self.pretrained_eccv16, 'Pretrained ECCV16')
                trained_eccv16 = self.colorize_image(image_path, self.trained_eccv16, 'Trained ECCV16')
                pretrained_siggraph17 = self.colorize_image(image_path, self.pretrained_siggraph17, 'Pretrained SIGGRAPH17')
                trained_siggraph17 = self.colorize_image(image_path, self.trained_siggraph17, 'Trained SIGGRAPH17')
                
                if all(img is not None for img in [pretrained_eccv16, trained_eccv16, pretrained_siggraph17, trained_siggraph17]):
                    # Create comparison figure
                    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
                    fig.suptitle(f'ImageNet Sample {i+1}: {sample["class_name"]} - Trained vs Pretrained', 
                                fontsize=16, fontweight='bold')
                    
                    # Row 1: Original, Grayscale, Pretrained ECCV16
                    axes[0, 0].imshow(original_np)
                    axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(grayscale_np, cmap='gray')
                    axes[0, 1].set_title('Grayscale Input', fontsize=14, fontweight='bold')
                    axes[0, 1].axis('off')
                    
                    axes[0, 2].imshow(pretrained_eccv16)
                    axes[0, 2].set_title('Pretrained ECCV16', fontsize=14, fontweight='bold')
                    axes[0, 2].axis('off')
                    
                    # Row 2: Trained ECCV16, Pretrained SIGGRAPH17, Trained SIGGRAPH17
                    axes[1, 0].imshow(trained_eccv16)
                    axes[1, 0].set_title('Trained ECCV16', fontsize=14, fontweight='bold')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(pretrained_siggraph17)
                    axes[1, 1].set_title('Pretrained SIGGRAPH17', fontsize=14, fontweight='bold')
                    axes[1, 1].axis('off')
                    
                    axes[1, 2].imshow(trained_siggraph17)
                    axes[1, 2].set_title('Trained SIGGRAPH17', fontsize=14, fontweight='bold')
                    axes[1, 2].axis('off')
                    
                    # Row 3: Side-by-side comparisons
                    # ECCV16 comparison
                    eccv16_comparison = np.concatenate([pretrained_eccv16, trained_eccv16], axis=1)
                    axes[2, 0].imshow(eccv16_comparison)
                    axes[2, 0].set_title('ECCV16: Pretrained | Trained', fontsize=12)
                    axes[2, 0].axis('off')
                    
                    # SIGGRAPH17 comparison
                    siggraph17_comparison = np.concatenate([pretrained_siggraph17, trained_siggraph17], axis=1)
                    axes[2, 1].imshow(siggraph17_comparison)
                    axes[2, 1].set_title('SIGGRAPH17: Pretrained | Trained', fontsize=12)
                    axes[2, 1].axis('off')
                    
                    # Best comparison (pretrained vs trained)
                    best_comparison = np.concatenate([pretrained_siggraph17, trained_siggraph17], axis=1)
                    axes[2, 2].imshow(best_comparison)
                    axes[2, 2].set_title('Best Models: Pretrained | Trained', fontsize=12)
                    axes[2, 2].axis('off')
                    
                    # Save comparison
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'trained_vs_pretrained_{i+1:02d}_{sample["class_name"]}.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Save individual colorized images
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_pretrained_eccv16.png'), pretrained_eccv16)
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_trained_eccv16.png'), trained_eccv16)
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_pretrained_siggraph17.png'), pretrained_siggraph17)
                    plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_trained_siggraph17.png'), trained_siggraph17)
                    
                    successful_comparisons += 1
                    
                else:
                    print(f"❌ Failed to colorize sample {i+1}")
                    
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {e}")
                continue
        
        print(f"\n✅ Created {successful_comparisons} successful comparisons!")
        return successful_comparisons
    
    def create_model_comparison_grid(self, samples, output_dir='imagenet_trained_results'):
        """Create a grid comparison of all models"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n📊 Creating model comparison grid...")
        
        # Select a subset of samples for the grid
        grid_samples = samples[:12]  # Show 12 examples in a 3x4 grid
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('ImageNet Colorization: Trained vs Pretrained Models', 
                    fontsize=20, fontweight='bold')
        
        for i, sample in enumerate(grid_samples):
            row = i // 4
            col = i % 4
            
            try:
                image_path = sample['image_path']
                
                # Colorize with trained SIGGRAPH17 (best model)
                trained_siggraph17_result = self.colorize_image(image_path, self.trained_siggraph17, 'Trained SIGGRAPH17')
                
                if trained_siggraph17_result is not None:
                    # Show trained SIGGRAPH17 result
                    axes[row, col].imshow(trained_siggraph17_result)
                    axes[row, col].set_title(f'{sample["class_name"]}\nTrained SIGGRAPH17', fontsize=10)
                    axes[row, col].axis('off')
                else:
                    axes[row, col].text(0.5, 0.5, 'Failed', ha='center', va='center')
                    axes[row, col].axis('off')
                    
            except Exception as e:
                print(f"Error in grid visualization: {e}")
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row, col].axis('off')
        
        # Save grid
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'imagenet_trained_models_grid.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Model comparison grid saved!")
    
    def create_improvement_visualization(self, samples, output_dir='imagenet_trained_results'):
        """Create visualization showing improvements from training"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n📈 Creating improvement visualization...")
        
        # Select best examples for improvement comparison
        improvement_samples = samples[:6]
        
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle('ImageNet Training Improvements: Before vs After', 
                    fontsize=20, fontweight='bold')
        
        for i, sample in enumerate(improvement_samples):
            try:
                image_path = sample['image_path']
                
                # Get pretrained and trained results
                pretrained_result = self.colorize_image(image_path, self.pretrained_siggraph17, 'Pretrained SIGGRAPH17')
                trained_result = self.colorize_image(image_path, self.trained_siggraph17, 'Trained SIGGRAPH17')
                
                if pretrained_result is not None and trained_result is not None:
                    # Top row: Pretrained results
                    axes[0, i].imshow(pretrained_result)
                    axes[0, i].set_title(f'{sample["class_name"]}\nPretrained', fontsize=12)
                    axes[0, i].axis('off')
                    
                    # Bottom row: Trained results
                    axes[1, i].imshow(trained_result)
                    axes[1, i].set_title(f'{sample["class_name"]}\nTrained', fontsize=12)
                    axes[1, i].axis('off')
                    
            except Exception as e:
                print(f"Error in improvement visualization: {e}")
                axes[0, i].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[1, i].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        # Save improvement visualization
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'imagenet_training_improvements.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Improvement visualization saved!")

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
    visualizer = ImageNetTrainedVisualizer(device=device)
    
    # Load samples
    samples = visualizer.load_imagenet_samples(imagenet_path, num_samples)
    
    if len(samples) == 0:
        print("❌ No samples loaded. Exiting.")
        return
    
    # Create visualizations
    successful_comparisons = visualizer.create_trained_vs_pretrained_comparison(samples)
    visualizer.create_model_comparison_grid(samples)
    visualizer.create_improvement_visualization(samples)
    
    print("\n" + "="*70)
    print("🎨 IMAGENET TRAINED vs PRETRAINED VISUALIZATION COMPLETE!")
    print("="*70)
    print("📁 Check the 'imagenet_trained_results' directory for:")
    print("   • Trained vs pretrained comparison images (3x3 grids)")
    print("   • Model comparison grids")
    print("   • Training improvement visualizations")
    print("   • Individual colorized images from each model")
    print(f"\n✅ Successfully processed {successful_comparisons} samples!")

if __name__ == "__main__":
    main()
