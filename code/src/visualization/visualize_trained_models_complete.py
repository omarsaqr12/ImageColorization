#!/usr/bin/env python3
"""
Complete Visualize Trained Colorization Models Script
Copy this entire file to replace your current visualize_trained_models.py
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
from skimage import color

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens

class TrainedModelVisualizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.load_models()
        
    def load_models(self):
        """Load both pre-trained and trained models"""
        print("Loading colorization models...")
        
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
            print("❌ Trained ECCV16 model not found")
            
        if os.path.exists('siggraph17_best_model.pth'):
            self.trained_siggraph17.load_state_dict(torch.load('siggraph17_best_model.pth', map_location=self.device))
            print("✅ Loaded trained SIGGRAPH17 model")
        else:
            print("❌ Trained SIGGRAPH17 model not found")
        
        print("✅ All models loaded successfully!")
    
    def rgb_to_lab(self, rgb_tensor):
        """Convert RGB tensor to LAB tensor"""
        # Ensure tensor is 4D (batch, channels, height, width)
        if rgb_tensor.dim() == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)  # Add batch dimension
        
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
    
    def colorize_with_pretrained(self, grayscale_image, model):
        """Colorize using pre-trained model (original method)"""
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
        with torch.no_grad():
            colorized_ab = model(tens_l_rs).cpu()
        
        # Post-process
        colorized_image = postprocess_tens(tens_l_orig, colorized_ab)
        
        return colorized_image
    
    def colorize_with_trained(self, grayscale_image, model):
        """Colorize using trained model (LAB method)"""
        # Ensure grayscale_image is a 4D tensor (batch, channels, height, width)
        if grayscale_image.dim() == 3:
            grayscale_image = grayscale_image.unsqueeze(0)  # Add batch dimension
        
        # Convert RGB to LAB
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
        
        print(f"✅ Loaded {len(samples)} samples")
        return samples
    
    def create_simple_comparison(self, samples, output_dir='trained_model_results'):
        """Create simple comparison visualization"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n🎨 Creating simple comparisons in '{output_dir}' directory...")
        
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}: {sample['class_name']}")
            
            try:
                # Get original image
                original_image = sample['image']
                original_np = original_image.numpy().transpose(1, 2, 0)
                
                # Convert to grayscale
                grayscale_image = transforms.Grayscale(num_output_channels=3)(original_image)
                
                # Colorize with all models
                pretrained_eccv16 = self.colorize_with_pretrained(grayscale_image, self.pretrained_eccv16)
                pretrained_siggraph17 = self.colorize_with_pretrained(grayscale_image, self.pretrained_siggraph17)
                trained_eccv16 = self.colorize_with_trained(grayscale_image, self.trained_eccv16)
                trained_siggraph17 = self.colorize_with_trained(grayscale_image, self.trained_siggraph17)
                
                # Create simple comparison figure
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Sample {i+1}: {sample["class_name"]} - Trained vs Pre-trained', 
                            fontsize=16, fontweight='bold')
                
                # Row 1: Original, Grayscale, Pre-trained ECCV16
                axes[0, 0].imshow(original_np)
                axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                grayscale_np = grayscale_image.numpy().transpose(1, 2, 0)
                axes[0, 1].imshow(grayscale_np, cmap='gray')
                axes[0, 1].set_title('Grayscale Input', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(pretrained_eccv16)
                axes[0, 2].set_title('Pre-trained ECCV16', fontsize=12, fontweight='bold')
                axes[0, 2].axis('off')
                
                # Row 2: Trained ECCV16, Pre-trained SIGGRAPH17, Trained SIGGRAPH17
                axes[1, 0].imshow(trained_eccv16)
                axes[1, 0].set_title('Trained ECCV16', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pretrained_siggraph17)
                axes[1, 1].set_title('Pre-trained SIGGRAPH17', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(trained_siggraph17)
                axes[1, 2].set_title('Trained SIGGRAPH17', fontsize=12, fontweight='bold')
                axes[1, 2].axis('off')
                
                # Save comparison
                plt.tight_layout()
                output_path = os.path.join(output_dir, f'comparison_{i+1:02d}_{sample["class_name"]}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save individual colorized images
                plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_pretrained_eccv16.png'), pretrained_eccv16)
                plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_trained_eccv16.png'), trained_eccv16)
                plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_pretrained_siggraph17.png'), pretrained_siggraph17)
                plt.imsave(os.path.join(output_dir, f'sample_{i+1:02d}_trained_siggraph17.png'), trained_siggraph17)
                
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {e}")
                continue
        
        print(f"\n✅ Comparisons saved to '{output_dir}' directory!")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize visualizer
    visualizer = TrainedModelVisualizer(device=device)
    
    # Load random samples
    samples = visualizer.load_cifar10_samples(num_samples=10)
    
    # Create visualizations
    visualizer.create_simple_comparison(samples)
    
    print("\n" + "="*70)
    print("🎨 TRAINED MODEL VISUALIZATION COMPLETE!")
    print("="*70)
    print("📁 Check the 'trained_model_results' directory for:")
    print("   • Comparison images (2x3 grids)")
    print("   • Individual colorized images from each model")
    print("\n💡 The visualizations show:")
    print("   • Original CIFAR-10 images")
    print("   • Grayscale inputs")
    print("   • Pre-trained vs Trained model results")

if __name__ == "__main__":
    main()
