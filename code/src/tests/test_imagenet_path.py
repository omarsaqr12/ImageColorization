#!/usr/bin/env python3
"""
ImageNet Pretrained Model Evaluation Script - Direct Path Version
"""

import os
import sys
import time
import numpy as np
import torch

# Add colorization module to path
sys.path.append('colorization')

def main():
    print("ImageNet Pretrained Model Evaluation")
    print("="*50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set the ImageNet path
    imagenet_path = "/home/mohab/Downloads/work/colorization/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/"
    
    print(f"ImageNet path: {imagenet_path}")
    
    # Check if path exists
    if not os.path.exists(imagenet_path):
        print(f"❌ ImageNet path does not exist: {imagenet_path}")
        print("\nThis is expected if you're running on Windows but the dataset is on Linux.")
        print("You have a few options:")
        print("1. Run this script on the Linux machine where the dataset is located")
        print("2. Copy the dataset to your Windows machine")
        print("3. Use a network mount or shared folder")
        print("4. Use WSL (Windows Subsystem for Linux) to access the Linux filesystem")
        return
    
    try:
        # Try to import colorization modules
        print("\nLoading colorization models...")
        from colorizers import eccv16, siggraph17
        
        # Load models
        pretrained_eccv16 = eccv16(pretrained=True).eval().to(device)
        pretrained_siggraph17 = siggraph17(pretrained=True).eval().to(device)
        
        print("✅ Models loaded successfully!")
        
        # Check for images in the dataset
        import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
        total_images = 0
        
        for ext in image_extensions:
            images = glob.glob(os.path.join(imagenet_path, '**', ext), recursive=True)
            total_images += len(images)
        
        print(f"✅ Found {total_images} images in ImageNet dataset")
        
        if total_images > 0:
            print("\n🎉 Everything looks good! You can now run the full evaluation.")
            print("To run the evaluation, use:")
            print("python evaluate_imagenet_simple.py")
        else:
            print("❌ No images found in the dataset path")
            
    except ImportError as e:
        print(f"❌ Failed to import colorization modules: {e}")
        print("Make sure the colorization module is in the correct path")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
