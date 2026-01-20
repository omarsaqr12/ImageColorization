#!/usr/bin/env python3
"""
ImageNet Evaluation Test Script
Tests the setup and basic functionality before running full evaluation
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add colorization module to path
sys.path.append('colorization')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from colorizers import eccv16, siggraph17
        print("✅ Colorization modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import colorization modules: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        from skimage import color
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import required packages: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if pretrained models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from colorizers import eccv16, siggraph17
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load models
        model_eccv16 = eccv16(pretrained=True).eval().to(device)
        model_siggraph17 = siggraph17(pretrained=True).eval().to(device)
        
        print("✅ ECCV16 model loaded successfully")
        print("✅ SIGGRAPH17 model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False

def test_image_processing():
    """Test basic image processing functionality"""
    print("\nTesting image processing...")
    
    try:
        from colorizers import preprocess_img, postprocess_tens
        
        # Create a dummy image
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test preprocessing
        tens_l_orig, tens_l_rs = preprocess_img(dummy_img, HW=(256, 256))
        
        print("✅ Image preprocessing works")
        
        # Test postprocessing
        dummy_ab = torch.zeros(1, 2, 256, 256)
        result = postprocess_tens(tens_l_orig, dummy_ab)
        
        print("✅ Image postprocessing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_imagenet_path(path):
    """Test if ImageNet path is valid"""
    print(f"\nTesting ImageNet path: {path}")
    
    if not os.path.exists(path):
        print(f"❌ Path does not exist: {path}")
        return False
    
    # Look for image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    image_count = 0
    
    for ext in image_extensions:
        import glob
        images = glob.glob(os.path.join(path, '**', ext), recursive=True)
        image_count += len(images)
    
    if image_count == 0:
        print(f"❌ No images found in {path}")
        print("Make sure the path contains subdirectories with image files")
        return False
    
    print(f"✅ Found {image_count} images in ImageNet dataset")
    return True

def test_sample_colorization():
    """Test colorization on a sample image"""
    print("\nTesting sample colorization...")
    
    try:
        from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        model_eccv16 = eccv16(pretrained=True).eval().to(device)
        
        # Create a test image
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Preprocess
        tens_l_orig, tens_l_rs = preprocess_img(test_img, HW=(256, 256))
        if device == 'cuda':
            tens_l_rs = tens_l_rs.cuda()
        
        # Colorize
        with torch.no_grad():
            predicted_ab = model_eccv16(tens_l_rs)
        
        # Postprocess
        colorized_img = postprocess_tens(tens_l_orig, predicted_ab.cpu())
        
        print("✅ Sample colorization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sample colorization test failed: {e}")
        return False

def main():
    print("ImageNet Evaluation Setup Test")
    print("="*40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        return
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed. Check colorization module.")
        return
    
    # Test image processing
    if not test_image_processing():
        print("\n❌ Image processing test failed.")
        return
    
    # Test sample colorization
    if not test_sample_colorization():
        print("\n❌ Sample colorization test failed.")
        return
    
    # Test ImageNet path (optional)
    imagenet_path = input("\nEnter ImageNet path to test (or press Enter to skip): ").strip()
    if imagenet_path:
        if not test_imagenet_path(imagenet_path):
            print("\n❌ ImageNet path test failed.")
            return
    
    print("\n" + "="*40)
    print("🎉 All tests passed! You're ready to run the evaluation.")
    print("="*40)
    print("\nNext steps:")
    print("1. Run: python evaluate_imagenet_pretrained.py")
    print("2. Run: python visualize_imagenet_pretrained.py")
    print("3. Run: python analyze_imagenet_results.py")

if __name__ == "__main__":
    main()
