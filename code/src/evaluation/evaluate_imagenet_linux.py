#!/usr/bin/env python3
"""
ImageNet Evaluation Script for Linux Machine
Run this on the Linux machine where your ImageNet dataset is located
"""

import os
import sys
import time
import numpy as np
import torch
import json
import glob
import random
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17, preprocess_img, postprocess_tens

def evaluate_imagenet_models(imagenet_path, num_samples=100):
    """Evaluate pretrained models on ImageNet"""
    
    print("ImageNet Pretrained Model Evaluation")
    print("="*50)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if path exists
    if not os.path.exists(imagenet_path):
        print(f"❌ ImageNet path does not exist: {imagenet_path}")
        return
    
    print(f"ImageNet path: {imagenet_path}")
    print(f"Number of samples: {num_samples}")
    
    # Load models
    print("\nLoading pretrained models...")
    pretrained_eccv16 = eccv16(pretrained=True).eval().to(device)
    pretrained_siggraph17 = siggraph17(pretrained=True).eval().to(device)
    print("✅ Models loaded successfully!")
    
    # Find images
    print("\nFinding images...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    all_images = []
    
    for ext in image_extensions:
        images = glob.glob(os.path.join(imagenet_path, '**', ext), recursive=True)
        all_images.extend(images)
    
    print(f"Found {len(all_images)} images")
    
    if len(all_images) == 0:
        print("❌ No images found!")
        return
    
    # Sample images
    if len(all_images) > num_samples:
        sample_images = random.sample(all_images, num_samples)
    else:
        sample_images = all_images
    
    print(f"Evaluating {len(sample_images)} images...")
    
    # Evaluation results
    results = {
        'eccv16': {'psnr': [], 'ssim': [], 'times': []},
        'siggraph17': {'psnr': [], 'ssim': [], 'times': []}
    }
    
    successful_count = 0
    
    for i, image_path in enumerate(sample_images):
        try:
            print(f"Processing {i+1}/{len(sample_images)}: {os.path.basename(image_path)}")
            
            # Load original image
            original_img = Image.open(image_path).convert('RGB')
            original_img = original_img.resize((256, 256))
            original_np = np.array(original_img) / 255.0
            
            # Evaluate ECCV16
            start_time = time.time()
            try:
                img = preprocess_img(image_path, HW=(256, 256))
                tens_l_orig, tens_l_rs = img
                if device == 'cuda':
                    tens_l_rs = tens_l_rs.cuda()
                
                with torch.no_grad():
                    predicted_ab = pretrained_eccv16(tens_l_rs)
                
                eccv16_result = postprocess_tens(tens_l_orig, predicted_ab.cpu())
                eccv16_time = (time.time() - start_time) * 1000
                
                # Calculate metrics for ECCV16
                eccv16_psnr = psnr(original_np, eccv16_result, data_range=1.0)
                eccv16_ssim = ssim(original_np, eccv16_result, multichannel=True, data_range=1.0, channel_axis=2)
                
                results['eccv16']['psnr'].append(eccv16_psnr)
                results['eccv16']['ssim'].append(eccv16_ssim)
                results['eccv16']['times'].append(eccv16_time)
                
            except Exception as e:
                print(f"  ECCV16 failed: {e}")
                continue
            
            # Evaluate SIGGRAPH17
            start_time = time.time()
            try:
                with torch.no_grad():
                    predicted_ab = pretrained_siggraph17(tens_l_rs)
                
                siggraph17_result = postprocess_tens(tens_l_orig, predicted_ab.cpu())
                siggraph17_time = (time.time() - start_time) * 1000
                
                # Calculate metrics for SIGGRAPH17
                siggraph17_psnr = psnr(original_np, siggraph17_result, data_range=1.0)
                siggraph17_ssim = ssim(original_np, siggraph17_result, multichannel=True, data_range=1.0, channel_axis=2)
                
                results['siggraph17']['psnr'].append(siggraph17_psnr)
                results['siggraph17']['ssim'].append(siggraph17_ssim)
                results['siggraph17']['times'].append(siggraph17_time)
                
            except Exception as e:
                print(f"  SIGGRAPH17 failed: {e}")
                continue
            
            successful_count += 1
            
            if successful_count % 10 == 0:
                print(f"  ✅ Successfully processed {successful_count} images")
                
        except Exception as e:
            print(f"  ❌ Error processing {image_path}: {e}")
            continue
    
    # Calculate summary statistics
    print(f"\n📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Successfully processed: {successful_count}/{len(sample_images)} images")
    
    for model_name in ['eccv16', 'siggraph17']:
        if len(results[model_name]['psnr']) > 0:
            psnr_mean = np.mean(results[model_name]['psnr'])
            ssim_mean = np.mean(results[model_name]['ssim'])
            time_mean = np.mean(results[model_name]['times'])
            
            print(f"\n{model_name.upper()} Results:")
            print(f"  PSNR: {psnr_mean:.3f} ± {np.std(results[model_name]['psnr']):.3f}")
            print(f"  SSIM: {ssim_mean:.3f} ± {np.std(results[model_name]['ssim']):.3f}")
            print(f"  Time: {time_mean:.2f} ± {np.std(results[model_name]['times']):.2f} ms")
    
    # Save results
    results_summary = {
        'metadata': {
            'dataset': 'ImageNet',
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device,
            'total_images': len(sample_images),
            'successful_images': successful_count
        },
        'results': results
    }
    
    with open('imagenet_evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📄 Results saved to: imagenet_evaluation_results.json")
    print("🎉 Evaluation completed!")

def main():
    # Set the ImageNet path (override via the IMAGENET_VAL_ROOT environment variable)
    imagenet_path = os.environ.get("IMAGENET_VAL_ROOT", "./data/imagenet/ILSVRC/Data/CLS-LOC/")
    
    # Number of samples to evaluate
    num_samples = 100  # Start with 100 for testing
    
    evaluate_imagenet_models(imagenet_path, num_samples)

if __name__ == "__main__":
    main()
