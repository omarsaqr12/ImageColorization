#!/usr/bin/env python3
"""
Apple Image Colorization Script
Uses the ECCV 2016 and SIGGRAPH 2017 colorization models
"""

import argparse
import matplotlib.pyplot as plt
import os
import sys

# Add the colorization directory to the path
sys.path.append('colorization')

from colorizers import *

def main():
    # Set up paths
    input_image = "close-up-delicious-apple_23-2151868338.jpg"
    output_prefix = "apple_colorized"
    
    print("Loading colorization models...")
    
    # Load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    
    print("Processing image...")
    
    # Load and preprocess the image
    img = load_img(input_image)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    
    # Generate colorized outputs
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    
    # Save outputs
    plt.imsave(f'{output_prefix}_eccv16.png', out_img_eccv16)
    plt.imsave(f'{output_prefix}_siggraph17.png', out_img_siggraph17)
    
    print(f"Colorized images saved as:")
    print(f"- {output_prefix}_eccv16.png (ECCV 2016 model)")
    print(f"- {output_prefix}_siggraph17.png (SIGGRAPH 2017 model)")
    
    # Display results
    plt.figure(figsize=(15,10))
    
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2,3,2)
    plt.imshow(img_bw)
    plt.title('Grayscale Input', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2,3,3)
    plt.imshow(out_img_eccv16)
    plt.title('ECCV 2016 Model', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2,3,4)
    plt.imshow(out_img_siggraph17)
    plt.title('SIGGRAPH 2017 Model', fontsize=12)
    plt.axis('off')
    
    # Add comparison
    plt.subplot(2,3,5)
    plt.imshow(out_img_eccv16)
    plt.title('ECCV 2016 (Detail)', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2,3,6)
    plt.imshow(out_img_siggraph17)
    plt.title('SIGGRAPH 2017 (Detail)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('apple_colorization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison image saved as: apple_colorization_comparison.png")

if __name__ == "__main__":
    main()
