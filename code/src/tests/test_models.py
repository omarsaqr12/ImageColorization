#!/usr/bin/env python3
"""
Simple Trained Model Test
Quick test to verify trained models work
"""

import os
import sys
import torch

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17

def test_trained_models():
    print("Testing trained models...")
    
    # Check if model files exist
    eccv16_exists = os.path.exists('eccv16_best_model.pth')
    siggraph17_exists = os.path.exists('siggraph17_best_model.pth')
    
    print(f"ECCV16 model file exists: {eccv16_exists}")
    print(f"SIGGRAPH17 model file exists: {siggraph17_exists}")
    
    if eccv16_exists:
        print(f"ECCV16 file size: {os.path.getsize('eccv16_best_model.pth') / (1024*1024):.1f} MB")
    
    if siggraph17_exists:
        print(f"SIGGRAPH17 file size: {os.path.getsize('siggraph17_best_model.pth') / (1024*1024):.1f} MB")
    
    # Try to load models
    try:
        print("\nLoading pre-trained ECCV16...")
        pretrained_eccv16 = eccv16(pretrained=True)
        print("✅ Pre-trained ECCV16 loaded")
        
        print("Loading pre-trained SIGGRAPH17...")
        pretrained_siggraph17 = siggraph17(pretrained=True)
        print("✅ Pre-trained SIGGRAPH17 loaded")
        
        if eccv16_exists:
            print("\nLoading trained ECCV16...")
            trained_eccv16 = eccv16(pretrained=False)
            trained_eccv16.load_state_dict(torch.load('eccv16_best_model.pth', map_location='cpu'))
            print("✅ Trained ECCV16 loaded")
        
        if siggraph17_exists:
            print("Loading trained SIGGRAPH17...")
            trained_siggraph17 = siggraph17(pretrained=False)
            trained_siggraph17.load_state_dict(torch.load('siggraph17_best_model.pth', map_location='cpu'))
            print("✅ Trained SIGGRAPH17 loaded")
        
        print("\n🎉 All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

if __name__ == "__main__":
    test_trained_models()
