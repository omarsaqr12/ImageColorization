#!/usr/bin/env python3
"""
Find ImageNet Dataset Path
This script will help you locate the correct ImageNet dataset path
"""

import os
import glob

def find_imagenet_path():
    """Find the ImageNet dataset path"""
    print("Searching for ImageNet dataset...")
    
    # Possible base directories
    base_dirs = [
        "/home/mohab/Downloads/work/colorization/",
        "/Downloads/work/colorization/",
        "./",
        "../"
    ]
    
    # Look for the imagenet directory
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"Checking: {base_dir}")
            
            # Look for imagenet-object-localization-challenge directory
            imagenet_dirs = glob.glob(os.path.join(base_dir, "*imagenet*"))
            
            for imagenet_dir in imagenet_dirs:
                print(f"  Found: {imagenet_dir}")
                
                # Check for ILSVRC subdirectory
                ilsrvc_path = os.path.join(imagenet_dir, "ILSVRC")
                if os.path.exists(ilsrvc_path):
                    print(f"    ✅ Found ILSVRC directory: {ilsrvc_path}")
                    
                    # Check for Data subdirectory
                    data_path = os.path.join(ilsrvc_path, "Data")
                    if os.path.exists(data_path):
                        print(f"    ✅ Found Data directory: {data_path}")
                        
                        # Check for CLS-LOC subdirectory
                        cls_loc_path = os.path.join(data_path, "CLS-LOC")
                        if os.path.exists(cls_loc_path):
                            print(f"    ✅ Found CLS-LOC directory: {cls_loc_path}")
                            
                            # Check for train, val, test subdirectories
                            subdirs = ['train', 'val', 'test']
                            found_subdirs = []
                            for subdir in subdirs:
                                subdir_path = os.path.join(cls_loc_path, subdir)
                                if os.path.exists(subdir_path):
                                    found_subdirs.append(subdir)
                            
                            if found_subdirs:
                                print(f"    ✅ Found subdirectories: {found_subdirs}")
                                
                                # Count images
                                total_images = 0
                                for subdir in found_subdirs:
                                    subdir_path = os.path.join(cls_loc_path, subdir)
                                    images = glob.glob(os.path.join(subdir_path, "**", "*.jpg"), recursive=True)
                                    images.extend(glob.glob(os.path.join(subdir_path, "**", "*.JPEG"), recursive=True))
                                    total_images += len(images)
                                    print(f"      {subdir}: {len(images)} images")
                                
                                print(f"\n🎉 FOUND IMAGENET DATASET!")
                                print(f"Path: {cls_loc_path}")
                                print(f"Total images: {total_images}")
                                print(f"\nUse this path in your evaluation script:")
                                print(f"'{cls_loc_path}'")
                                return cls_loc_path
                            else:
                                print(f"    ❌ No train/val/test subdirectories found")
                        else:
                            print(f"    ❌ CLS-LOC directory not found")
                    else:
                        print(f"    ❌ Data directory not found")
                else:
                    print(f"    ❌ ILSVRC directory not found")
    
    print("\n❌ ImageNet dataset not found!")
    print("Make sure the dataset is properly extracted and in the expected location.")
    return None

def main():
    print("ImageNet Dataset Path Finder")
    print("="*40)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Find ImageNet path
    imagenet_path = find_imagenet_path()
    
    if imagenet_path:
        print(f"\n✅ Success! Use this path: {imagenet_path}")
    else:
        print(f"\n❌ Could not find ImageNet dataset")
        print("Please check that the dataset is properly extracted.")

if __name__ == "__main__":
    main()
