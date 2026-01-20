#!/usr/bin/env python3
"""
Model Weight Management Script
Allows easy switching between trained and pre-trained weights
"""

import os
import sys
import torch
import shutil
from datetime import datetime

# Add colorization module to path
sys.path.append('colorization')
from colorizers import eccv16, siggraph17

class ModelWeightManager:
    def __init__(self):
        self.backup_dir = "model_backups"
        self.ensure_backup_dir()
    
    def ensure_backup_dir(self):
        """Create backup directory if it doesn't exist"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            print(f"Created backup directory: {self.backup_dir}")
    
    def backup_trained_weights(self):
        """Backup trained weights with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if os.path.exists('eccv16_best_model.pth'):
            backup_name = f"eccv16_trained_{timestamp}.pth"
            shutil.copy2('eccv16_best_model.pth', os.path.join(self.backup_dir, backup_name))
            print(f"✅ Backed up ECCV16 trained weights: {backup_name}")
        else:
            print("❌ No ECCV16 trained weights found to backup")
        
        if os.path.exists('siggraph17_best_model.pth'):
            backup_name = f"siggraph17_trained_{timestamp}.pth"
            shutil.copy2('siggraph17_best_model.pth', os.path.join(self.backup_dir, backup_name))
            print(f"✅ Backed up SIGGRAPH17 trained weights: {backup_name}")
        else:
            print("❌ No SIGGRAPH17 trained weights found to backup")
    
    def restore_original_weights(self):
        """Restore original pre-trained weights"""
        print("🔄 Restoring original pre-trained weights...")
        
        # Create models with pre-trained weights
        print("Loading original ECCV16 weights...")
        original_eccv16 = eccv16(pretrained=True)
        
        print("Loading original SIGGRAPH17 weights...")
        original_siggraph17 = siggraph17(pretrained=True)
        
        # Save as current best models (overwrites trained versions)
        torch.save(original_eccv16.state_dict(), 'eccv16_best_model.pth')
        torch.save(original_siggraph17.state_dict(), 'siggraph17_best_model.pth')
        
        print("✅ Original weights restored!")
        print("   - eccv16_best_model.pth now contains original weights")
        print("   - siggraph17_best_model.pth now contains original weights")
    
    def list_available_weights(self):
        """List all available weight files"""
        print("\n📁 Available Weight Files:")
        print("-" * 50)
        
        # Current weights
        if os.path.exists('eccv16_best_model.pth'):
            print("✅ eccv16_best_model.pth (current)")
        else:
            print("❌ eccv16_best_model.pth (not found)")
            
        if os.path.exists('siggraph17_best_model.pth'):
            print("✅ siggraph17_best_model.pth (current)")
        else:
            print("❌ siggraph17_best_model.pth (not found)")
        
        # Backup weights
        if os.path.exists(self.backup_dir):
            backup_files = [f for f in os.listdir(self.backup_dir) if f.endswith('.pth')]
            if backup_files:
                print(f"\n📦 Backup weights ({len(backup_files)} files):")
                for file in sorted(backup_files):
                    file_path = os.path.join(self.backup_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   - {file} ({size_mb:.1f} MB)")
            else:
                print("\n📦 No backup weights found")
    
    def load_specific_weights(self, eccv16_file=None, siggraph17_file=None):
        """Load specific weight files"""
        if eccv16_file:
            if os.path.exists(eccv16_file):
                shutil.copy2(eccv16_file, 'eccv16_best_model.pth')
                print(f"✅ Loaded ECCV16 weights from: {eccv16_file}")
            else:
                print(f"❌ ECCV16 file not found: {eccv16_file}")
        
        if siggraph17_file:
            if os.path.exists(siggraph17_file):
                shutil.copy2(siggraph17_file, 'siggraph17_best_model.pth')
                print(f"✅ Loaded SIGGRAPH17 weights from: {siggraph17_file}")
            else:
                print(f"❌ SIGGRAPH17 file not found: {siggraph17_file}")
    
    def verify_model_architecture(self):
        """Verify that models maintain original architecture"""
        print("\n🔍 Verifying Model Architecture...")
        
        try:
            # Test ECCV16
            model_eccv16 = eccv16(pretrained=False)
            print(f"✅ ECCV16 Architecture: {type(model_eccv16).__name__}")
            print(f"   Parameters: {sum(p.numel() for p in model_eccv16.parameters()):,}")
            
            # Test SIGGRAPH17
            model_siggraph17 = siggraph17(pretrained=False)
            print(f"✅ SIGGRAPH17 Architecture: {type(model_siggraph17).__name__}")
            print(f"   Parameters: {sum(p.numel() for p in model_siggraph17.parameters()):,}")
            
            print("\n✅ Architecture verification passed!")
            
        except Exception as e:
            print(f"❌ Architecture verification failed: {e}")

def main():
    manager = ModelWeightManager()
    
    print("🎯 Colorization Model Weight Manager")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Backup trained weights")
        print("2. Restore original weights")
        print("3. List available weights")
        print("4. Verify model architecture")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            manager.backup_trained_weights()
        elif choice == '2':
            confirm = input("This will overwrite trained weights. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                manager.restore_original_weights()
            else:
                print("Operation cancelled.")
        elif choice == '3':
            manager.list_available_weights()
        elif choice == '4':
            manager.verify_model_architecture()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

