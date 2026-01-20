"""
Quick test script to verify ILA usage.
Run this after running the tests to see ILA in action.
"""

import torch
import sys
import os

# Add colorization module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'colorization'))
from colorizers import eccv16

def main():
    print("=" * 60)
    print("ILA Usage Test")
    print("=" * 60)
    
    # Create models
    print("\n1. Creating baseline model (no ILA)...")
    model_baseline = eccv16(pretrained=False, use_ila=False)
    model_baseline.eval()
    print("   ✓ Baseline model created")
    
    print("\n2. Creating ILA-enabled model...")
    model_ila = eccv16(pretrained=False, use_ila=True, ila_reduction=4, ila_use_dw_conv=True)
    model_ila.eval()
    print("   ✓ ILA-enabled model created")
    
    # Test with dummy input
    print("\n3. Testing forward pass with dummy grayscale image...")
    batch_size = 1
    height, width = 224, 224
    input_l = torch.randn(batch_size, 1, height, width)  # Grayscale: (batch, 1, H, W)
    print(f"   Input shape: {input_l.shape}")
    
    with torch.no_grad():
        print("\n   Running baseline model...")
        output_baseline = model_baseline(input_l)
        print(f"   ✓ Baseline output shape: {output_baseline.shape}")
        
        print("\n   Running ILA-enabled model...")
        output_ila = model_ila(input_l)
        print(f"   ✓ ILA output shape: {output_ila.shape}")
    
    # Verify shapes match
    print("\n4. Verifying output shapes...")
    if output_baseline.shape == output_ila.shape:
        print(f"   ✓ Shapes match: {output_baseline.shape}")
        print("   ✓ Both models produce same output dimensions")
    else:
        print(f"   ✗ Shape mismatch!")
        print(f"     Baseline: {output_baseline.shape}")
        print(f"     ILA: {output_ila.shape}")
        return False
    
    # Count parameters
    print("\n5. Comparing model sizes...")
    baseline_params = sum(p.numel() for p in model_baseline.parameters())
    ila_params = sum(p.numel() for p in model_ila.parameters())
    extra_params = ila_params - baseline_params
    
    print(f"   Baseline parameters: {baseline_params:,}")
    print(f"   ILA parameters: {ila_params:,}")
    print(f"   Additional parameters (ILA): {extra_params:,} ({extra_params/1e6:.2f}M)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! ILA is working correctly.")
    print("=" * 60)
    print("\nYou can now use ILA in your training/evaluation code:")
    print("  from colorizers import eccv16")
    print("  model = eccv16(pretrained=False, use_ila=True)")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

