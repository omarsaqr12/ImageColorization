"""
Unit tests for ILA integration into ECCV16.

This module verifies that:
1. Baseline ECCV16 (use_ila=False) produces expected output shapes
2. ILA-enabled ECCV16 (use_ila=True) produces same output shapes
3. ILA blocks maintain feature map dimensions
4. Forward pass completes without errors
"""

import torch
import sys
import os

# Add colorization module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../colorization'))
from colorizers import eccv16, ILA


def test_baseline_shape():
    """Test that baseline ECCV16 produces expected output shape."""
    print("Testing baseline ECCV16 (use_ila=False)...")
    
    model = eccv16(pretrained=False, use_ila=False)
    model.eval()
    
    # Create dummy input: grayscale image (B, 1, H, W)
    batch_size = 2
    height, width = 224, 224
    input_l = torch.randn(batch_size, 1, height, width)
    
    with torch.no_grad():
        output = model(input_l)
    
    # Expected output: (B, 2, H, W) - ab channels
    expected_shape = (batch_size, 2, height, width)
    actual_shape = output.shape
    
    assert actual_shape == expected_shape, \
        f"Baseline output shape mismatch: expected {expected_shape}, got {actual_shape}"
    
    print(f"✓ Baseline output shape: {actual_shape}")
    return True


def test_ila_enabled_shape():
    """Test that ILA-enabled ECCV16 produces same output shape as baseline."""
    print("\nTesting ILA-enabled ECCV16 (use_ila=True)...")
    
    model = eccv16(pretrained=False, use_ila=True, ila_reduction=4, ila_use_dw_conv=True)
    model.eval()
    
    # Create dummy input: grayscale image (B, 1, H, W)
    batch_size = 2
    height, width = 224, 224
    input_l = torch.randn(batch_size, 1, height, width)
    
    with torch.no_grad():
        output = model(input_l)
    
    # Expected output: (B, 2, H, W) - ab channels (same as baseline)
    expected_shape = (batch_size, 2, height, width)
    actual_shape = output.shape
    
    assert actual_shape == expected_shape, \
        f"ILA-enabled output shape mismatch: expected {expected_shape}, got {actual_shape}"
    
    print(f"✓ ILA-enabled output shape: {actual_shape}")
    return True


def test_shape_consistency():
    """Test that baseline and ILA-enabled models produce same output shapes."""
    print("\nTesting shape consistency between baseline and ILA-enabled...")
    
    batch_size = 2
    height, width = 224, 224
    input_l = torch.randn(batch_size, 1, height, width)
    
    # Baseline model
    model_baseline = eccv16(pretrained=False, use_ila=False)
    model_baseline.eval()
    
    # ILA-enabled model
    model_ila = eccv16(pretrained=False, use_ila=True, ila_reduction=4, ila_use_dw_conv=True)
    model_ila.eval()
    
    with torch.no_grad():
        output_baseline = model_baseline(input_l)
        output_ila = model_ila(input_l)
    
    assert output_baseline.shape == output_ila.shape, \
        f"Shape mismatch: baseline {output_baseline.shape} vs ILA {output_ila.shape}"
    
    print(f"✓ Both models produce same output shape: {output_baseline.shape}")
    return True


def test_ila_block_shapes():
    """Test that ILA blocks maintain feature map dimensions."""
    print("\nTesting ILA block output shapes...")
    
    # Test ILA blocks at different channel dimensions
    test_cases = [
        (128, 32, 32),   # After model2
        (256, 16, 16),   # After model3
        (512, 8, 8),     # After model4
    ]
    
    for in_channels, height, width in test_cases:
        ila = ILA(in_channels=in_channels, reduction=4, use_dw_conv=True)
        ila.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, in_channels, height, width)
        
        with torch.no_grad():
            out = ila(x)
        
        expected_shape = (batch_size, in_channels, height, width)
        assert out.shape == expected_shape, \
            f"ILA block shape mismatch for {in_channels} channels: " \
            f"expected {expected_shape}, got {out.shape}"
        
        print(f"✓ ILA block ({in_channels} channels, {height}x{width}): {out.shape}")
    
    return True


def test_ila_forward_pass():
    """Test that ILA forward pass completes without errors."""
    print("\nTesting ILA forward pass...")
    
    ila = ILA(in_channels=256, reduction=4, use_dw_conv=True)
    ila.eval()
    
    batch_size = 1
    x = torch.randn(batch_size, 256, 32, 32)
    
    try:
        with torch.no_grad():
            out = ila(x)
        print(f"✓ ILA forward pass successful: {x.shape} -> {out.shape}")
        return True
    except Exception as e:
        print(f"✗ ILA forward pass failed: {e}")
        raise


def test_ila_without_dw_conv():
    """Test ILA block without depthwise convolution."""
    print("\nTesting ILA block without depthwise convolution...")
    
    ila = ILA(in_channels=256, reduction=4, use_dw_conv=False)
    ila.eval()
    
    batch_size = 2
    x = torch.randn(batch_size, 256, 32, 32)
    
    with torch.no_grad():
        out = ila(x)
    
    expected_shape = x.shape
    assert out.shape == expected_shape, \
        f"ILA (no dw_conv) shape mismatch: expected {expected_shape}, got {out.shape}"
    
    print(f"✓ ILA (no dw_conv) output shape: {out.shape}")
    return True


def test_different_reductions():
    """Test ILA with different reduction factors."""
    print("\nTesting ILA with different reduction factors...")
    
    reductions = [2, 4, 8]
    in_channels = 256
    
    for reduction in reductions:
        ila = ILA(in_channels=in_channels, reduction=reduction, use_dw_conv=True)
        ila.eval()
        
        batch_size = 1
        x = torch.randn(batch_size, in_channels, 32, 32)
        
        with torch.no_grad():
            out = ila(x)
        
        assert out.shape == x.shape, \
            f"ILA (reduction={reduction}) shape mismatch: expected {x.shape}, got {out.shape}"
        
        print(f"✓ ILA (reduction={reduction}) output shape: {out.shape}")
    
    return True


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running ILA Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Baseline Shape", test_baseline_shape),
        ("ILA Enabled Shape", test_ila_enabled_shape),
        ("Shape Consistency", test_shape_consistency),
        ("ILA Block Shapes", test_ila_block_shapes),
        ("ILA Forward Pass", test_ila_forward_pass),
        ("ILA Without DW Conv", test_ila_without_dw_conv),
        ("Different Reductions", test_different_reductions),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

