"""
Demo script for ILA profiling functionality.

This script demonstrates how to use the ILA.profile() method to analyze
computational cost and memory overhead of ILA blocks.
"""

import sys
import os

# Add colorization module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../colorization'))
from colorizers.ila_block import ILA, profile


def demo_profiling():
    """Demonstrate ILA profiling for different configurations."""
    
    print("=" * 70)
    print("ILA Block Profiling Demo")
    print("=" * 70)
    
    # Test configurations matching ECCV16 integration points
    configs = [
        {
            'name': 'After model2 (128 channels)',
            'in_channels': 128,
            'reduction': 4,
            'use_dw_conv': True,
            'input_shape': (1, 128, 64, 64),  # Approximate size after model2
        },
        {
            'name': 'After model3 (256 channels)',
            'in_channels': 256,
            'reduction': 4,
            'use_dw_conv': True,
            'input_shape': (1, 256, 32, 32),  # Approximate size after model3
        },
        {
            'name': 'After model4 (512 channels)',
            'in_channels': 512,
            'reduction': 4,
            'use_dw_conv': True,
            'input_shape': (1, 512, 16, 16),  # Approximate size after model4
        },
        {
            'name': 'After model4 (512 channels, no dw_conv)',
            'in_channels': 512,
            'reduction': 4,
            'use_dw_conv': False,
            'input_shape': (1, 512, 16, 16),
        },
        {
            'name': 'After model4 (512 channels, reduction=8)',
            'in_channels': 512,
            'reduction': 8,
            'use_dw_conv': True,
            'input_shape': (1, 512, 16, 16),
        },
    ]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 70)
        
        ila = ILA(
            in_channels=config['in_channels'],
            reduction=config['reduction'],
            use_dw_conv=config['use_dw_conv']
        )
        
        stats = profile(ila, input_shape=config['input_shape'])
        
        print(f"Input shape: {stats['input_shape']}")
        print(f"Reduction: {stats['reduction']}")
        print(f"Use depthwise conv: {stats['use_dw_conv']}")
        print(f"Complexity: {stats['complexity']}")
        print(f"\nParameters: {stats['params']:,}")
        print(f"FLOPs: {stats['flops']:,} ({stats['flops']/1e6:.2f} MFLOPs)")
        print(f"\nMemory overhead:")
        print(f"  Attention matrix: {stats['attention_memory_mb']:.2f} MB")
        print(f"  Feature maps: {stats['feature_memory_mb']:.2f} MB")
        print(f"  Total: {stats['memory_mb']:.2f} MB")
    
    # Compare with full self-attention
    print("\n" + "=" * 70)
    print("Comparison: ILA vs Full Self-Attention")
    print("=" * 70)
    
    C = 512
    H, W = 16, 16
    N = H * W
    reduction = 4
    C_proj = C // reduction
    
    # ILA parameters
    ila_params = 2 * (C * C_proj) + C * C + C * C + C * 9 + 2 * C
    # Full self-attention parameters (3 * C^2 for Q, K, V)
    full_attn_params = 3 * C * C
    
    # ILA FLOPs (approximate, for one forward pass)
    B = 1
    ila_flops = (
        2 * B * C * C_proj * H * W +  # Q, K projections
        B * C * C * H * W +           # V projection
        B * C * 9 * H * W +            # Depthwise conv
        B * N * N * C_proj +           # Q @ K^T
        3 * B * N * N +                # Softmax
        B * N * N * C +                 # Attention @ V
        B * C * C * H * W               # Output projection
    )
    
    # Full self-attention FLOPs
    full_attn_flops = (
        3 * B * C * C * H * W +        # Q, K, V projections
        B * N * N * C +                 # Q @ K^T
        3 * B * N * N +                 # Softmax
        B * N * N * C                    # Attention @ V
    )
    
    print(f"\nConfiguration: C={C}, H={H}, W={W}, N={N}, reduction={reduction}")
    print(f"\nParameters:")
    print(f"  ILA: {ila_params:,} ({ila_params/1e6:.2f}M)")
    print(f"  Full Self-Attention: {full_attn_params:,} ({full_attn_params/1e6:.2f}M)")
    print(f"  Reduction: {full_attn_params/ila_params:.2f}x")
    
    print(f"\nFLOPs (approximate):")
    print(f"  ILA: {ila_flops:,} ({ila_flops/1e6:.2f} MFLOPs)")
    print(f"  Full Self-Attention: {full_attn_flops:,} ({full_attn_flops/1e6:.2f} MFLOPs)")
    print(f"  Reduction: {full_attn_flops/ila_flops:.2f}x")
    
    print(f"\nMemory (attention matrix):")
    attn_mem_mb = B * N * N * 4 / (1024 * 1024)
    print(f"  Both require: {attn_mem_mb:.2f} MB")
    print(f"  (Same for both - attention matrix size depends on spatial resolution)")


if __name__ == "__main__":
    demo_profiling()

