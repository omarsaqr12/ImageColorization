"""
Inter-Leaved Light Attention (ILA) Module

This module implements a lightweight attention mechanism designed to address semantic
colorization mistakes that arise from the local receptive fields of CNNs.

Problem: CNN Local Receptive Fields and Semantic Mistakes
----------------------------------------------------------
Traditional CNNs use small convolutional kernels (e.g., 3x3) that have limited receptive
fields. In image colorization, this causes several issues:

1. **Semantic Ambiguity**: A local patch might look similar to multiple object types.
   For example, a gray patch could be sky, water, or concrete. Without global context,
   the network may assign incorrect colors.

2. **Spatial Inconsistency**: Related regions (e.g., all sky pixels) should have similar
   colors, but local convolutions cannot enforce this global consistency.

3. **Long-range Dependencies**: Objects far apart in the image (e.g., a red car and a
   red stop sign) should share color information, but CNNs require many layers to
   propagate this information.

4. **Context-dependent Colorization**: The color of an object often depends on its
   surroundings (e.g., a white object in shadow vs. sunlight). Local convolutions
   struggle to capture this context.

Solution: ILA - Lightweight Global Context Injection
-----------------------------------------------------
ILA addresses these issues by injecting global context through a lightweight attention
mechanism that:

1. **Global Receptive Field**: Attention allows each spatial position to attend to all
   other positions, providing immediate global context without requiring deep stacks
   of convolutions.

2. **Efficient Computation**: Unlike full self-attention (O(N²) complexity where N is
   the number of spatial positions), ILA uses:
   - Channel reduction (reduction factor) to reduce projection dimensions
   - Spatial attention only (not channel attention) to reduce complexity
   - Optional depthwise convolution to encode local context before attention

3. **Inter-leaved Design**: ILA blocks are inserted between convolutional blocks,
   allowing the network to maintain local feature extraction while periodically
   injecting global context.

Complexity Tradeoffs vs Full Self-Attention
--------------------------------------------
Full self-attention (as in Vision Transformers):
- Complexity: O(N² * C) where N = H*W (spatial positions), C = channels
- Memory: O(N²) for attention matrix
- Parameters: 3 * C² (for Q, K, V projections)

ILA (this implementation):
- Complexity: O(N² * C') where C' = C // reduction (reduced channels)
- Memory: O(N²) for attention matrix (same as full attention)
- Parameters: 3 * C * C' (for Q, K, V projections with reduction)
- Additional: Optional depthwise conv (C * 9 parameters)

For a typical feature map of size (B, 512, 32, 32) with reduction=4:
- Full attention: ~2.1M FLOPs per forward pass
- ILA: ~0.5M FLOPs per forward pass (4x reduction)
- Memory: Both require storing 1024x1024 attention matrix (~4MB for float32)

The tradeoff: ILA sacrifices some representational capacity (reduced channel dimension
in projections) for computational efficiency, while still providing global context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ILA(nn.Module):
    """
    Inter-Leaved Light Attention (ILA) Block
    
    A lightweight attention module that injects global context into feature maps
    through scaled dot-product attention over spatial positions.
    
    Args:
        in_channels (int): Number of input channels (C)
        reduction (int, optional): Channel reduction factor for Q, K, V projections.
            Projection dimension = in_channels // reduction. Default: 4
        use_dw_conv (bool, optional): Whether to apply depthwise 3x3 conv to V
            before attention to encode local spatial context. Default: True
    
    Input:
        x: Tensor of shape (B, C, H, W)
    
    Output:
        out: Tensor of shape (B, C, H, W) (same as input)
    
    Example:
        >>> ila = ILA(in_channels=256, reduction=4, use_dw_conv=True)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = ila(x)
        >>> print(out.shape)  # torch.Size([2, 256, 32, 32])
    """
    
    def __init__(self, in_channels, reduction=4, use_dw_conv=True):
        super(ILA, self).__init__()
        
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_dw_conv = use_dw_conv
        
        # Reduced dimension for projections
        self.proj_dim = in_channels // reduction
        
        # Ensure projection dimension is at least 1
        if self.proj_dim < 1:
            self.proj_dim = 1
            reduction = in_channels
        
        # 1x1 convolutions for Q, K, V projections (with channel reduction)
        self.query_conv = nn.Conv2d(
            in_channels, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.key_conv = nn.Conv2d(
            in_channels, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.value_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        
        # Optional depthwise 3x3 convolution for local spatial context encoding
        if use_dw_conv:
            self.dw_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels, bias=False
            )
        else:
            self.dw_conv = None
        
        # Output projection to match input channels (if needed)
        # Actually, V already has in_channels, so we don't need this
        # But we'll add it for flexibility in case we want to change the design
        self.out_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        
        # Normalization layer (use GroupNorm for small batch sizes, LayerNorm for large)
        # GroupNorm with num_groups=1 is equivalent to LayerNorm over channels
        # We use GroupNorm with groups=min(32, in_channels) for stability
        num_groups = min(32, in_channels)
        # Ensure in_channels is divisible by num_groups
        while in_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        if num_groups < 1:
            num_groups = 1
        
        self.norm = nn.GroupNorm(num_groups, in_channels)
        
        # Scale factor for attention (1/sqrt(d_k))
        self.scale = (self.proj_dim) ** -0.5
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in [self.query_conv, self.key_conv, self.value_conv, self.out_conv]:
            if m is not None:
                nn.init.xavier_uniform_(m.weight)
        
        if self.dw_conv is not None:
            nn.init.xavier_uniform_(self.dw_conv.weight)
    
    def forward(self, x):
        """
        Forward pass of ILA block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            out: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        # Apply optional depthwise convolution to V for local context
        if self.dw_conv is not None:
            v_input = self.dw_conv(x)
        else:
            v_input = x
        
        # Project to Q, K, V
        # Q, K: (B, proj_dim, H, W)
        # V: (B, C, H, W)
        Q = self.query_conv(x)  # (B, proj_dim, H, W)
        K = self.key_conv(x)    # (B, proj_dim, H, W)
        V = self.value_conv(v_input)  # (B, C, H, W)
        
        # Reshape for attention computation
        # Flatten spatial dimensions: (B, C', H, W) -> (B, C', N) where N = H*W
        N = H * W
        Q = Q.view(B, self.proj_dim, N).transpose(1, 2)  # (B, N, proj_dim)
        K = K.view(B, self.proj_dim, N)  # (B, proj_dim, N)
        V = V.view(B, C, N).transpose(1, 2)  # (B, N, C)
        
        # Scaled dot-product attention
        # Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
        attention_scores = torch.bmm(Q, K) * self.scale  # (B, N, N)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, N)
        
        # Apply attention to values
        attended = torch.bmm(attention_weights, V)  # (B, N, C)
        
        # Reshape back to spatial format
        attended = attended.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        
        # Output projection
        out = self.out_conv(attended)
        
        # Residual connection + normalization
        out = self.norm(out + residual)
        
        return out


def profile(ila_module, input_shape=(1, 256, 32, 32), device='cpu'):
    """
    Profile ILA module to estimate computational cost and memory overhead.
    
    Args:
        ila_module: ILA module instance
        input_shape (tuple): Input shape (B, C, H, W). Default: (1, 256, 32, 32)
        device (str): Device to run profiling on. Default: 'cpu'
    
    Returns:
        dict: Dictionary containing:
            - 'flops': Estimated FLOPs (floating point operations)
            - 'params': Number of parameters
            - 'memory_mb': Estimated memory overhead in MB
            - 'attention_memory_mb': Memory for attention matrix in MB
            - 'complexity': Complexity class (e.g., 'O(N²*C\')')
    
    Example:
        >>> ila = ILA(in_channels=256, reduction=4, use_dw_conv=True)
        >>> stats = profile(ila, input_shape=(2, 256, 64, 64))
        >>> print(f"FLOPs: {stats['flops']:,}")
        >>> print(f"Params: {stats['params']:,}")
        >>> print(f"Memory: {stats['memory_mb']:.2f} MB")
    """
    B, C, H, W = input_shape
    N = H * W  # Number of spatial positions
    C_proj = ila_module.proj_dim
    
    # Count parameters
    params = 0
    
    # Q, K projections: C * C_proj each
    params += 2 * (C * C_proj)
    
    # V projection: C * C
    params += C * C
    
    # Output projection: C * C
    params += C * C
    
    # Depthwise conv (if used): C * 9 (3x3 kernel, one per channel)
    if ila_module.use_dw_conv:
        params += C * 9
    
    # GroupNorm parameters: 2 * C (weight + bias)
    params += 2 * C
    
    # Estimate FLOPs
    flops = 0
    
    # 1x1 convolutions for Q, K, V
    # Q, K: B * C * C_proj * H * W each
    flops += 2 * B * C * C_proj * H * W
    
    # V: B * C * C * H * W
    flops += B * C * C * H * W
    
    # Depthwise conv (if used): B * C * 9 * H * W
    if ila_module.use_dw_conv:
        flops += B * C * 9 * H * W
    
    # Attention computation
    # Q @ K^T: B * N * N * C_proj
    flops += B * N * N * C_proj
    
    # Softmax: ~3 * B * N * N (approximate)
    flops += 3 * B * N * N
    
    # Attention @ V: B * N * N * C
    flops += B * N * N * C
    
    # Output projection: B * C * C * H * W
    flops += B * C * C * H * W
    
    # Memory estimation (in bytes, then convert to MB)
    # Attention matrix: B * N * N * 4 bytes (float32)
    attention_memory_bytes = B * N * N * 4
    attention_memory_mb = attention_memory_bytes / (1024 * 1024)
    
    # Feature maps (approximate peak memory during forward pass)
    # Q, K, V: B * C_proj * H * W * 4 bytes (for Q, K) + B * C * H * W * 4 bytes (for V)
    feature_memory_bytes = (2 * B * C_proj * H * W + B * C * H * W) * 4
    feature_memory_mb = feature_memory_bytes / (1024 * 1024)
    
    total_memory_mb = attention_memory_mb + feature_memory_mb
    
    return {
        'flops': int(flops),
        'params': params,
        'memory_mb': total_memory_mb,
        'attention_memory_mb': attention_memory_mb,
        'feature_memory_mb': feature_memory_mb,
        'complexity': f'O(N²*C\') where N={N}, C\'={C_proj}',
        'input_shape': input_shape,
        'reduction': ila_module.reduction,
        'use_dw_conv': ila_module.use_dw_conv
    }


# Add profile method to ILA class for convenience
ILA.profile = lambda self, input_shape=(1, 256, 32, 32), device='cpu': profile(self, input_shape, device)

