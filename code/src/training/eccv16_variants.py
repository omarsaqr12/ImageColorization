#!/usr/bin/env python3
"""
ECCV16 Variant Builder and Configuration Generator

This module defines:
  - build_eccv16_variant(...): construct ECCV16-based models with optional modules
  - generate_model_configurations(): enumerate all valid feature combinations

It is intentionally self-contained so it can be used by:
  - search / training pipelines
  - evaluation scripts

NOTE: The goal is to closely follow the original ECCV16 architecture from
`colorization/colorizers/eccv16.py` while adding optional heads and losses in
an opt‑in way so that the baseline behavior is unchanged when all flags are
False and weight_init_mode="pretrained".
"""

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Reuse BaseColor utilities and the reference ECCV16 implementation
from colorization.colorizers.eccv16 import ECCVGenerator
from colorization.colorizers.base_color import BaseColor


# -----------------------------
#  Optional Building Blocks
# -----------------------------

class SEBlock(nn.Module):
    """Simple Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PyramidPooling(nn.Module):
    """
    Lightweight spatial pyramid pooling with 1x1, 2x2, 4x4 bins.
    The pooled features are upsampled and concatenated with the input.
    """

    def __init__(self, in_channels: int, pool_sizes=(1, 2, 4)):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        self.paths = nn.ModuleList()
        for _ in pool_sizes:
            self.paths.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(_),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                )
            )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_sizes), in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        feats = [x]
        for path in self.paths:
            y = path(x)
            y = nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        out = torch.cat(feats, dim=1)
        return self.conv_out(out)


class GlobalSemanticHead(nn.Module):
    """
    Global semantic head:
      - global average pooling at bottleneck
      - MLP to produce a semantic embedding
      - FiLM modulation of bottleneck activations
    """

    def __init__(self, channels: int, embedding_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, embedding_dim, bias=True),
            nn.ReLU(inplace=True),
        )
        # FiLM parameters (gamma, beta) from embedding
        self.film = nn.Linear(embedding_dim, channels * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        pooled = self.gap(x).view(b, c)
        emb = self.mlp(pooled)
        gamma_beta = self.film(emb).view(b, 2, c, 1, 1)
        gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
        return x * (1 + gamma) + beta


class ColorClassificationHead(nn.Module):
    """
    313-bin quantized ab classification head.

    This is a simple 1x1 conv that predicts logits over 313 bins, followed by
    an optional annealed-mean decoding step handled outside this module.
    """

    def __init__(self, in_channels: int = 512, num_bins: int = 313):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_bins, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# -----------------------------
#  Variant Wrapper
# -----------------------------

@dataclass
class ECCV16VariantConfig:
    use_multiscale: bool = False
    use_global_semantic_head: bool = False
    use_perceptual_loss: bool = False
    use_attention: bool = False
    use_gan: bool = False
    use_color_classification: bool = False
    use_class_rebalance: bool = False
    weight_init_mode: str = "pretrained"
    pretrained_weights_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ECCV16Variant(BaseColor):
    """
    ECCV16 backbone with optional modules attached at the bottleneck/decoder.

    This wraps an internal ECCVGenerator and exposes:
      - forward(input_l) -> ab_output (for colorization)
      - optional classification logits via .color_logits if enabled
    """

    def __init__(self, config: ECCV16VariantConfig):
        super().__init__()
        self.config = config

        # Recreate ECCV base backbone (we re-implement modules so we can hook in extra blocks)
        self.backbone = ECCVGenerator()

        # Attach optional modules
        bottleneck_channels = 512
        self.multiscale = PyramidPooling(bottleneck_channels) if config.use_multiscale else nn.Identity()
        self.semantic_head = GlobalSemanticHead(bottleneck_channels) if config.use_global_semantic_head else nn.Identity()
        self.attention = SEBlock(bottleneck_channels) if config.use_attention else nn.Identity()

        # Optional color classification head taps into logits feature map (313 channels)
        self.use_color_classification = config.use_color_classification
        if self.use_color_classification:
            # Reuse 313‑channel conv8_3 output as logits
            self.color_head = ColorClassificationHead(in_channels=313, num_bins=313)
        else:
            self.color_head = None

        # Copy output heads from backbone for standard regression output
        self.softmax = self.backbone.softmax
        self.model_out = self.backbone.model_out
        self.upsample4 = self.backbone.upsample4

        # Weight initialization / loading
        self._init_weights()

    # -----------------------------
    #  Weight init modes
    # -----------------------------

    def _init_weights(self) -> None:
        cfg = self.config
        # Encoder + decoder (i.e., backbone parameters)
        encoder_params = list(
            self.backbone.model1.parameters()
        ) + list(self.backbone.model2.parameters()) + list(self.backbone.model3.parameters()) + list(
            self.backbone.model4.parameters()
        ) + list(self.backbone.model5.parameters()) + list(self.backbone.model6.parameters()) + list(
            self.backbone.model7.parameters()
        ) + list(self.backbone.model8.parameters())

        if cfg.weight_init_mode == "pretrained":
            # Load ImageNet‑pretrained ECCV16 weights
            from torch.utils import model_zoo

            url = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
            state_dict = model_zoo.load_url(url, map_location="cpu", check_hash=True)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[ECCV16Variant] Loaded pretrained weights with missing={len(missing)}, unexpected={len(unexpected)}")

            # Optionally override from local path
            if cfg.pretrained_weights_path and os.path.exists(cfg.pretrained_weights_path):
                sd = torch.load(cfg.pretrained_weights_path, map_location="cpu")
                self.backbone.load_state_dict(sd, strict=False)

            # Lower LR for encoder/decoder will be handled in optimizer construction
            self.encoder_lr_multiplier = 0.1
            self.freeze_encoder_epochs = 5

        elif cfg.weight_init_mode == "random":
            # Custom Xavier/He initialization
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            self.encoder_lr_multiplier = 1.0
            self.freeze_encoder_epochs = 0
        else:
            raise ValueError(f"Unknown weight_init_mode: {cfg.weight_init_mode}")

    # -----------------------------
    #  Forward
    # -----------------------------

    def forward(self, input_l: torch.Tensor, return_logits: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass that mirrors ECCV16 while injecting optional modules at bottleneck.
        Returns a dict with:
          - 'ab_output': final 2‑channel ab prediction (upsampled)
          - 'logits': 313‑channel logits before softmax (for color classification / GAN / distillation)
          - 'color_logits': optional 313‑bin classification logits if enabled
        """
        # Re‑implement the backbone forward so we can hook bottleneck features
        conv1_2 = self.backbone.model1(self.backbone.normalize_l(input_l))
        conv2_2 = self.backbone.model2(conv1_2)
        conv3_3 = self.backbone.model3(conv2_2)
        conv4_3 = self.backbone.model4(conv3_3)
        conv5_3 = self.backbone.model5(conv4_3)
        conv6_3 = self.backbone.model6(conv5_3)

        # Bottleneck before final encoder block
        bottleneck = conv6_3
        bottleneck = self.multiscale(bottleneck)
        bottleneck = self.semantic_head(bottleneck)
        bottleneck = self.attention(bottleneck)

        conv7_3 = self.backbone.model7(bottleneck)
        conv8_3 = self.backbone.model8(conv7_3)  # [B, 313, H/4, W/4]

        logits = conv8_3
        prob = self.softmax(logits)
        out_reg = self.model_out(prob)
        ab_output = self.unnormalize_ab(self.upsample4(out_reg))

        out: Dict[str, torch.Tensor] = {
            "ab_output": ab_output,
            "logits": logits,
        }

        if self.use_color_classification and self.color_head is not None:
            out["color_logits"] = self.color_head(logits)

        if not return_logits:
            return {"ab_output": ab_output}
        return out


# -----------------------------
#  Public Builder Function
# -----------------------------

def build_eccv16_variant(
    use_multiscale: bool = False,
    use_global_semantic_head: bool = False,
    use_perceptual_loss: bool = False,
    use_attention: bool = False,
    use_gan: bool = False,
    use_color_classification: bool = False,
    use_class_rebalance: bool = False,
    weight_init_mode: str = "pretrained",
    pretrained_weights_path: Optional[str] = None,
) -> ECCV16Variant:
    """
    Construct an ECCV16 variant with the requested features.

    NOTE:
      - This only builds the model and does not attach losses.
      - Perceptual loss, GAN loss, and class rebalancing are handled by the
        training pipeline, but the flags are kept here so the config is
        complete and serializable.
    """
    cfg = ECCV16VariantConfig(
        use_multiscale=use_multiscale,
        use_global_semantic_head=use_global_semantic_head,
        use_perceptual_loss=use_perceptual_loss,
        use_attention=use_attention,
        use_gan=use_gan,
        use_color_classification=use_color_classification,
        use_class_rebalance=use_class_rebalance,
        weight_init_mode=weight_init_mode,
        pretrained_weights_path=pretrained_weights_path,
    )
    return ECCV16Variant(cfg)


# -----------------------------
#  Configuration Search Space
# -----------------------------

FEATURE_FLAGS = {
    "use_multiscale": [True, False],
    "use_global_semantic_head": [True, False],
    "use_perceptual_loss": [True, False],
    "use_attention": [True, False],
    "use_gan": [True, False],
    "use_color_classification": [True, False],
    "use_class_rebalance": [True, False],
    "weight_init_mode": ["pretrained", "random"],
}


def is_valid_config(cfg: Dict[str, Any]) -> bool:
    """
    Enforce configuration constraints:
      - If use_color_classification=True, force use_class_rebalance=True
      - If use_gan=True AND perceptual=False AND attention=False → disable GAN (i.e., treat as invalid here)
    """
    if cfg["use_color_classification"] and not cfg["use_class_rebalance"]:
        return False

    if cfg["use_gan"] and (not cfg["use_perceptual_loss"]) and (not cfg["use_attention"]):
        # The spec says to disable GAN in this case; we enforce it by not
        # emitting such configurations.
        return False

    if cfg["weight_init_mode"] not in ("pretrained", "random"):
        return False

    return True


def generate_model_configurations() -> List[Dict[str, Any]]:
    """
    Generate all valid combinations of feature flags.

    Returns:
        List of configuration dicts suitable for JSON serialization.
    """
    import itertools

    keys = list(FEATURE_FLAGS.keys())
    values_product = itertools.product(*(FEATURE_FLAGS[k] for k in keys))

    configs: List[Dict[str, Any]] = []
    for vals in values_product:
        cfg = {k: v for k, v in zip(keys, vals)}
        if is_valid_config(cfg):
            configs.append(cfg)

    return configs


__all__ = [
    "ECCV16VariantConfig",
    "ECCV16Variant",
    "build_eccv16_variant",
    "generate_model_configurations",
    "FEATURE_FLAGS",
]


