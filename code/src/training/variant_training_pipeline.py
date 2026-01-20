#!/usr/bin/env python3
"""
Variant Training Pipeline for ECCV16 Models on CIFAR-10.

This module implements the two-phase training schedule described in the prompt:

  Phase A — Warmup (default 10 epochs)
    - If pretrained: freeze encoder
    - If random: encoder not frozen
    - Very small GAN weight
    - Faster LR

  Phase B — Full Training (default 60 epochs)
    - Gradual unfreezing if pretrained
    - Lower LR (cosine decay)
    - GAN and perceptual losses enabled if selected

Early stopping:
  - Patience is fixed at 5 epochs (stop if validation loss does not improve
    for 5 consecutive epochs).

Checkpoints:
  - checkpoints/<variant_id>/warmup.pth
  - checkpoints/<variant_id>/final.pth
"""

import os
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from .eccv16_variants import build_eccv16_variant
from .perceptual_loss import PerceptualLoss
from .gan_loss import (
    discriminator_loss,
    generator_gan_loss,
    compute_gradient_penalty,
)
from .discriminator import create_discriminator


def get_cifar10_loaders(
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare CIFAR-10 train/val loaders for colorization training.

    We first try to use the existing dataset under `data/cifar10_data` (which
    you already have in this project). Only if that is missing do we fall back
    to downloading into `./cifar-10-python`.
    """
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    # Preferred: use existing data/cifar10_data directory
    cifar_root_existing = "./data/cifar10_data"
    if os.path.exists(cifar_root_existing):
        dataset = torchvision.datasets.CIFAR10(
            root=cifar_root_existing,
            train=True,
            download=False,
            transform=transform,
        )
    else:
        # Fallback: download if not already present
        cifar_root_fallback = "./cifar-10-python"
        dataset = torchvision.datasets.CIFAR10(
            root=cifar_root_fallback,
            train=True,
            download=True,
            transform=transform,
        )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def rgb_to_lab(rgb_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Convert RGB tensor [B,3,H,W] to LAB using skimage via numpy.
    """
    from skimage import color

    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
    lab_np = np.zeros_like(rgb_np, dtype=np.float32)
    for i in range(rgb_np.shape[0]):
        lab_np[i] = color.rgb2lab(np.clip(rgb_np[i], 0, 1))
    lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
    return lab_tensor.to(device)


def lab_to_rgb(lab_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Convert LAB tensor [B,3,H,W] to RGB in [0,1].
    """
    from skimage import color
    # Detach so that we don't backprop through the non-differentiable
    # skimage color conversion.
    lab_np = lab_tensor.detach().permute(0, 2, 3, 1).cpu().numpy()
    rgb_np = np.zeros_like(lab_np, dtype=np.float32)
    for i in range(lab_np.shape[0]):
        rgb_np[i] = np.clip(color.lab2rgb(lab_np[i]), 0, 1)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()
    return rgb_tensor.to(device)


def build_optimizer(
    model: nn.Module,
    base_lr: float,
) -> optim.Optimizer:
    """
    Build Adam optimizer with encoder vs head LR multipliers based on
    weight_init_mode encoded in the model (ECCV16Variant).
    """
    # Separate encoder (backbone) vs head parameters using name prefixes
    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    if hasattr(model, "encoder_lr_multiplier"):
        enc_mult = float(model.encoder_lr_multiplier)
    else:
        enc_mult = 1.0

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": base_lr * enc_mult})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    return optim.Adam(param_groups, betas=(0.5, 0.999))


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    use_perceptual_loss: bool,
    perceptual_loss_fn: PerceptualLoss,
    use_gan: bool,
    discriminator: nn.Module,
    optimizer_d: optim.Optimizer,
    gan_weight: float,
    gp_weight: float = 0.0,
) -> float:
    """
    Train one epoch over CIFAR-10.
    Returns average total loss.
    """
    model.train()
    if use_gan and discriminator is not None:
        discriminator.train()

    l2_loss = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    for rgb_images, _ in train_loader:
        rgb_images = rgb_images.to(device)
        lab_images = rgb_to_lab(rgb_images, device)
        l_channel = lab_images[:, 0:1, :, :]
        ab_target = lab_images[:, 1:, :, :]

        # -----------------
        #  Discriminator
        # -----------------
        if use_gan and discriminator is not None:
            optimizer_d.zero_grad()
            with torch.no_grad():
                gen_out = model(l_channel, return_logits=True)
                fake_ab = gen_out["ab_output"]
            d_input_real = torch.cat([l_channel, ab_target], dim=1)
            d_input_fake = torch.cat([l_channel, fake_ab.detach()], dim=1)
            d_real = discriminator(d_input_real)
            d_fake = discriminator(d_input_fake)
            d_loss = discriminator_loss(d_real, d_fake, gan_type="lsgan")
            if gp_weight > 0.0:
                gp = compute_gradient_penalty(discriminator, d_input_real, d_input_fake, device=device)
                d_loss = d_loss + gp_weight * gp
            d_loss.backward()
            optimizer_d.step()

        # -----------------
        #  Generator
        # -----------------
        optimizer.zero_grad()
        out = model(l_channel, return_logits=True)
        ab_pred = out["ab_output"]

        # Base L2 loss
        loss_l2 = l2_loss(ab_pred, ab_target)
        total = loss_l2

        # Perceptual loss
        if use_perceptual_loss:
            pred_lab = torch.cat([l_channel, ab_pred], dim=1)
            targ_lab = lab_images
            pred_rgb = lab_to_rgb(pred_lab, device)
            targ_rgb = lab_to_rgb(targ_lab, device)
            p_loss = perceptual_loss_fn(pred_rgb, targ_rgb)
            total = total + 0.1 * p_loss

        # GAN loss (CIFAR-10: very small lambda, single D update already done above)
        if use_gan and discriminator is not None and gan_weight > 0.0:
            d_input_fake = torch.cat([l_channel, ab_pred], dim=1)
            d_fake = discriminator(d_input_fake)
            g_gan = generator_gan_loss(d_fake, gan_type="lsgan")
            total = total + gan_weight * g_gan

        total.backward()
        optimizer.step()

        total_loss += float(total.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """
    Simple validation using L2 loss on ab channels.
    """
    model.eval()
    l2_loss = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    for rgb_images, _ in val_loader:
        rgb_images = rgb_images.to(device)
        lab_images = rgb_to_lab(rgb_images, device)
        l_channel = lab_images[:, 0:1, :, :]
        ab_target = lab_images[:, 1:, :, :]

        out = model(l_channel, return_logits=True)
        ab_pred = out["ab_output"]
        loss = l2_loss(ab_pred, ab_target)
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def train_variant_on_cifar10(
    variant_config: Dict[str, Any],
    variant_id: str,
    device: str = "cuda",
    warmup_epochs: int = 10,
    full_epochs: int = 60,
    batch_size: int = 64,
    base_lr: float = 2e-4,
    early_stopping_patience: int = 5,
    experiments_root: str = "experiments",
) -> Dict[str, Any]:
    """
    Train a single ECCV16 variant on CIFAR-10 (warmup + full phases) with
    early stopping patience fixed at 5 epochs.

    Returns:
        dict with training history and best validation loss.
    """
    os.makedirs(experiments_root, exist_ok=True)
    # variant_id already includes "variant_" prefix (e.g., "variant_097")
    exp_name = f"{variant_id}_{variant_config['weight_init_mode']}"
    exp_dir = os.path.join(experiments_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(variant_config, f, indent=2)

    # Build model
    model = build_eccv16_variant(**variant_config).to(device)

    # Data
    train_loader, val_loader = get_cifar10_loaders(batch_size=batch_size)

    # Optional components
    use_perc = bool(variant_config.get("use_perceptual_loss", False))
    use_gan = bool(variant_config.get("use_gan", False))

    perceptual_loss_fn = None
    if use_perc:
        perceptual_loss_fn = PerceptualLoss(
            perceptual_layers=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
            perceptual_norm="L1",
            vgg_type="vgg16",
            device=device,
        )

    discriminator = None
    optimizer_d = None
    if use_gan:
        # CIFAR-10 constraint: lambda_gan <= 0.02, n_discriminator_updates=1
        discriminator = create_discriminator(
            input_channels=3,
            ndf=64,
            n_layers=3,
            device=device,
        )
        optimizer_d = optim.Adam(discriminator.parameters(), lr=base_lr, betas=(0.5, 0.999))

    history = {
        "warmup_train_loss": [],
        "warmup_val_loss": [],
        "full_train_loss": [],
        "full_val_loss": [],
    }

    # ---------------
    #  Phase A: Warmup
    # ---------------
    optimizer = build_optimizer(model, base_lr=base_lr * 2.0)  # Faster LR

    # For pretrained variants, freeze encoder during warmup
    if variant_config["weight_init_mode"] == "pretrained":
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                p.requires_grad = False

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, warmup_epochs + 1):
        print(f"[{exp_name}] Warmup epoch {epoch}/{warmup_epochs}")
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            use_perceptual_loss=False,  # warmup without expensive perceptual loss
            perceptual_loss_fn=perceptual_loss_fn,
            use_gan=use_gan,
            discriminator=discriminator,
            optimizer_d=optimizer_d,
            gan_weight=0.01 if use_gan else 0.0,  # very small GAN weight
            gp_weight=0.0,
        )
        val_loss = validate_one_epoch(model, val_loader, device)

        print(
            f"[{exp_name}] Warmup epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        history["warmup_train_loss"].append(train_loss)
        history["warmup_val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "warmup.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # ---------------
    #  Phase B: Full
    # ---------------
    # Gradual unfreezing for pretrained: unfreeze encoder now
    if variant_config["weight_init_mode"] == "pretrained":
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                p.requires_grad = True

    # New optimizer with lower LR and encoder LR multiplier
    optimizer = build_optimizer(model, base_lr=base_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=full_epochs)

    patience_counter = 0

    for epoch in range(1, full_epochs + 1):
        print(f"[{exp_name}] Full epoch {epoch}/{full_epochs}")
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            use_perceptual_loss=use_perc,
            perceptual_loss_fn=perceptual_loss_fn,
            use_gan=use_gan,
            discriminator=discriminator,
            optimizer_d=optimizer_d,
            gan_weight=0.02 if use_gan else 0.0,  # upper bound for CIFAR-10
            gp_weight=0.0,
        )
        val_loss = validate_one_epoch(model, val_loader, device)

        scheduler.step()

        print(
            f"[{exp_name}] Full epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        history["full_train_loss"].append(train_loss)
        history["full_val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Save history
    with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save simple training history plots for quick inspection
    try:
        epochs_warm = range(1, len(history["warmup_train_loss"]) + 1)
        epochs_full = range(1, len(history["full_train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs_warm, history["warmup_train_loss"], label="train")
        axes[0].plot(epochs_warm, history["warmup_val_loss"], label="val")
        axes[0].set_title("Warmup")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_full, history["full_train_loss"], label="train")
        axes[1].plot(epochs_full, history["full_val_loss"], label="val")
        axes[1].set_title("Full training")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(exp_dir, "training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Plotting is best-effort; don't break training if matplotlib or
        # display backends are unavailable.
        pass

    return {"best_val_loss": best_val, "history": history, "experiment_dir": exp_dir}


__all__ = ["train_variant_on_cifar10"]


