#!/usr/bin/env python3
"""
Variant Evaluation for ECCV16 Models on CIFAR-10 and ImageNet.

This module provides evaluation helpers used by the automated search pipeline.
It computes:
  - PSNR
  - SSIM
  - LPIPS (if available)
  - ΔE (using LAB distance)
  - Colorfulness

Results are saved as JSON files:
  results/<variant_id>__<weight_init_mode>.json
"""

import os
import glob
import json
import random
import warnings
from typing import Dict, Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Suppress LAB→RGB warnings
warnings.filterwarnings('ignore', message='.*negative Z values.*')

from colorization.colorizers.util import preprocess_img, postprocess_tens

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False


def _rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    """RGB [H,W,3] in [0,1] -> LAB [H,W,3]."""
    return color.rgb2lab(np.clip(rgb, 0, 1))


def _delta_e_mean(lab1: np.ndarray, lab2: np.ndarray) -> float:
    diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    return float(diff.mean())


def _colorfulness_index(image: np.ndarray) -> float:
    """
    Hasler and Suesstrunk colorfulness metric.
    """
    if image.max() <= 1.0:
        image = (image * 255.0).astype(np.uint8)

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg_mean, yb_mean = rg.mean(), yb.mean()
    rg_std, yb_std = rg.std(), yb.std()
    return float(np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2))


@torch.no_grad()
def evaluate_on_cifar10(
    model: torch.nn.Module,
    device: str,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate colorization model on CIFAR-10 test split.
    """
    transform = transforms.ToTensor()
    # Prefer existing dataset under data/cifar10_data; only download if missing
    cifar_root_existing = "./data/cifar10_data"
    if os.path.exists(cifar_root_existing):
        test_dataset = torchvision.datasets.CIFAR10(
            root=cifar_root_existing,
            train=False,
            download=False,
            transform=transform,
        )
    else:
        test_dataset = torchvision.datasets.CIFAR10(
            root="./cifar-10-python",
            train=False,
            download=True,
            transform=transform,
        )
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    psnr_vals, ssim_vals, lpips_vals, de_vals, color_vals = [], [], [], [], []

    count = 0
    pbar = tqdm(loader, desc="CIFAR-10 eval", total=min(num_samples, len(loader))) if TQDM_AVAILABLE else loader
    for rgb_tensor, _ in pbar:
        if count >= num_samples:
            break
        rgb_tensor = rgb_tensor.to(device)
        # Convert to HWC uint8 for the original ECCV16 preprocess_img utility
        rgb_np = rgb_tensor[0].permute(1, 2, 0).cpu().numpy()  # float32 in [0,1]
        rgb_uint8 = (np.clip(rgb_np, 0, 1) * 255.0).astype(np.uint8)

        # Convert to grayscale (L) via LAB
        lab = color.rgb2lab(np.clip(rgb_np, 0, 1))
        L = lab[..., 0:1]
        tens_l_orig, tens_l_rs = preprocess_img(rgb_uint8, HW=(256, 256))
        if device != "cpu":
            tens_l_rs = tens_l_rs.to(device)

        # Forward
        out = model(tens_l_rs)["ab_output"] if isinstance(model(tens_l_rs), dict) else model(tens_l_rs)
        colorized = postprocess_tens(tens_l_orig, out.cpu())
        colorized_np = np.array(colorized)

        # Metrics
        psnr_vals.append(psnr(rgb_np, colorized_np, data_range=1.0))
        ssim_vals.append(
            ssim(
                rgb_np,
                colorized_np,
                data_range=1.0,
                channel_axis=2,
            )
        )
        lab_orig = _rgb_to_lab_np(rgb_np)
        lab_pred = _rgb_to_lab_np(colorized_np)
        de_vals.append(_delta_e_mean(lab_orig, lab_pred))
        color_vals.append(_colorfulness_index(colorized_np))

        if lpips_model is not None:
            t1 = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
            t2 = torch.from_numpy(colorized_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
            lp = float(lpips_model(t1, t2).item())
            lpips_vals.append(lp)

        count += 1
        if TQDM_AVAILABLE:
            pbar.set_postfix({'count': count})

    metrics = {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "deltaE": float(np.mean(de_vals)),
        "colorfulness": float(np.mean(color_vals)),
    }
    if lpips_vals:
        metrics["lpips"] = float(np.mean(lpips_vals))
    else:
        metrics["lpips"] = None
    return metrics


class ImageNetSubsetDataset(Dataset):
    """
    Lightweight ImageNet subset dataset that samples up to max_samples images
    from the full ImageNet tree without requiring a pre-created subset.
    """

    def __init__(self, root_dir: str, transform=None, max_samples: int = 10000):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all image paths recursively
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        image_paths = []
        for ext in exts:
            image_paths.extend(
                glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
            )

        if not image_paths:
            raise FileNotFoundError(
                f"No images found under ImageNet root: {root_dir}"
            )

        if max_samples and len(image_paths) > max_samples:
            self.image_paths = random.sample(image_paths, max_samples)
        else:
            self.image_paths = image_paths

        print(
            f"ImageNetSubsetDataset: using {len(self.image_paths)} images from {root_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path


@torch.no_grad()
def evaluate_on_imagenet(
    model: torch.nn.Module,
    device: str,
    imagenet_root: str,
    num_samples: int = 10000,
) -> Dict[str, float]:
    """
    Evaluate colorization model on an ImageNet validation subset.
    """
    if not os.path.exists(imagenet_root):
        raise FileNotFoundError(f"ImageNet root not found: {imagenet_root}")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageNetSubsetDataset(
        root_dir=imagenet_root,
        transform=transform,
        max_samples=num_samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    psnr_vals, ssim_vals, lpips_vals, de_vals, color_vals = [], [], [], [], []

    count = 0
    pbar = tqdm(loader, desc="ImageNet eval", total=min(num_samples, len(dataset))) if TQDM_AVAILABLE else loader
    for rgb_tensor, _ in pbar:
        if count >= num_samples:
            break
        rgb_tensor = rgb_tensor.to(device)
        rgb_np = rgb_tensor[0].permute(1, 2, 0).cpu().numpy()  # float32 in [0,1]
        rgb_uint8 = (np.clip(rgb_np, 0, 1) * 255.0).astype(np.uint8)

        tens_l_orig, tens_l_rs = preprocess_img(rgb_uint8, HW=(256, 256))
        if device != "cpu":
            tens_l_rs = tens_l_rs.to(device)

        out = model(tens_l_rs)["ab_output"] if isinstance(model(tens_l_rs), dict) else model(tens_l_rs)
        colorized = postprocess_tens(tens_l_orig, out.cpu())
        colorized_np = np.array(colorized)

        psnr_vals.append(psnr(rgb_np, colorized_np, data_range=1.0))
        ssim_vals.append(
            ssim(
                rgb_np,
                colorized_np,
                data_range=1.0,
                channel_axis=2,
            )
        )
        lab_orig = _rgb_to_lab_np(rgb_np)
        lab_pred = _rgb_to_lab_np(colorized_np)
        de_vals.append(_delta_e_mean(lab_orig, lab_pred))
        color_vals.append(_colorfulness_index(colorized_np))

        if lpips_model is not None:
            t1 = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
            t2 = torch.from_numpy(colorized_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
            lp = float(lpips_model(t1, t2).item())
            lpips_vals.append(lp)

        count += 1
        if TQDM_AVAILABLE:
            pbar.set_postfix({'count': count})

    metrics = {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "deltaE": float(np.mean(de_vals)),
        "colorfulness": float(np.mean(color_vals)),
    }
    if lpips_vals:
        metrics["lpips"] = float(np.mean(lpips_vals))
    else:
        metrics["lpips"] = None
    return metrics


def save_variant_results(
    variant_id: str,
    weight_init_mode: str,
    cifar_metrics: Dict[str, Any],
    imagenet_metrics: Dict[str, Any],
    results_root: str = "results",
) -> str:
    """
    Save combined CIFAR-10 and ImageNet results for a given variant.
    """
    os.makedirs(results_root, exist_ok=True)
    fname = f"{variant_id}__{weight_init_mode}.json"
    path = os.path.join(results_root, fname)

    payload = {
        "cifar10": cifar_metrics,
        "imagenet": imagenet_metrics,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


__all__ = ["evaluate_on_cifar10", "evaluate_on_imagenet", "save_variant_results"]


