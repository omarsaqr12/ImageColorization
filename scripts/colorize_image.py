"""Save a colorized image using an upstream ECCV16 or SIGGRAPH17 backbone.

Example:
    python scripts/colorize_image.py input.jpg --output colorized.png --download-pretrained

The pretrained backbone weights are fetched from their original upstream
hosting on first use. They are NOT the project's distilled-student weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLORIZATION = ROOT / "code" / "colorization"


def load_rgb(path: Path):
    """Read grayscale, RGB, or RGBA inputs as a three-channel RGB array."""
    import numpy as np
    from PIL import Image, ImageOps

    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    with Image.open(path) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        return np.asarray(rgb)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Headless upstream image-colorization demo")
    parser.add_argument("image", type=Path, help="Local grayscale or color image")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG/JPEG image path")
    parser.add_argument("--model", choices=("eccv16", "siggraph17"), default="eccv16")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--download-pretrained", action="store_true",
                        help="Explicitly allow downloading upstream pretrained weights")
    args = parser.parse_args(argv)

    image = load_rgb(args.image)  # Validate input before loading a large network.
    if not args.download_pretrained:
        parser.error("--download-pretrained is required: these upstream models load weights "
                     "from the internet; this command does not load distilled students")

    sys.path.insert(0, str(COLORIZATION))
    import matplotlib.pyplot as plt
    import torch
    from colorizers import eccv16, siggraph17, postprocess_tens, preprocess_img

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")
    device = torch.device(args.device)
    backbone = eccv16 if args.model == "eccv16" else siggraph17
    model = backbone(pretrained=True).eval().to(device)
    original_l, resized_l = preprocess_img(image, HW=(256, 256))
    with torch.inference_mode():
        predicted_ab = model(resized_l.to(device)).cpu()
    colorized = postprocess_tens(original_l, predicted_ab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.output, colorized)
    print(f"Wrote {args.output} using the upstream {args.model} model (not a distilled student).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
