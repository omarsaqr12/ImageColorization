#!/usr/bin/env python3
import os
import json
import sys

def compute_composite_score(metrics):
    """Lower is better"""
    fid = metrics.get("fid", 0.0)
    lpips = metrics.get("lpips", 0.0) or 0.0
    deltaE = metrics.get("deltaE", 0.0)
    colorfulness = metrics.get("colorfulness", 0.0)
    psnr = metrics.get("psnr", 0.0)
    ssim = metrics.get("ssim", 0.0)
    
    score = (
        0.40 * fid +
        0.20 * lpips +
        0.20 * deltaE +
        0.10 * (1.0 - colorfulness / 100.0) +  # Normalize colorfulness
        0.05 * (1.0 - psnr / 30.0) +  # Normalize PSNR (assuming max ~30dB)
        0.05 * (1.0 - ssim)  # SSIM is already 0-1
    )
    return score

results_dir = "results"
all_results = []

print("Analyzing results...", file=sys.stderr)

for fname in os.listdir(results_dir):
    if not fname.endswith("__pretrained.json"):
        continue
    
    path = os.path.join(results_dir, fname)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        imagenet = data.get("imagenet", {})
        if not imagenet:
            continue
        
        variant_id = fname.replace("__pretrained.json", "")
        score = compute_composite_score(imagenet)
        
        all_results.append({
            "variant": variant_id,
            "score": score,
            "imagenet": imagenet,
            "cifar10": data.get("cifar10", {})
        })
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        continue

all_results.sort(key=lambda x: x["score"])

print(f"\n{'='*80}")
print(f"TOP 20 BEST MODELS (Based on ImageNet Composite Score)")
print(f"{'='*80}\n")
print("Composite Score Formula:")
print("  0.40*FID + 0.20*LPIPS + 0.20*ΔE + 0.10*(1-Colorfulness/100) + 0.05*(1-PSNR/30) + 0.05*(1-SSIM)")
print("  (Lower is better)\n")
print(f"Found {len(all_results)} pretrained variants with ImageNet metrics\n")
print("-" * 80)

for i, r in enumerate(all_results[:20], 1):
    print(f"\n{i}. {r['variant']}")
    print(f"   Score: {r['score']:.6f}")
    img = r['imagenet']
    print(f"   ImageNet: PSNR={img.get('psnr', 0):.2f}dB, SSIM={img.get('ssim', 0):.4f}, "
          f"LPIPS={img.get('lpips', 0):.4f}, ΔE={img.get('deltaE', 0):.2f}, "
          f"Color={img.get('colorfulness', 0):.1f}")
    cifar = r['cifar10']
    if cifar:
        print(f"   CIFAR-10: PSNR={cifar.get('psnr', 0):.2f}dB, SSIM={cifar.get('ssim', 0):.4f}")

if all_results:
    best = all_results[0]
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best['variant']}")
    print(f"   Composite Score: {best['score']:.6f}")
    print(f"   Checkpoint: experiments/{best['variant']}_pretrained/checkpoints/final.pth")
    print(f"{'='*80}")
