#!/usr/bin/env python3
"""Find best model using exact pipeline formula"""
import os
import json

def compute_composite_score(metrics):
    """Exact formula from run_full_pipeline.py - treats raw metrics as normalized"""
    FID_norm = float(metrics.get("fid", 0.0))
    LPIPS_norm = float(metrics.get("lpips", 0.0) or 0.0)
    DE_norm = float(metrics.get("deltaE", 0.0))
    Colorfulness_norm = float(metrics.get("colorfulness", 0.0))
    PSNR_norm = float(metrics.get("psnr", 0.0))
    SSIM_norm = float(metrics.get("ssim", 0.0))
    
    score = (
        0.40 * FID_norm +
        0.20 * LPIPS_norm +
        0.20 * DE_norm +
        0.10 * (1.0 - Colorfulness_norm) +
        0.05 * (1.0 - PSNR_norm) +
        0.05 * (1.0 - SSIM_norm)
    )
    return float(score)

results = []
for fname in sorted(os.listdir("results")):
    if not fname.endswith("__pretrained.json"):
        continue
    path = os.path.join("results", fname)
    try:
        data = json.load(open(path))
        imagenet = data.get("imagenet", {})
        if imagenet:
            variant = fname.replace("__pretrained.json", "")
            score = compute_composite_score(imagenet)
            results.append((variant, score, imagenet, data.get("cifar10", {})))
    except:
        pass

results.sort(key=lambda x: x[1])

output = []
output.append("="*80)
output.append("TOP 20 BEST MODELS (ImageNet Composite Score)")
output.append("="*80)
output.append("\nFormula: 0.40*FID + 0.20*LPIPS + 0.20*ΔE + 0.10*(1-Colorfulness) + 0.05*(1-PSNR) + 0.05*(1-SSIM)")
output.append("(Lower is better - note: formula treats raw metrics as normalized)")
output.append(f"\nFound {len(results)} pretrained variants\n")

for i, (variant, score, img, cifar) in enumerate(results[:20], 1):
    output.append(f"{i}. {variant} (score: {score:.6f})")
    output.append(f"   ImageNet: PSNR={img.get('psnr',0):.2f}dB SSIM={img.get('ssim',0):.4f} LPIPS={img.get('lpips',0):.4f} ΔE={img.get('deltaE',0):.2f} Color={img.get('colorfulness',0):.1f}")
    if cifar:
        output.append(f"   CIFAR-10: PSNR={cifar.get('psnr',0):.2f}dB SSIM={cifar.get('ssim',0):.4f}")

if results:
    best = results[0]
    output.append("\n" + "="*80)
    output.append(f"🏆 BEST MODEL: {best[0]}")
    output.append(f"   Score: {best[1]:.6f}")
    output.append(f"   Checkpoint: experiments/{best[0]}_pretrained/checkpoints/final.pth")
    output.append("="*80)

with open("best_model_results.txt", "w") as f:
    f.write("\n".join(output))

print("\n".join(output))
