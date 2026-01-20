#!/usr/bin/env python3
"""Analyze random initialization models and find the best one"""
import os
import json

def compute_composite_score(metrics):
    """Exact formula from run_full_pipeline.py"""
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

# Analyze random models
random_results = []
pretrained_results = []

print("Analyzing models...")

for fname in sorted(os.listdir("results")):
    if not fname.endswith(".json"):
        continue
    
    path = os.path.join("results", fname)
    try:
        data = json.load(open(path))
        imagenet = data.get("imagenet", {})
        if not imagenet:
            continue
        
        if fname.endswith("__random.json"):
            variant = fname.replace("__random.json", "")
            score = compute_composite_score(imagenet)
            random_results.append((variant, score, imagenet, data.get("cifar10", {})))
        elif fname.endswith("__pretrained.json"):
            variant = fname.replace("__pretrained.json", "")
            score = compute_composite_score(imagenet)
            pretrained_results.append((variant, score, imagenet, data.get("cifar10", {})))
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        continue

# Sort by score (lower is better)
random_results.sort(key=lambda x: x[1])
pretrained_results.sort(key=lambda x: x[1])

# Write analysis
output = []
output.append("="*80)
output.append("RANDOM vs PRETRAINED MODEL COMPARISON")
output.append("="*80)
output.append(f"\nFound {len(random_results)} random variants and {len(pretrained_results)} pretrained variants")
output.append("\nComposite Score Formula:")
output.append("  0.40*FID + 0.20*LPIPS + 0.20*ΔE + 0.10*(1-Colorfulness) + 0.05*(1-PSNR) + 0.05*(1-SSIM)")
output.append("  (Lower is better)\n")

output.append("\n" + "="*80)
output.append("TOP 20 BEST RANDOM MODELS")
output.append("="*80 + "\n")

for i, (variant, score, img, cifar) in enumerate(random_results[:20], 1):
    output.append(f"{i}. {variant} (score: {score:.6f})")
    output.append(f"   ImageNet: PSNR={img.get('psnr',0):.2f}dB SSIM={img.get('ssim',0):.4f} LPIPS={img.get('lpips',0):.4f} ΔE={img.get('deltaE',0):.2f} Color={img.get('colorfulness',0):.1f}")
    if cifar:
        output.append(f"   CIFAR-10: PSNR={cifar.get('psnr',0):.2f}dB SSIM={cifar.get('ssim',0):.4f}")

output.append("\n" + "="*80)
output.append("TOP 20 BEST PRETRAINED MODELS")
output.append("="*80 + "\n")

for i, (variant, score, img, cifar) in enumerate(pretrained_results[:20], 1):
    output.append(f"{i}. {variant} (score: {score:.6f})")
    output.append(f"   ImageNet: PSNR={img.get('psnr',0):.2f}dB SSIM={img.get('ssim',0):.4f} LPIPS={img.get('lpips',0):.4f} ΔE={img.get('deltaE',0):.2f} Color={img.get('colorfulness',0):.1f}")
    if cifar:
        output.append(f"   CIFAR-10: PSNR={cifar.get('psnr',0):.2f}dB SSIM={cifar.get('ssim',0):.4f}")

# Compare best of each
if random_results and pretrained_results:
    best_random = random_results[0]
    best_pretrained = pretrained_results[0]
    
    output.append("\n" + "="*80)
    output.append("BEST MODEL COMPARISON")
    output.append("="*80)
    output.append(f"\n🏆 Best Random: {best_random[0]} (score: {best_random[1]:.6f})")
    output.append(f"   ImageNet: PSNR={best_random[2].get('psnr',0):.2f}dB SSIM={best_random[2].get('ssim',0):.4f} LPIPS={best_random[2].get('lpips',0):.4f} ΔE={best_random[2].get('deltaE',0):.2f}")
    output.append(f"   CIFAR-10: PSNR={best_random[3].get('psnr',0):.2f}dB SSIM={best_random[3].get('ssim',0):.4f}")
    
    output.append(f"\n🏆 Best Pretrained: {best_pretrained[0]} (score: {best_pretrained[1]:.6f})")
    output.append(f"   ImageNet: PSNR={best_pretrained[2].get('psnr',0):.2f}dB SSIM={best_pretrained[2].get('ssim',0):.4f} LPIPS={best_pretrained[2].get('lpips',0):.4f} ΔE={best_pretrained[2].get('deltaE',0):.2f}")
    output.append(f"   CIFAR-10: PSNR={best_pretrained[3].get('psnr',0):.2f}dB SSIM={best_pretrained[3].get('ssim',0):.4f}")
    
    if best_random[1] < best_pretrained[1]:
        output.append(f"\n✅ Random initialization is BETTER (lower score: {best_random[1]:.6f} vs {best_pretrained[1]:.6f})")
        output.append(f"   Best overall model: {best_random[0]} (random)")
        output.append(f"   Checkpoint: experiments/{best_random[0]}_random/checkpoints/final.pth")
    else:
        output.append(f"\n✅ Pretrained initialization is BETTER (lower score: {best_pretrained[1]:.6f} vs {best_random[1]:.6f})")
        output.append(f"   Best overall model: {best_pretrained[0]} (pretrained)")
        output.append(f"   Checkpoint: experiments/{best_pretrained[0]}_pretrained/checkpoints/final.pth")

# Statistics
if random_results and pretrained_results:
    random_scores = [r[1] for r in random_results]
    pretrained_scores = [r[1] for r in pretrained_results]
    
    output.append("\n" + "="*80)
    output.append("STATISTICS")
    output.append("="*80)
    output.append(f"\nRandom Models:")
    output.append(f"  Mean score: {sum(random_scores)/len(random_scores):.6f}")
    output.append(f"  Best score: {min(random_scores):.6f}")
    output.append(f"  Worst score: {max(random_scores):.6f}")
    output.append(f"\nPretrained Models:")
    output.append(f"  Mean score: {sum(pretrained_scores)/len(pretrained_scores):.6f}")
    output.append(f"  Best score: {min(pretrained_scores):.6f}")
    output.append(f"  Worst score: {max(pretrained_scores):.6f}")

# Write to file
with open("random_models_analysis.txt", "w") as f:
    f.write("\n".join(output))

# Print to console
print("\n".join(output))
