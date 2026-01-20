import os
import json

def compute_composite_score(metrics):
    """Lower is better - using same formula as pipeline"""
    fid = float(metrics.get("fid", 0.0))
    lpips = float(metrics.get("lpips", 0.0) or 0.0)
    deltaE = float(metrics.get("deltaE", 0.0))
    colorfulness = float(metrics.get("colorfulness", 0.0))
    psnr = float(metrics.get("psnr", 0.0))
    ssim = float(metrics.get("ssim", 0.0))
    
    # Note: The original formula treats raw metrics as normalized
    # For proper normalization, we'd need to scale, but for ranking, raw works
    score = (
        0.40 * fid +
        0.20 * lpips +
        0.20 * deltaE +
        0.10 * (1.0 - colorfulness / 100.0) +  # Normalize colorfulness to 0-1
        0.05 * (1.0 - psnr / 30.0) +  # Normalize PSNR (assuming max ~30dB)
        0.05 * (1.0 - ssim)  # SSIM is already 0-1
    )
    return score

results_dir = "results"
all_results = []

for fname in sorted(os.listdir(results_dir)):
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
        print(f"Error reading {path}: {e}")
        continue

all_results.sort(key=lambda x: x["score"])

# Write to file
with open("best_models_analysis.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("TOP 20 BEST MODELS (Based on ImageNet Composite Score)\n")
    f.write("="*80 + "\n\n")
    f.write("Composite Score Formula:\n")
    f.write("  0.40*FID + 0.20*LPIPS + 0.20*ΔE + 0.10*(1-Colorfulness/100) + 0.05*(1-PSNR/30) + 0.05*(1-SSIM)\n")
    f.write("  (Lower is better)\n\n")
    f.write(f"Found {len(all_results)} pretrained variants with ImageNet metrics\n\n")
    f.write("-"*80 + "\n\n")
    
    for i, r in enumerate(all_results[:20], 1):
        f.write(f"{i}. {r['variant']}\n")
        f.write(f"   Score: {r['score']:.6f}\n")
        img = r['imagenet']
        f.write(f"   ImageNet Metrics:\n")
        f.write(f"     PSNR: {img.get('psnr', 0):.2f} dB\n")
        f.write(f"     SSIM: {img.get('ssim', 0):.4f}\n")
        f.write(f"     LPIPS: {img.get('lpips', 0):.4f}\n")
        f.write(f"     ΔE: {img.get('deltaE', 0):.2f}\n")
        f.write(f"     Colorfulness: {img.get('colorfulness', 0):.1f}\n")
        if 'fid' in img:
            f.write(f"     FID: {img.get('fid', 0):.4f}\n")
        cifar = r['cifar10']
        if cifar:
            f.write(f"   CIFAR-10 Metrics:\n")
            f.write(f"     PSNR: {cifar.get('psnr', 0):.2f} dB\n")
            f.write(f"     SSIM: {cifar.get('ssim', 0):.4f}\n")
        f.write("\n")
    
    if all_results:
        best = all_results[0]
        f.write("="*80 + "\n")
        f.write(f"🏆 BEST MODEL: {best['variant']}\n")
        f.write(f"   Composite Score: {best['score']:.6f}\n")
        f.write(f"   Checkpoint: experiments/{best['variant']}_pretrained/checkpoints/final.pth\n")
        f.write("="*80 + "\n")

print(f"Analysis complete! Found {len(all_results)} models.")
print(f"Results written to: best_models_analysis.txt")
if all_results:
    best = all_results[0]
    print(f"\nBest model: {best['variant']} (score: {best['score']:.6f})")
