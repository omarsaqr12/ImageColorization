import os
import json

def compute_score(metrics):
    fid = float(metrics.get("fid", 0.0))
    lpips = float(metrics.get("lpips", 0.0) or 0.0)
    deltaE = float(metrics.get("deltaE", 0.0))
    color = float(metrics.get("colorfulness", 0.0))
    psnr = float(metrics.get("psnr", 0.0))
    ssim = float(metrics.get("ssim", 0.0))
    return (0.40 * fid + 0.20 * lpips + 0.20 * deltaE + 
            0.10 * (1.0 - color) + 0.05 * (1.0 - psnr) + 0.05 * (1.0 - ssim))

results = []
for f in sorted(os.listdir("results")):
    if f.endswith("__pretrained.json"):
        try:
            data = json.load(open(os.path.join("results", f)))
            img = data.get("imagenet", {})
            if img:
                v = f.replace("__pretrained.json", "")
                s = compute_score(img)
                results.append((v, s, img, data.get("cifar10", {})))
        except:
            pass

results.sort(key=lambda x: x[1])

with open("best_model_ranking.txt", "w") as out:
    out.write("="*70 + "\n")
    out.write("TOP 20 BEST MODELS\n")
    out.write("="*70 + "\n\n")
    for i, (v, s, img, cif) in enumerate(results[:20], 1):
        out.write(f"{i}. {v}: score={s:.4f}\n")
        out.write(f"   ImageNet: PSNR={img.get('psnr',0):.2f} SSIM={img.get('ssim',0):.3f} LPIPS={img.get('lpips',0):.3f} ΔE={img.get('deltaE',0):.1f}\n")
    if results:
        out.write(f"\n🏆 BEST: {results[0][0]} (score: {results[0][1]:.4f})\n")

print(f"Analysis complete. Found {len(results)} models.")
print(f"Best model: {results[0][0] if results else 'None'}")
