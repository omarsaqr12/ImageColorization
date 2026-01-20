import os
import json

results_dir = "results"
all_results = []

for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith(".json"):
        continue
    
    path = os.path.join(results_dir, fname)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        if "__pretrained" in fname:
            variant_id = fname.replace("__pretrained.json", "")
            init_mode = "pretrained"
        elif "__random" in fname:
            variant_id = fname.replace("__random.json", "")
            init_mode = "random"
        else:
            continue
        
        all_results.append({
            "variant_id": variant_id,
            "init_mode": init_mode,
            "filename": fname,
            "cifar10": data.get("cifar10", {}),
            "imagenet": data.get("imagenet", {})
        })
    except:
        pass

metrics = [
    ("psnr", "imagenet", True),
    ("ssim", "imagenet", True),
    ("lpips", "imagenet", False),
    ("deltaE", "imagenet", False),
    ("colorfulness", "imagenet", True),
    ("psnr", "cifar10", True),
    ("ssim", "cifar10", True),
    ("lpips", "cifar10", False),
    ("deltaE", "cifar10", False),
    ("colorfulness", "cifar10", True),
]

output = []
output.append("="*80)
output.append("BEST 5 MODELS FOR EACH METRIC")
output.append("="*80)

for metric_name, dataset, higher_is_better in metrics:
    valid = []
    for r in all_results:
        metrics_dict = r.get(dataset, {})
        value = metrics_dict.get(metric_name)
        if value is not None:
            valid.append((r, value))
    
    valid.sort(key=lambda x: x[1], reverse=higher_is_better)
    top5 = valid[:5]
    
    output.append(f"\n{'='*80}")
    output.append(f"TOP 5 BY {metric_name.upper()} ({dataset.upper()})")
    output.append(f"{'='*80}")
    output.append(f"{'Rank':<6} {'Variant':<15} {'Init':<12} {metric_name.upper():<12} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'ΔE':<8}")
    output.append("-"*80)
    
    for i, (model, value) in enumerate(top5, 1):
        metrics_dict = model.get(dataset, {})
        output.append(f"{i:<6} {model['variant_id']:<15} {model['init_mode']:<12} {value:<12.4f} "
                     f"{metrics_dict.get('psnr',0):<8.2f} {metrics_dict.get('ssim',0):<8.4f} "
                     f"{metrics_dict.get('lpips',0) or 0:<8.4f} {metrics_dict.get('deltaE',0):<8.2f}")

with open("best_5_by_metric.txt", "w") as f:
    f.write("\n".join(output))

print(f"Found {len(all_results)} models")
print("Results saved to: best_5_by_metric.txt")
