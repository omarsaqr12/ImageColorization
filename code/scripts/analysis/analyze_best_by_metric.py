#!/usr/bin/env python3
"""Simpler version to find best models by metric"""
import os
import json
import sys

def main():
    results_dir = "results"
    all_results = []
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist!")
        return
    
    print("Loading results...", file=sys.stderr)
    
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
        except Exception as e:
            print(f"Error reading {fname}: {e}", file=sys.stderr)
            continue
    
    print(f"Loaded {len(all_results)} results\n", file=sys.stderr)
    
    # Metrics to analyze
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
    
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("BEST 5 MODELS FOR EACH METRIC")
    output_lines.append("="*80)
    
    for metric_name, dataset, higher_is_better in metrics:
        valid = []
        for r in all_results:
            metrics_dict = r.get(dataset, {})
            value = metrics_dict.get(metric_name)
            if value is not None:
                valid.append((r, value))
        
        valid.sort(key=lambda x: x[1], reverse=higher_is_better)
        top5 = valid[:5]
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"TOP 5 BY {metric_name.upper()} ({dataset.upper()})")
        output_lines.append(f"{'='*80}")
        output_lines.append(f"{'Rank':<6} {'Variant':<15} {'Init':<12} {metric_name.upper():<12} {'PSNR':<8} {'SSIM':<8}")
        output_lines.append("-"*80)
        
        for i, (model, value) in enumerate(top5, 1):
            metrics_dict = model.get(dataset, {})
            output_lines.append(f"{i:<6} {model['variant_id']:<15} {model['init_mode']:<12} {value:<12.4f} {metrics_dict.get('psnr',0):<8.2f} {metrics_dict.get('ssim',0):<8.4f}")
    
    output_text = "\n".join(output_lines)
    
    # Save to file
    with open("best_models_by_metric_summary.txt", "w") as f:
        f.write(output_text)
    
    # Also print to stdout
    print(output_text)
    print(f"\n✅ Summary saved to: best_models_by_metric_summary.txt")

if __name__ == "__main__":
    main()
