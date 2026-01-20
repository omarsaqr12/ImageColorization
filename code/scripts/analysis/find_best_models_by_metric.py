#!/usr/bin/env python3
"""
Find Best 5 Models for Each Metric

This script analyzes all variant results and identifies the top 5 models
for each metric (PSNR, SSIM, LPIPS, ΔE, Colorfulness) on both CIFAR-10 and ImageNet.
"""

import os
import json
from typing import Dict, List, Any
from collections import defaultdict


def load_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Load all result files."""
    all_results = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist!")
        return all_results
    
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        
        path = os.path.join(results_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Extract variant info from filename
            if "__pretrained" in fname:
                variant_id = fname.replace("__pretrained.json", "")
                init_mode = "pretrained"
            elif "__random" in fname:
                variant_id = fname.replace("__random.json", "")
                init_mode = "random"
            else:
                continue
            
            cifar_metrics = data.get("cifar10", {})
            imagenet_metrics = data.get("imagenet", {})
            
            all_results.append({
                "variant_id": variant_id,
                "init_mode": init_mode,
                "filename": fname,
                "cifar10": cifar_metrics,
                "imagenet": imagenet_metrics
            })
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            continue
    
    return all_results


def find_top_n_by_metric(
    results: List[Dict[str, Any]],
    metric_name: str,
    dataset: str = "imagenet",
    top_n: int = 5,
    higher_is_better: bool = True
) -> List[Dict[str, Any]]:
    """
    Find top N models by a specific metric.
    
    Args:
        results: List of result dictionaries
        metric_name: Name of metric (e.g., 'psnr', 'ssim', 'lpips', 'deltaE', 'colorfulness')
        dataset: 'cifar10' or 'imagenet'
        top_n: Number of top models to return
        higher_is_better: True if higher values are better (e.g., PSNR, SSIM)
    
    Returns:
        List of top N results sorted by the metric
    """
    valid_results = []
    
    for r in results:
        metrics = r.get(dataset, {})
        value = metrics.get(metric_name)
        
        if value is not None:
            valid_results.append({
            **r,
            "metric_value": value,
            "metric_name": metric_name,
            "dataset": dataset
        })
    
    # Sort: descending if higher is better, ascending if lower is better
    valid_results.sort(
        key=lambda x: x["metric_value"],
        reverse=higher_is_better
    )
    
    return valid_results[:top_n]


def print_top_models_table(
    top_models: List[Dict[str, Any]],
    metric_name: str,
    dataset: str,
    higher_is_better: bool
):
    """Print a formatted table of top models."""
    print(f"\n{'='*80}")
    print(f"TOP 5 MODELS BY {metric_name.upper()} ({dataset.upper()})")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Variant':<15} {'Init':<12} {metric_name.upper():<12} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'ΔE':<8}")
    print("-" * 80)
    
    for i, model in enumerate(top_models, 1):
        metrics = model.get(dataset, {})
        print(f"{i:<6} "
              f"{model['variant_id']:<15} "
              f"{model['init_mode']:<12} "
              f"{model['metric_value']:<12.4f} "
              f"{metrics.get('psnr', 0):<8.2f} "
              f"{metrics.get('ssim', 0):<8.4f} "
              f"{metrics.get('lpips', 0) or 0:<8.4f} "
              f"{metrics.get('deltaE', 0):<8.2f}")


def main():
    import sys
    print("="*80, flush=True)
    print("FINDING BEST 5 MODELS FOR EACH METRIC", flush=True)
    print("="*80, flush=True)
    
    # Load all results
    print("\n📁 Loading results...", flush=True)
    all_results = load_all_results()
    print(f"   Found {len(all_results)} result files", flush=True)
    
    if not all_results:
        print("❌ No results found!")
        return
    
    # Metrics configuration: (metric_name, dataset, higher_is_better)
    metrics_config = [
        # ImageNet metrics
        ("psnr", "imagenet", True),
        ("ssim", "imagenet", True),
        ("lpips", "imagenet", False),  # Lower is better
        ("deltaE", "imagenet", False),  # Lower is better
        ("colorfulness", "imagenet", True),
        
        # CIFAR-10 metrics
        ("psnr", "cifar10", True),
        ("ssim", "cifar10", True),
        ("lpips", "cifar10", False),  # Lower is better
        ("deltaE", "cifar10", False),  # Lower is better
        ("colorfulness", "cifar10", True),
    ]
    
    # Find top 5 for each metric
    all_top_models = {}
    
    for metric_name, dataset, higher_is_better in metrics_config:
        top_models = find_top_n_by_metric(
            all_results,
            metric_name,
            dataset,
            top_n=5,
            higher_is_better=higher_is_better
        )
        all_top_models[f"{dataset}_{metric_name}"] = top_models
        
        # Print table
        print_top_models_table(top_models, metric_name, dataset, higher_is_better)
    
    # Save results to JSON
    output_file = "best_models_by_metric.json"
    output_data = {}
    
    for key, top_models in all_top_models.items():
        output_data[key] = [
            {
                "variant_id": m["variant_id"],
                "init_mode": m["init_mode"],
                "filename": m["filename"],
                "metric_value": m["metric_value"],
                "all_metrics": {
                    "cifar10": m.get("cifar10", {}),
                    "imagenet": m.get("imagenet", {})
                }
            }
            for m in top_models
        ]
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Create summary text file
    summary_file = "best_models_by_metric_summary.txt"
    with open(summary_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("BEST 5 MODELS FOR EACH METRIC\n")
        f.write("="*80 + "\n\n")
        
        for metric_name, dataset, higher_is_better in metrics_config:
            key = f"{dataset}_{metric_name}"
            top_models = all_top_models[key]
            
            f.write(f"\n{'='*80}\n")
            f.write(f"TOP 5 BY {metric_name.upper()} ({dataset.upper()})\n")
            f.write(f"{'='*80}\n\n")
            
            for i, model in enumerate(top_models, 1):
                f.write(f"{i}. {model['variant_id']} ({model['init_mode']})\n")
                f.write(f"   {metric_name}: {model['metric_value']:.4f}\n")
                metrics = model.get(dataset, {})
                f.write(f"   PSNR: {metrics.get('psnr', 0):.2f} dB\n")
                f.write(f"   SSIM: {metrics.get('ssim', 0):.4f}\n")
                if metrics.get('lpips'):
                    f.write(f"   LPIPS: {metrics.get('lpips', 0):.4f}\n")
                if metrics.get('deltaE'):
                    f.write(f"   ΔE: {metrics.get('deltaE', 0):.2f}\n")
                if metrics.get('colorfulness'):
                    f.write(f"   Colorfulness: {metrics.get('colorfulness', 0):.1f}\n")
                f.write(f"   File: {model['filename']}\n\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    # Count how many times each variant appears in top 5
    variant_counts = defaultdict(int)
    for top_models in all_top_models.values():
        for model in top_models:
            variant_counts[model['variant_id']] += 1
    
    # Sort by count
    sorted_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nVariants appearing most frequently in top 5 lists:")
    print(f"{'Variant':<15} {'Count':<8}")
    print("-" * 25)
    for variant_id, count in sorted_variants[:10]:
        print(f"{variant_id:<15} {count:<8}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
