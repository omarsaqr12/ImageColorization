#!/usr/bin/env python3
"""Summarize variant evaluation results from variant_0 folder."""

import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Any
import statistics

def load_all_results(results_dir: str = "variant_0") -> List[Dict[str, Any]]:
    """Load all variant result JSON files."""
    files = glob.glob(os.path.join(results_dir, "variant_*__*.json"))
    results = []
    
    for fpath in sorted(files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                # Extract variant ID and init mode from filename
                basename = os.path.basename(fpath)
                parts = basename.replace('.json', '').split('__')
                if len(parts) == 2:
                    variant_id = parts[0]
                    init_mode = parts[1]
                    data['_variant_id'] = variant_id
                    data['_init_mode'] = init_mode
                    data['_filename'] = basename
                    results.append(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
    
    return results

def compute_statistics(results: List[Dict[str, Any]], dataset: str = "imagenet") -> Dict[str, Any]:
    """Compute statistics for a dataset."""
    metrics_list = defaultdict(list)
    
    for r in results:
        dataset_data = r.get(dataset, {})
        for metric, value in dataset_data.items():
            if isinstance(value, (int, float)):
                metrics_list[metric].append(value)
    
    stats = {}
    for metric, values in metrics_list.items():
        if values:
            stats[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'count': len(values)
            }
    
    return stats

def compute_composite_score(
    metrics: Dict[str, float],
    w_fid: float = 0.40,
    w_lpips: float = 0.20,
    w_deltaE: float = 0.20,
    w_color: float = 0.10,
    w_psnr: float = 0.05,
    w_ssim: float = 0.05,
) -> float:
    """Compute composite score from normalized metrics. Lower is better."""
    FID_norm = float(metrics.get("fid", 0.0))
    LPIPS_norm = float(metrics.get("lpips", 0.0) or 0.0)
    DE_norm = float(metrics.get("deltaE", 0.0))
    Colorfulness_norm = float(metrics.get("colorfulness", 0.0))
    PSNR_norm = float(metrics.get("psnr", 0.0))
    SSIM_norm = float(metrics.get("ssim", 0.0))

    score = (
        w_fid * FID_norm
        + w_lpips * LPIPS_norm
        + w_deltaE * DE_norm
        + w_color * (1.0 - Colorfulness_norm)
        + w_psnr * (1.0 - PSNR_norm)
        + w_ssim * (1.0 - SSIM_norm)
    )
    return float(score)

def find_best_worst(results: List[Dict[str, Any]], dataset: str = "imagenet") -> Dict[str, Any]:
    """Find best and worst models based on composite score."""
    scored = []
    for r in results:
        dataset_data = r.get(dataset, {})
        if dataset_data:
            score = compute_composite_score(dataset_data)
            scored.append({
                'variant_id': r.get('_variant_id', 'unknown'),
                'init_mode': r.get('_init_mode', 'unknown'),
                'score': score,
                'metrics': dataset_data
            })
    
    if not scored:
        return {}
    
    scored.sort(key=lambda x: x['score'])
    
    return {
        'best': scored[0],
        'worst': scored[-1],
        'top5': scored[:5],
        'bottom5': scored[-5:]
    }

def print_summary(results: List[Dict[str, Any]]):
    """Print comprehensive summary."""
    print("=" * 80)
    print("VARIANT EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal result files: {len(results)}")
    
    # Count by init mode
    pretrained_count = sum(1 for r in results if r.get('_init_mode') == 'pretrained')
    random_count = sum(1 for r in results if r.get('_init_mode') == 'random')
    print(f"  - Pretrained: {pretrained_count}")
    print(f"  - Random: {random_count}")
    
    # ImageNet statistics
    print("\n" + "=" * 80)
    print("IMAGENET METRICS STATISTICS")
    print("=" * 80)
    imagenet_stats = compute_statistics(results, "imagenet")
    for metric, stats in sorted(imagenet_stats.items()):
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Min:    {stats['min']:.4f} (best)")
        print(f"  Max:    {stats['max']:.4f} (worst)")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Count:  {stats['count']}")
    
    # CIFAR-10 statistics
    print("\n" + "=" * 80)
    print("CIFAR-10 METRICS STATISTICS")
    print("=" * 80)
    cifar_stats = compute_statistics(results, "cifar10")
    for metric, stats in sorted(cifar_stats.items()):
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Min:    {stats['min']:.4f} (best)")
        print(f"  Max:    {stats['max']:.4f} (worst)")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Count:  {stats['count']}")
    
    # Best/Worst models (ImageNet)
    print("\n" + "=" * 80)
    print("BEST/WORST MODELS (ImageNet Composite Score)")
    print("=" * 80)
    best_worst = find_best_worst(results, "imagenet")
    if best_worst:
        print(f"\n🏆 BEST MODEL (lowest composite score):")
        best = best_worst['best']
        print(f"  Variant: {best['variant_id']} ({best['init_mode']})")
        print(f"  Composite Score: {best['score']:.4f}")
        print(f"  Metrics:")
        for metric, value in sorted(best['metrics'].items()):
            print(f"    {metric}: {value:.4f}")
        
        print(f"\n❌ WORST MODEL (highest composite score):")
        worst = best_worst['worst']
        print(f"  Variant: {worst['variant_id']} ({worst['init_mode']})")
        print(f"  Composite Score: {worst['score']:.4f}")
        print(f"  Metrics:")
        for metric, value in sorted(worst['metrics'].items()):
            print(f"    {metric}: {value:.4f}")
        
        print(f"\n📊 TOP 5 MODELS (ImageNet):")
        for i, model in enumerate(best_worst['top5'], 1):
            print(f"  {i}. {model['variant_id']} ({model['init_mode']}): score={model['score']:.4f}")
    
    # Best/Worst models (CIFAR-10)
    print("\n" + "=" * 80)
    print("BEST/WORST MODELS (CIFAR-10 Composite Score)")
    print("=" * 80)
    best_worst_cifar = find_best_worst(results, "cifar10")
    if best_worst_cifar:
        print(f"\n🏆 BEST MODEL (lowest composite score):")
        best = best_worst_cifar['best']
        print(f"  Variant: {best['variant_id']} ({best['init_mode']})")
        print(f"  Composite Score: {best['score']:.4f}")
        
        print(f"\n📊 TOP 5 MODELS (CIFAR-10):")
        for i, model in enumerate(best_worst_cifar['top5'], 1):
            print(f"  {i}. {model['variant_id']} ({model['init_mode']}): score={model['score']:.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    results = load_all_results("variant_0")
    if results:
        print_summary(results)
    else:
        print("No results found in variant_0 folder!")

