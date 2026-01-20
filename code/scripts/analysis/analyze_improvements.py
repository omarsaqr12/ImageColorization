#!/usr/bin/env python3
"""Analyze if improvements are significant enough."""

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
                basename = os.path.basename(fpath)
                parts = basename.replace('.json', '').split('__')
                if len(parts) == 2:
                    variant_id = parts[0]
                    init_mode = parts[1]
                    data['_variant_id'] = variant_id
                    data['_init_mode'] = init_mode
                    results.append(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
    
    return results

def analyze_improvements(results: List[Dict[str, Any]]):
    """Analyze if improvements are significant."""
    
    print("=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    # Separate by init mode
    pretrained = [r for r in results if r.get('_init_mode') == 'pretrained']
    random_init = [r for r in results if r.get('_init_mode') == 'random']
    
    print(f"\nPretrained models: {len(pretrained)}")
    print(f"Random init models: {len(random_init)}")
    
    # Analyze ImageNet metrics
    print("\n" + "=" * 80)
    print("IMAGENET METRICS COMPARISON")
    print("=" * 80)
    
    metrics_to_analyze = ['psnr', 'ssim', 'lpips', 'deltaE', 'colorfulness']
    
    for metric in metrics_to_analyze:
        pretrained_vals = [r['imagenet'].get(metric, 0) for r in pretrained if 'imagenet' in r]
        random_vals = [r['imagenet'].get(metric, 0) for r in random_init if 'imagenet' in r]
        
        if not pretrained_vals or not random_vals:
            continue
            
        pretrained_mean = statistics.mean(pretrained_vals)
        random_mean = statistics.mean(random_vals)
        pretrained_best = max(pretrained_vals) if metric in ['psnr', 'ssim'] else min(pretrained_vals)
        random_best = max(random_vals) if metric in ['psnr', 'ssim'] else min(random_vals)
        
        # Determine if higher or lower is better
        higher_is_better = metric in ['psnr', 'ssim', 'colorfulness']
        
        if higher_is_better:
            best_overall = max(pretrained_best, random_best)
            improvement = best_overall - pretrained_mean
            improvement_pct = (improvement / pretrained_mean) * 100 if pretrained_mean > 0 else 0
        else:
            best_overall = min(pretrained_best, random_best)
            improvement = pretrained_mean - best_overall
            improvement_pct = (improvement / pretrained_mean) * 100 if pretrained_mean > 0 else 0
        
        print(f"\n{metric.upper()}:")
        print(f"  Pretrained mean: {pretrained_mean:.4f}")
        print(f"  Random mean:     {random_mean:.4f}")
        print(f"  Best overall:   {best_overall:.4f}")
        print(f"  Improvement:    {improvement:.4f} ({improvement_pct:+.2f}%)")
        
        if higher_is_better:
            if random_mean > pretrained_mean:
                print(f"  ⚠️  Random init performs BETTER on average!")
            else:
                print(f"  ✅ Pretrained performs better on average")
        else:
            if random_mean < pretrained_mean:
                print(f"  ⚠️  Random init performs BETTER on average!")
            else:
                print(f"  ✅ Pretrained performs better on average")
    
    # Compare best vs worst
    print("\n" + "=" * 80)
    print("BEST vs WORST COMPARISON (ImageNet)")
    print("=" * 80)
    
    all_imagenet = []
    for r in results:
        if 'imagenet' in r:
            all_imagenet.append(r['imagenet'])
    
    for metric in metrics_to_analyze:
        vals = [r.get(metric, 0) for r in all_imagenet if metric in r]
        if not vals:
            continue
            
        best_val = max(vals) if metric in ['psnr', 'ssim', 'colorfulness'] else min(vals)
        worst_val = min(vals) if metric in ['psnr', 'ssim', 'colorfulness'] else max(vals)
        mean_val = statistics.mean(vals)
        
        if metric in ['psnr', 'ssim', 'colorfulness']:
            improvement = best_val - worst_val
            improvement_pct = (improvement / worst_val) * 100 if worst_val > 0 else 0
        else:
            improvement = worst_val - best_val
            improvement_pct = (improvement / worst_val) * 100 if worst_val > 0 else 0
        
        print(f"\n{metric.upper()}:")
        print(f"  Best:  {best_val:.4f}")
        print(f"  Worst: {worst_val:.4f}")
        print(f"  Mean:  {mean_val:.4f}")
        print(f"  Range: {improvement:.4f} ({improvement_pct:+.2f}% improvement)")
    
    # Statistical significance check
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE CHECK")
    print("=" * 80)
    
    # Compare pretrained vs random for each metric
    for metric in metrics_to_analyze:
        pretrained_vals = [r['imagenet'].get(metric, 0) for r in pretrained if 'imagenet' in r and metric in r['imagenet']]
        random_vals = [r['imagenet'].get(metric, 0) for r in random_init if 'imagenet' in r and metric in r['imagenet']]
        
        if len(pretrained_vals) < 2 or len(random_vals) < 2:
            continue
        
        pretrained_mean = statistics.mean(pretrained_vals)
        random_mean = statistics.mean(random_vals)
        pretrained_std = statistics.stdev(pretrained_vals) if len(pretrained_vals) > 1 else 0
        random_std = statistics.stdev(random_vals) if len(random_vals) > 1 else 0
        
        # Simple t-test approximation (effect size)
        pooled_std = ((pretrained_std ** 2 + random_std ** 2) / 2) ** 0.5
        if pooled_std > 0:
            effect_size = abs(pretrained_mean - random_mean) / pooled_std
        else:
            effect_size = 0
        
        print(f"\n{metric.upper()}:")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
        if effect_size < 0.2:
            print(f"  → Negligible difference")
        elif effect_size < 0.5:
            print(f"  → Small difference")
        elif effect_size < 0.8:
            print(f"  → Medium difference")
        else:
            print(f"  → Large difference")
    
    # Practical significance assessment
    print("\n" + "=" * 80)
    print("PRACTICAL SIGNIFICANCE ASSESSMENT")
    print("=" * 80)
    
    # For colorization, typical good values:
    # PSNR: >20 is good, >22 is excellent
    # SSIM: >0.8 is good, >0.9 is excellent  
    # LPIPS: <0.2 is good, <0.15 is excellent
    # ΔE: <20 is good, <15 is excellent
    
    best_psnr = max([r['imagenet'].get('psnr', 0) for r in results if 'imagenet' in r])
    best_ssim = max([r['imagenet'].get('ssim', 0) for r in results if 'imagenet' in r])
    best_lpips = min([r['imagenet'].get('lpips', 1) for r in results if 'imagenet' in r])
    best_de = min([r['imagenet'].get('deltaE', 100) for r in results if 'imagenet' in r])
    
    print(f"\nBest achieved values:")
    print(f"  PSNR: {best_psnr:.2f} {'✅ Excellent' if best_psnr > 22 else '✅ Good' if best_psnr > 20 else '⚠️  Below 20'}")
    print(f"  SSIM: {best_ssim:.3f} {'✅ Excellent' if best_ssim > 0.9 else '✅ Good' if best_ssim > 0.8 else '⚠️  Below 0.8'}")
    print(f"  LPIPS: {best_lpips:.3f} {'✅ Excellent' if best_lpips < 0.15 else '✅ Good' if best_lpips < 0.2 else '⚠️  Above 0.2'}")
    print(f"  ΔE: {best_de:.2f} {'✅ Excellent' if best_de < 15 else '✅ Good' if best_de < 20 else '⚠️  Above 20'}")
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    excellent_count = sum([
        best_psnr > 22,
        best_ssim > 0.9,
        best_lpips < 0.15,
        best_de < 15
    ])
    
    good_count = sum([
        20 < best_psnr <= 22,
        0.8 < best_ssim <= 0.9,
        0.15 <= best_lpips < 0.2,
        15 <= best_de < 20
    ])
    
    print(f"\nMetrics achieving 'Excellent' threshold: {excellent_count}/4")
    print(f"Metrics achieving 'Good' threshold: {good_count}/4")
    
    if excellent_count >= 3:
        print("\n✅ IMPROVEMENTS ARE SIGNIFICANT - Multiple metrics in excellent range")
    elif excellent_count >= 2 or good_count >= 3:
        print("\n✅ IMPROVEMENTS ARE MODERATE - Some metrics in excellent/good range")
    elif good_count >= 2:
        print("\n⚠️  IMPROVEMENTS ARE MINIMAL - Most metrics only in good range")
    else:
        print("\n❌ IMPROVEMENTS ARE INSUFFICIENT - Most metrics below good thresholds")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    results = load_all_results("variant_0")
    if results:
        analyze_improvements(results)
    else:
        print("No results found!")

