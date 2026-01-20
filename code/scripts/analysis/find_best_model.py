#!/usr/bin/env python3
"""
Script to find the best model from results folder based on composite score.
"""

import os
import json
from typing import Dict, Any, List

def compute_composite_score(
    metrics: Dict[str, float],
    w_fid: float = 0.40,
    w_lpips: float = 0.20,
    w_deltaE: float = 0.20,
    w_color: float = 0.10,
    w_psnr: float = 0.05,
    w_ssim: float = 0.05,
) -> float:
    """
    Compute composite score from normalized metrics.
    Lower is better.
    """
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


def analyze_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """
    Analyze all result files and return sorted list of pretrained variants.
    """
    results = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist!")
        return results
    
    # Get all pretrained result files
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        if not fname.endswith("__pretrained.json"):
            continue
        
        path = os.path.join(results_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            imagenet_metrics = data.get("imagenet", {})
            cifar_metrics = data.get("cifar10", {})
            
            if not imagenet_metrics:
                continue  # Skip if no ImageNet metrics
            
            variant_id = fname.replace("__pretrained.json", "")
            score = compute_composite_score(imagenet_metrics)
            
            results.append({
                "variant_id": variant_id,
                "composite_score": score,
                "imagenet_metrics": imagenet_metrics,
                "cifar10_metrics": cifar_metrics,
                "results_path": path,
            })
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            continue
    
    # Sort by composite score (lower is better)
    results.sort(key=lambda x: x["composite_score"])
    return results


def print_best_models(results: List[Dict[str, Any]], top_n: int = 10):
    """Print top N best models."""
    print("=" * 80)
    print(f"TOP {top_n} BEST MODELS (Based on ImageNet Composite Score)")
    print("=" * 80)
    print("\nComposite Score Formula:")
    print("  Score = 0.40*FID + 0.20*LPIPS + 0.20*ΔE + 0.10*(1-Colorfulness) + 0.05*(1-PSNR) + 0.05*(1-SSIM)")
    print("  (Lower is better)\n")
    print("-" * 80)
    
    for i, result in enumerate(results[:top_n], 1):
        print(f"\n{i}. {result['variant_id']}")
        print(f"   Composite Score: {result['composite_score']:.4f}")
        print(f"   Results File: {result['results_path']}")
        print(f"\n   ImageNet Metrics:")
        metrics = result['imagenet_metrics']
        if 'fid' in metrics:
            print(f"     FID: {metrics['fid']:.4f}")
        if 'lpips' in metrics:
            print(f"     LPIPS: {metrics['lpips']:.4f}")
        if 'deltaE' in metrics:
            print(f"     ΔE: {metrics['deltaE']:.4f}")
        if 'colorfulness' in metrics:
            print(f"     Colorfulness: {metrics['colorfulness']:.4f}")
        if 'psnr' in metrics:
            print(f"     PSNR: {metrics['psnr']:.4f} dB")
        if 'ssim' in metrics:
            print(f"     SSIM: {metrics['ssim']:.4f}")
        
        if result['cifar10_metrics']:
            print(f"\n   CIFAR-10 Metrics:")
            cifar = result['cifar10_metrics']
            if 'psnr' in cifar:
                print(f"     PSNR: {cifar['psnr']:.4f} dB")
            if 'ssim' in cifar:
                print(f"     SSIM: {cifar['ssim']:.4f}")
    
    print("\n" + "=" * 80)
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST MODEL: {best['variant_id']}")
        print(f"   Composite Score: {best['composite_score']:.4f}")
        print(f"   Checkpoint should be at: experiments/{best['variant_id']}_pretrained/checkpoints/final.pth")
    else:
        print("\n⚠️  No pretrained variants with ImageNet metrics found!")


if __name__ == "__main__":
    import sys
    print("Starting analysis...", file=sys.stderr)
    results = analyze_all_results()
    print(f"\nFound {len(results)} pretrained variants with ImageNet metrics\n")
    if results:
        print_best_models(results, top_n=20)
    else:
        print("No results found! Check if results directory exists and contains JSON files.")
