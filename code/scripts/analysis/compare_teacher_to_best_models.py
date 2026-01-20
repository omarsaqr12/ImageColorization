#!/usr/bin/env python3
"""
Compare Current KD Teacher Model to Best Models by Metric

This script:
1. Identifies the current teacher model used for KD (variant_097_random)
2. Loads best_models_by_metric.json (if exists) or analyzes results
3. Compares the teacher's performance to the top 5 models for each metric
"""

import os
import json
from typing import Dict, List, Any

# Current teacher used for KD
CURRENT_TEACHER = {
    "variant_id": "variant_097",
    "init_mode": "random",
    "checkpoint": "experiments/variant_097_random/checkpoints/final.pth"
}


def load_teacher_metrics(results_dir: str = "results") -> Dict[str, Any]:
    """Load metrics for the current teacher model."""
    teacher_file = os.path.join(results_dir, f"{CURRENT_TEACHER['variant_id']}__{CURRENT_TEACHER['init_mode']}.json")
    
    if not os.path.exists(teacher_file):
        print(f"❌ Teacher result file not found: {teacher_file}")
        return {}
    
    with open(teacher_file, "r") as f:
        data = json.load(f)
    
    return {
        "variant_id": CURRENT_TEACHER['variant_id'],
        "init_mode": CURRENT_TEACHER['init_mode'],
        "cifar10": data.get("cifar10", {}),
        "imagenet": data.get("imagenet", {})
    }


def load_best_models_json(json_file: str = "best_models_by_metric.json") -> Dict[str, Any]:
    """Load best models from JSON file if it exists."""
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    return {}


def compare_teacher_to_best(teacher_metrics: Dict[str, Any], best_models: Dict[str, Any]):
    """Compare teacher metrics to best models for each metric."""
    
    print("="*80)
    print("COMPARING KD TEACHER TO BEST MODELS BY METRIC")
    print("="*80)
    
    print(f"\n📚 Current KD Teacher: {CURRENT_TEACHER['variant_id']} ({CURRENT_TEACHER['init_mode']})")
    print(f"   Checkpoint: {CURRENT_TEACHER['checkpoint']}\n")
    
    if not teacher_metrics:
        print("❌ Could not load teacher metrics!")
        return
    
    # Metrics to compare
    metrics_config = [
        ("psnr", "imagenet", True, "dB"),
        ("ssim", "imagenet", True, ""),
        ("lpips", "imagenet", False, ""),
        ("deltaE", "imagenet", False, ""),
        ("colorfulness", "imagenet", True, ""),
        ("psnr", "cifar10", True, "dB"),
        ("ssim", "cifar10", True, ""),
        ("lpips", "cifar10", False, ""),
        ("deltaE", "cifar10", False, ""),
        ("colorfulness", "cifar10", True, ""),
    ]
    
    for metric_name, dataset, higher_is_better, unit in metrics_config:
        key = f"{dataset}_{metric_name}"
        
        print(f"\n{'='*80}")
        print(f"{metric_name.upper()} ({dataset.upper()})")
        print(f"{'='*80}")
        
        # Get teacher value
        teacher_value = teacher_metrics.get(dataset, {}).get(metric_name)
        if teacher_value is None:
            print(f"   Teacher: No data")
            continue
        
        print(f"   Teacher ({CURRENT_TEACHER['variant_id']} {CURRENT_TEACHER['init_mode']}): {teacher_value:.4f} {unit}")
        
        # Get best models
        if key in best_models and best_models[key]:
            top5 = best_models[key]
            
            print(f"\n   Top 5 Models:")
            print(f"   {'Rank':<6} {'Variant':<15} {'Init':<12} {'Value':<12} {'vs Teacher':<12}")
            print(f"   {'-'*60}")
            
            for i, model in enumerate(top5, 1):
                best_value = model.get("metric_value", 0)
                diff = best_value - teacher_value
                diff_pct = (diff / teacher_value * 100) if teacher_value != 0 else 0
                
                # Determine if teacher is in top 5
                is_teacher = (model.get("variant_id") == CURRENT_TEACHER['variant_id'] and 
                             model.get("init_mode") == CURRENT_TEACHER['init_mode'])
                
                marker = "👈 TEACHER" if is_teacher else ""
                
                if higher_is_better:
                    comparison = f"+{diff:.4f} ({diff_pct:+.2f}%)" if diff > 0 else f"{diff:.4f} ({diff_pct:.2f}%)"
                else:
                    comparison = f"{diff:.4f} ({diff_pct:.2f}%)" if diff < 0 else f"+{diff:.4f} ({diff_pct:+.2f}%)"
                
                print(f"   {i:<6} {model.get('variant_id', 'N/A'):<15} {model.get('init_mode', 'N/A'):<12} "
                     f"{best_value:<12.4f} {comparison:<12} {marker}")
            
            # Check teacher's rank
            teacher_rank = None
            for i, model in enumerate(top5, 1):
                if (model.get("variant_id") == CURRENT_TEACHER['variant_id'] and 
                    model.get("init_mode") == CURRENT_TEACHER['init_mode']):
                    teacher_rank = i
                    break
            
            if teacher_rank:
                print(f"\n   ✅ Teacher is ranked #{teacher_rank} in top 5!")
            else:
                # Find where teacher would rank
                all_values = [m.get("metric_value", 0) for m in top5]
                if higher_is_better:
                    better_count = sum(1 for v in all_values if v > teacher_value)
                else:
                    better_count = sum(1 for v in all_values if v < teacher_value)
                
                if better_count < 5:
                    print(f"\n   ⚠️  Teacher is NOT in top 5 (would be rank #{better_count + 1})")
                else:
                    print(f"\n   ❌ Teacher is NOT in top 5 (rank > 5)")
        else:
            print(f"   ⚠️  No best models data available for {key}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    teacher_in_top5_count = 0
    total_metrics = 0
    
    for metric_name, dataset, higher_is_better, unit in metrics_config:
        key = f"{dataset}_{metric_name}"
        if key in best_models and best_models[key]:
            total_metrics += 1
            for model in best_models[key]:
                if (model.get("variant_id") == CURRENT_TEACHER['variant_id'] and 
                    model.get("init_mode") == CURRENT_TEACHER['init_mode']):
                    teacher_in_top5_count += 1
                    break
    
    print(f"\nTeacher appears in top 5 for {teacher_in_top5_count}/{total_metrics} metrics")
    
    if teacher_in_top5_count == total_metrics:
        print("✅ Teacher is excellent - appears in top 5 for all metrics!")
    elif teacher_in_top5_count >= total_metrics * 0.7:
        print("✅ Teacher is very good - appears in top 5 for most metrics")
    elif teacher_in_top5_count >= total_metrics * 0.5:
        print("⚠️  Teacher is good but not optimal - consider alternatives")
    else:
        print("❌ Teacher may not be the best choice - consider using a better model")


def main():
    # Load teacher metrics
    teacher_metrics = load_teacher_metrics()
    
    # Try to load best models JSON
    best_models = load_best_models_json()
    
    if not best_models:
        print("⚠️  best_models_by_metric.json not found.")
        print("   Run 'python find_best_models_by_metric.py' first to generate it.")
        print("\n   Showing teacher metrics only:\n")
        if teacher_metrics:
            print(f"Teacher: {CURRENT_TEACHER['variant_id']} ({CURRENT_TEACHER['init_mode']})")
            print(f"\nImageNet Metrics:")
            img = teacher_metrics.get("imagenet", {})
            for key, value in img.items():
                print(f"  {key}: {value:.4f}")
            print(f"\nCIFAR-10 Metrics:")
            cifar = teacher_metrics.get("cifar10", {})
            for key, value in cifar.items():
                print(f"  {key}: {value:.4f}")
        return
    
    # Compare
    compare_teacher_to_best(teacher_metrics, best_models)


if __name__ == "__main__":
    main()
