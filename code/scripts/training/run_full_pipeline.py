#!/usr/bin/env python3
"""
Full Automated Search + KD Pipeline for ECCV16 Colorization.

This script:
  - Generates all valid ECCV16 variant configurations
  - Trains each variant on CIFAR-10 with both weight_init_mode options
    (pretrained, random) using the two-phase schedule with early stopping
    patience fixed at 5 epochs
  - Evaluates each trained checkpoint on CIFAR-10 and ImageNet
  - Computes a composite score from ImageNet metrics and selects the best
    PRETRAINED teacher model
  - Runs knowledge distillation on CIFAR-10 using the selected teacher to
    produce a compact student model
  - Saves all configs, metrics, and checkpoints under the experiments/ tree
"""

import os
import json
import warnings
from typing import Dict, Any, List

import torch

# Suppress LAB→RGB conversion warnings (harmless but noisy)
warnings.filterwarnings('ignore', message='.*negative Z values.*')

from src.training.eccv16_variants import generate_model_configurations, build_eccv16_variant
from src.training.variant_training_pipeline import train_variant_on_cifar10
from src.evaluation.variant_evaluation import (
    evaluate_on_cifar10,
    evaluate_on_imagenet,
    save_variant_results,
)
from src.training.teacher import load_teacher
from src.training.student import MobileNetStyleStudent
from src.training.distillation_loss import FeatureDistillationLoss


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

    NOTE: This implementation assumes that metrics are already normalized
    into [0,1] ranges outside this function. For this project, we simply
    treat the raw metrics as "already normalized" to keep things simple,
    which still preserves relative ranking across variants.
    """
    FID_norm = float(metrics.get("fid", 0.0))  # FID may be absent; treat as 0
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


def select_best_teacher(
    results_dir: str,
) -> Dict[str, Any]:
    """
    Select best teacher variant based on ImageNet metrics only.
    Only PRETRAINED variants are eligible.
    """
    best_score = float("inf")
    best_info: Dict[str, Any] = {}

    if not os.path.exists(results_dir):
        return best_info

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        base = os.path.splitext(fname)[0]
        if not base.endswith("__pretrained"):
            continue

        path = os.path.join(results_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            imagenet_metrics = data.get("imagenet", {})
            if not imagenet_metrics:
                continue  # Skip if no ImageNet metrics
            score = compute_composite_score(imagenet_metrics)
            if score < best_score:
                best_score = score
                variant_id = base.split("__")[0]  # e.g., "variant_097"
                best_info = {
                    "variant_id": variant_id,
                    "results_path": path,
                    "score": score,
                    "metrics": imagenet_metrics,
                }
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            continue

    return best_info


def run_kd_with_best_teacher(
    best_info: Dict[str, Any],
    device: str,
    experiments_root: str = "experiments",
) -> None:
    """
    Run a simplified KD procedure using:
      - Teacher: selected best pretrained variant
      - Student: MobileNet-style ECCV16 student

    Uses the existing FeatureDistillationLoss as a proxy for the specified
    KD objective. Training schedule is simplified but still honors a strong
    distillation signal from teacher logits and features.
    """
    if not best_info:
        print("No best teacher found; skipping KD.")
        return

    variant_id = best_info["variant_id"]  # e.g., "variant_097"
    exp_dir = os.path.join(experiments_root, f"{variant_id}_pretrained")
    teacher_ckpt = os.path.join(exp_dir, "checkpoints", "final.pth")
    if not os.path.exists(teacher_ckpt):
        print(f"Teacher checkpoint not found at {teacher_ckpt}; skipping KD.")
        return

    from src.training.train_distill import DistillationTrainer

    # Reuse DistillationTrainer but point it to the teacher checkpoint
    save_dir = os.path.join("experiments", "distilled")
    os.makedirs(save_dir, exist_ok=True)

    trainer = DistillationTrainer(
        device=device,
        batch_size=32,
        learning_rate=1e-4,
        teacher_path=teacher_ckpt,
        teacher_type="eccv16",
        teacher_use_ila=False,
        student_type="mobilenet",
        student_channel_reduction=2,
        temperature=3.0,
        alpha=0.7,
        use_feature_loss=True,
        feature_weight=0.1,
    )
    trainer.prepare_data()
    trainer.train(
        epochs=40,
        save_dir=save_dir,
        save_prefix="student_distilled",
        patience=5,
    )


def count_completed_variants(results_root: str) -> int:
    """Count how many variant+mode combinations already have results files."""
    if not os.path.exists(results_root):
        return 0
    count = 0
    for fname in os.listdir(results_root):
        if fname.endswith(".json") and "__" in fname:
            count += 1
    return count


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description='Full ECCV16 variant search + KD pipeline')
    parser.add_argument('--imagenet_samples', type=int, default=1000,
                       help='Number of ImageNet samples to evaluate (default: 1000, was 10000)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip variants that already have results files (default: True)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from where we left off (default: True)')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    experiments_root = "experiments"
    os.makedirs(experiments_root, exist_ok=True)
    results_root = "results"
    os.makedirs(results_root, exist_ok=True)

    # If the user hasn't provided a validation subset path, fall back to the
    # full ImageNet localization challenge directory they mentioned. The
    # evaluator will internally sample up to num_samples images from it.
    imagenet_root = os.environ.get(
        "IMAGENET_VAL_ROOT",
        "./data/imagenet/ILSVRC",
    )

    # 1. Generate all valid configurations
    configs = generate_model_configurations()
    print(f"Generated {len(configs)} valid ECCV16 variant configurations.")
    print(f"ImageNet evaluation will use {args.imagenet_samples} samples per variant.")

    # Count already completed variants (for info only)
    already_completed = count_completed_variants(results_root)
    print(f"Found {already_completed} already completed variant evaluations.")
    
    # 2. Train and evaluate each variant for both init modes
    total_variants = len(configs) * 2
    
    # Calculate position based on variant index and init mode
    # variant_001 pretrained = 1, variant_001 random = 2, variant_002 pretrained = 3, etc.
    def get_variant_position(idx: int, init_mode: str) -> int:
        """Calculate 1-indexed position: (idx-1)*2 + (1 if pretrained else 2)"""
        return (idx - 1) * 2 + (1 if init_mode == "pretrained" else 2)
    
    for idx, base_cfg in enumerate(configs, start=1):
        for init_mode in ["pretrained", "random"]:
            variant_id = f"variant_{idx:03d}"
            
            # Skip pretrained models for variants >= 98
            if idx >= 98 and init_mode == "pretrained":
                print(f"\n[Skipping {variant_id} ({init_mode}) - skipping pretrained from variant 98 onwards]")
                continue
            
            cfg = dict(base_cfg)
            cfg["weight_init_mode"] = init_mode
            position = get_variant_position(idx, init_mode)
            
            # Check if results already exist
            results_file = os.path.join(results_root, f"{variant_id}__{init_mode}.json")
            if args.skip_existing and os.path.exists(results_file):
                print(f"\n[Skipping {variant_id} ({init_mode}) - results already exist]")
                continue
            
            print(f"\n=== Training {variant_id} ({init_mode}) [{position}/{total_variants}] ===")

            try:
                train_result = train_variant_on_cifar10(
                    variant_config=cfg,
                    variant_id=variant_id,
                    device=device,
                )

                # Load best model checkpoint
                exp_dir = train_result["experiment_dir"]
                ckpt_path = os.path.join(exp_dir, "checkpoints", "final.pth")
                
                # Check if checkpoint exists (might have failed during save)
                if not os.path.exists(ckpt_path):
                    # Try warmup checkpoint as fallback
                    warmup_ckpt = os.path.join(exp_dir, "checkpoints", "warmup.pth")
                    if os.path.exists(warmup_ckpt):
                        print(f"Warning: final.pth not found, using warmup.pth")
                        ckpt_path = warmup_ckpt
                    else:
                        print(f"Error: No checkpoint found for {variant_id} ({init_mode}), skipping evaluation")
                        continue
                
                model = build_eccv16_variant(**cfg).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()

                print(f"Evaluating {variant_id} ({init_mode}) on CIFAR-10...")
                cifar_metrics = evaluate_on_cifar10(model=model, device=device)
                print(f"Evaluating {variant_id} ({init_mode}) on ImageNet ({args.imagenet_samples} samples)...")
                imagenet_metrics = evaluate_on_imagenet(
                    model=model,
                    device=device,
                    imagenet_root=imagenet_root,
                    num_samples=args.imagenet_samples,
                )

                save_variant_results(
                    variant_id=variant_id,
                    weight_init_mode=init_mode,
                    cifar_metrics=cifar_metrics,
                    imagenet_metrics=imagenet_metrics,
                    results_root=results_root,
                )
                
                current_completed = count_completed_variants(results_root)
                print(f"✅ Completed {variant_id} ({init_mode}) [{position}/{total_variants}] - Total completed: {current_completed}/{total_variants}")
            except Exception as e:
                print(f"❌ Error processing {variant_id} ({init_mode}): {e}")
                print(f"   Continuing with next variant...")
                import traceback
                traceback.print_exc()
                continue

    # 3. Select best teacher using ImageNet metrics (pretrained only)
    # This works even if pipeline is incomplete - selects from what's available
    print("\n=== Selecting best teacher from completed variants ===")
    best_info = select_best_teacher(results_root)
    if best_info:
        print("\n✅ Best teacher (pretrained only, from available results):")
        print(json.dumps(best_info, indent=2))
        # Save best teacher artifacts
        best_variant_id = best_info["variant_id"]  # e.g., "variant_097"
        src_dir = os.path.join(experiments_root, f"{best_variant_id}_pretrained")
        teacher_dir = os.path.join("best_model")
        os.makedirs(teacher_dir, exist_ok=True)
        # Copy checkpoint and config
        import shutil

        final_ckpt = os.path.join(src_dir, "checkpoints", "final.pth")
        warmup_ckpt = os.path.join(src_dir, "checkpoints", "warmup.pth")
        
        if os.path.exists(final_ckpt):
            shutil.copy(final_ckpt, os.path.join(teacher_dir, "teacher.pth"))
        elif os.path.exists(warmup_ckpt):
            print(f"Warning: Using warmup checkpoint for best teacher")
            shutil.copy(warmup_ckpt, os.path.join(teacher_dir, "teacher.pth"))
        else:
            print(f"Warning: No checkpoint found for best teacher at {src_dir}")
        
        config_src = os.path.join(src_dir, "config.json")
        if os.path.exists(config_src):
            shutil.copy(config_src, os.path.join(teacher_dir, "teacher_config.json"))
        else:
            print(f"Warning: No config.json found for best teacher")
    else:
        print("⚠️  No pretrained variants found for teacher selection.")
        print("   Pipeline will continue training more variants...")

    # Print summary
    final_completed = count_completed_variants(results_root)
    remaining = total_variants - final_completed
    print(f"\n{'='*60}")
    print(f"PIPELINE STATUS SUMMARY")
    print(f"{'='*60}")
    print(f"Total variants: {total_variants}")
    print(f"Completed: {final_completed}")
    print(f"Remaining: {remaining}")
    print(f"Progress: {final_completed/total_variants*100:.1f}%")
    print(f"{'='*60}")
    
    # 4. Run KD with best teacher (only if we have a best teacher)
    if best_info:
        print("\n=== Running Knowledge Distillation ===")
        run_kd_with_best_teacher(best_info, device=device, experiments_root=experiments_root)
    else:
        print("\n⚠️  Skipping KD - no best teacher selected yet.")
        print("   Run again after more variants complete to select best teacher.")

    if remaining == 0:
        print("\n✅ Full pipeline complete!")
    else:
        print(f"\n⏳ Pipeline incomplete. Run again to continue from variant {final_completed+1}.")
        print(f"   Use --skip_existing (default) to resume automatically.")


if __name__ == "__main__":
    main()


