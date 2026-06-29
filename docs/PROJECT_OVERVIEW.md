# Project Overview

This project studies **automatic image colorization** and **model compression via knowledge
distillation**. It has two parts:

1. **A systematic architecture search** over colorization models built on the ECCV16 backbone
   (Zhang et al.), exploring attention, GAN, perceptual-loss, and multi-scale modules to find
   the strongest "teacher".
2. **Knowledge distillation** of that teacher into compact MobileNet-style students that are
   100–1400× smaller while matching (and on CIFAR-10, exceeding) the teacher's quality.

---

## Pipeline at a Glance

```
generate variants ─▶ train each on CIFAR-10 ─▶ evaluate on CIFAR-10 + ImageNet
        │                                                   │
        │                                          composite score (FID-weighted)
        ▼                                                   ▼
   168 valid configs                              select best teacher (variant_097)
        │                                                   │
        └───────────────────────────────────────▶ distill ▶ MobileNet students (2x / 4x / 8x)
```

End-to-end entry point: [`code/scripts/training/run_full_pipeline.py`](../code/scripts/training/run_full_pipeline.py).

---

## Directory Map

| Path | Purpose |
|------|---------|
| `code/colorization/colorizers/` | Base ECCV16 / SIGGRAPH17 architectures + the ILA (Intermediate Layer Attention) block and preprocessing utilities. |
| `code/src/training/` | Variant builder, two-phase training pipeline, teacher/student wrappers, distillation / GAN / perceptual losses, discriminator. |
| `code/src/evaluation/` | CIFAR-10 and ImageNet evaluation with PSNR, SSIM, LPIPS, FID, KID, ΔE2000, FSIM. |
| `code/src/visualization/` | Plotting utilities (training curves, comparison grids, CDFs). |
| `code/src/utils/` | Dataset-path discovery, weight management, analysis helpers. |
| `code/src/tests/` | Smoke tests for models, the ILA integration, and dataset paths. |
| `code/scripts/` | Runnable entry points grouped by `training/`, `evaluation/`, `analysis/`, `colorization/`, `visualization/`. |
| `models/` | Pre-trained weights (not in git — download link in the README). |
| `docs/` | This overview and the technical deep-dive. |

---

## The Variant Search

Each variant toggles eight feature flags (see
[`code/src/training/eccv16_variants.py`](../code/src/training/eccv16_variants.py)):

`use_multiscale`, `use_global_semantic_head`, `use_perceptual_loss`, `use_attention`,
`use_gan`, `use_color_classification`, `use_class_rebalance`, and `weight_init_mode`
(`pretrained` / `random`).

Two validity constraints prune the search space:

- `use_color_classification = True` forces `use_class_rebalance = True`.
- `use_gan = True` requires `use_perceptual_loss` **or** `use_attention`.

This yields **168 valid configurations**. The full pipeline trains the lower-index variants
under both initialization modes and the remainder under random init, for **~265 training runs**
in total. Each model is scored on a FID-weighted composite of ImageNet metrics, and the best
**teacher** is selected automatically.

---

## Key Results

- **Best teacher** (`variant_097`): Global Semantic Head + Perceptual Loss + GAN + color
  classification with class rebalancing, trained from random init.
- **FID** on the ImageNet subset dropped from a **41.47** milestone baseline to **21.26** for the
  best trained ECCV16 combination — roughly a **49% reduction**.
- **Distillation**: the 32.24M-parameter / 122.97 MB teacher was compressed up to **1467×**
  (MobileNet-8x, 0.08 MB). All student sizes **beat the teacher** on PSNR, SSIM, and LPIPS on
  CIFAR-10.

See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for architecture, training schedule, and the
full results tables.
