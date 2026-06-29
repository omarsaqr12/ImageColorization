# Technical Details

A deep dive into the architectures, training schedule, distillation objective, and evaluation
protocol. For the high-level picture, start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

## 1. Backbones

The work builds on two published colorization networks:

- **ECCV16** — *Colorful Image Colorization* (Zhang, Isola, Efros, 2016). A fully convolutional
  network that predicts the `ab` chrominance channels from the `L` lightness channel in CIELAB
  space, trained as a 313-bin color classification problem with class rebalancing for rare
  colors. **All trained variants in this project extend this backbone.**
- **SIGGRAPH17** — *Real-Time User-Guided Colorization with Learned Deep Priors*. Used here as a
  strong pretrained **baseline** for comparison.

Both live in [`code/colorization/colorizers/`](../code/colorization/colorizers/).

---

## 2. Variant Modules

The ECCV16 backbone is augmented with optional modules
([`eccv16_variants.py`](../code/src/training/eccv16_variants.py)):

| Module | Flag | What it adds |
|--------|------|--------------|
| Multi-scale pooling | `use_multiscale` | Spatial Pyramid Pooling at the bottleneck (1×1, 2×2, 4×4 bins). |
| Global semantic head | `use_global_semantic_head` | Global semantic embedding with FiLM modulation of decoder features. |
| Perceptual loss | `use_perceptual_loss` | VGG feature-matching loss ([`perceptual_loss.py`](../code/src/training/perceptual_loss.py)). |
| SE attention | `use_attention` | Squeeze-and-Excitation channel attention. |
| Adversarial training | `use_gan` | PatchGAN discriminator + adversarial loss ([`discriminator.py`](../code/src/training/discriminator.py), [`gan_loss.py`](../code/src/training/gan_loss.py)). |
| Color classification | `use_color_classification` | 313-bin color-distribution head. |
| Class rebalancing | `use_class_rebalance` | Reweights rare colors during training. |
| Init mode | `weight_init_mode` | `pretrained` vs `random`. |

The ILA (Intermediate Layer Attention) block in
[`ila_block.py`](../code/colorization/colorizers/ila_block.py) is also available on the ECCV16
backbone and was used in earlier milestone experiments.

---

## 3. Training Schedule

Implemented in
[`variant_training_pipeline.py`](../code/src/training/variant_training_pipeline.py). Training is
two-phase with early stopping (patience = 5 epochs):

- **Phase A — Warmup (~10 epochs).** Pretrained variants freeze the encoder; random-init
  variants train end-to-end. GAN weight is kept very small and the learning rate is higher.
- **Phase B — Full training (~60 epochs).** Pretrained encoders are gradually unfrozen, the
  learning rate follows a cosine decay, and GAN / perceptual losses are enabled if selected.

Checkpoints are written to `checkpoints/<variant_id>/warmup.pth` and `.../final.pth`.

### Teacher selection

Each variant is evaluated on a 1000-image ImageNet validation subset and ranked by a composite
score (lower is better), defined in
[`run_full_pipeline.py`](../code/scripts/training/run_full_pipeline.py):

```
score = 0.40·FID + 0.20·LPIPS + 0.20·ΔE
      + 0.10·(1 − colorfulness) + 0.05·(1 − PSNR) + 0.05·(1 − SSIM)
```

The winning configuration was **`variant_097`**:

```json
{
  "use_multiscale": false,
  "use_global_semantic_head": true,
  "use_perceptual_loss": true,
  "use_attention": false,
  "use_gan": true,
  "use_color_classification": true,
  "use_class_rebalance": true,
  "weight_init_mode": "random"
}
```

**Findings:** the Global Semantic Head + Perceptual Loss + GAN combination wins; random
initialization outperformed pretrained on average; multi-scale pooling and SE attention did not
help; color classification with rebalancing improved rare-color reproduction.

---

## 4. Knowledge Distillation

The selected teacher (32.24M params, 122.97 MB) is distilled into compact students
([`student.py`](../code/src/training/student.py),
[`train_distill.py`](../code/src/training/train_distill.py)).

- **Student family:** MobileNet-style and lightweight ECCV16 variants with channel-reduction
  factors of 2×, 4×, and 8×.
- **Objective** ([`distillation_loss.py`](../code/src/training/distillation_loss.py)): a blend of
  the task loss and a distillation term on teacher logits/probabilities (temperature `T = 3.0`,
  `alpha = 0.7`) plus an intermediate **feature-matching** loss (`feature_weight = 0.1`).

### Compression & quality (CIFAR-10)

| Model | Params | Size | Compression | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|------|-------------|--------|--------|---------|
| Teacher | 32.24M | 122.97 MB | 1× | 23.04 dB | 0.926 | 0.074 |
| MobileNet-2x | 0.24M | 0.93 MB | 132× | 24.75 dB | 0.933 | 0.056 |
| MobileNet-4x | 0.07M | 0.26 MB | 466× | 24.77 dB | 0.933 | 0.057 |
| MobileNet-8x | 0.02M | 0.08 MB | **1467×** | 24.68 dB | 0.932 | 0.057 |

Every student **outperforms** the teacher on all three metrics — the distillation signal plus a
smaller, better-regularized hypothesis space generalizes better on 32×32 CIFAR-10 images.

---

## 5. Evaluation Metrics

Evaluation code lives in [`code/src/evaluation/`](../code/src/evaluation/).

| Metric | Type | Direction |
|--------|------|-----------|
| PSNR | Pixel | Higher ↑ |
| SSIM | Pixel/structural | Higher ↑ |
| LPIPS | Perceptual | Lower ↓ |
| FID | Distributional | Lower ↓ |
| KID | Distributional | Lower ↓ |
| ΔE2000 | Color (CIELAB) | Lower ↓ |
| FSIM | Feature similarity | Higher ↑ |

### ImageNet milestone comparison

| Metric | Milestone baseline | Best trained ECCV16 combo | Change |
|--------|--------------------|---------------------------|--------|
| FID ↓ | 41.47 | **21.26** | −49% |
| PSNR ↑ | 19.68 | 21.17 | +1.49 dB |
| LPIPS ↓ | 0.217 | 0.165 | −24% |
| ΔE2000 ↓ | 21.12 | 20.73 | −0.39 |

(See [`compare_with_milestone.py`](../code/scripts/analysis/compare_with_milestone.py).)

---

## 6. Datasets

- **CIFAR-10** — auto-downloaded via torchvision; primary training/evaluation set.
- **ImageNet** — validation subset (1000 images by default) for FID/perceptual evaluation. Point
  the code at your local copy with the `IMAGENET_VAL_ROOT` environment variable.

---

## 7. Reproducing

```bash
# Full search + distillation pipeline
python code/scripts/training/run_full_pipeline.py --imagenet_samples 1000

# Train one variant
python code/src/training/variant_training_pipeline.py --variant_id 097 --init_mode random

# Distill students at multiple sizes
python code/scripts/training/train_multiple_student_sizes.py

# Evaluate + rank
python code/scripts/evaluation/evaluate_all_models_comprehensive.py
python code/scripts/analysis/find_best_models_by_metric.py
```
