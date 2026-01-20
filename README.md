# 🎨 Image Colorization with Knowledge Distillation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**Advanced Machine Learning Final Project**

*Automatic image colorization using deep learning with knowledge distillation for model compression*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Models](#-models) • [Results](#-results)

</div>

---

## 📋 Overview

This project implements state-of-the-art automatic image colorization models based on **ECCV16** and **SIGGRAPH17** architectures by Zhang et al. We extend these baselines with:

- **200+ variant models** with different architectural configurations
- **Knowledge distillation** to create lightweight, deployable student models
- **Comprehensive evaluation** using multiple perceptual and pixel-level metrics
- **Multiple loss functions** including perceptual, GAN, and distillation losses

### 🎯 Key Achievements

| Metric | Best Variant | Baseline ECCV16 | Improvement |
|--------|--------------|-----------------|-------------|
| PSNR ↑ | 27.42 dB | 25.31 dB | +8.3% |
| SSIM ↑ | 0.934 | 0.912 | +2.4% |
| LPIPS ↓ | 0.089 | 0.124 | +28.2% |

---

## ✨ Features

- 🔥 **Multiple Architectures**: ECCV16, SIGGRAPH17, and 200+ custom variants
- 🎓 **Knowledge Distillation**: Compress models to 2x, 4x, and 8x smaller sizes
- 📊 **Comprehensive Metrics**: PSNR, SSIM, LPIPS, FID, KID, ΔE2000, FSIM
- 🖼️ **Dataset Support**: CIFAR-10 and ImageNet
- ⚡ **GPU Accelerated**: Full CUDA support for training and inference
- 📈 **Visualization Tools**: Training curves, comparison plots, CDF analysis

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB RAM (for full pipeline)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/image-colorization-distillation.git
   cd image-colorization-distillation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install PyTorch** (with CUDA support)
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CPU only
   pip install torch torchvision torchaudio
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

---

## 📁 Project Structure

```
├── code/
│   ├── colorization/           # Base colorization library (ECCV16/SIGGRAPH17)
│   │   ├── colorizers/         # Model implementations
│   │   │   ├── eccv16.py       # ECCV16 architecture
│   │   │   ├── siggraph17.py   # SIGGRAPH17 architecture
│   │   │   ├── ila_block.py    # Intermediate Layer Attention
│   │   │   └── util.py         # Preprocessing utilities
│   │   └── demo_release.py     # Demo script
│   │
│   ├── src/                    # Core source modules
│   │   ├── training/           # Training components (13 modules)
│   │   ├── evaluation/         # Evaluation modules (11 modules)
│   │   ├── visualization/      # Plotting utilities (6 modules)
│   │   ├── utils/              # Helper functions (9 modules)
│   │   └── tests/              # Unit tests (5 modules)
│   │
│   └── scripts/                # Executable scripts
│       ├── training/           # Training scripts (run_full_pipeline.py)
│       ├── evaluation/         # Evaluation scripts
│       ├── analysis/           # Analysis & comparison (13 scripts)
│       ├── colorization/       # Colorization demos
│       └── visualization/      # Plot generation
│
├── models/                     # Pre-trained model weights (download from Google Drive)
│   ├── variant_097_random/     # Best teacher model (with checkpoints)
│   └── distilled_students/     # 6 compressed student models
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file

> **Note:** The `models/` folder is not included in the repository due to file size.
> Download from [Google Drive](https://drive.google.com/drive/folders/1ojJ07dq15GKh0d-W7HLnlD3LbqI48vgT?usp=sharing) and place in the project root.
```

---

## ⚡ Quick Start

### 1. Download Pre-trained Models

**Required Models** (~285MB total):

📥 **[Download from Google Drive](https://drive.google.com/drive/folders/1ojJ07dq15GKh0d-W7HLnlD3LbqI48vgT?usp=sharing)**

Download and extract to create the `models/` folder:
```
models/
├── variant_097_random/     # Best teacher model (~250MB)
│   ├── checkpoints/
│   │   └── final.pth
│   ├── config.json
│   └── training_history.json
└── distilled_students/     # 6 compressed students (~35MB)
    ├── student_lightweight_2x_best_model.pth
    ├── student_lightweight_4x_best_model.pth
    ├── student_lightweight_8x_best_model.pth
    ├── student_mobilenet_2x_best_model.pth
    ├── student_mobilenet_4x_best_model.pth
    └── student_mobilenet_8x_best_model.pth
```

**Additional Models** (optional, ~60GB):
- The Google Drive folder also contains 200+ variant models for comprehensive evaluation

### 2. Run Colorization

```bash
# Colorize images using top-performing models
python code/scripts/colorization/colorize_all_models_comprehensive.py --num_images 100
```

### 3. Evaluate Models

```bash
# Comprehensive evaluation on CIFAR-10
python code/scripts/evaluation/evaluate_all_models_comprehensive.py

# Find best models by each metric
python code/scripts/analysis/find_best_models_by_metric.py
```

### 4. Train Your Own Model

```bash
# Train a variant model
python code/src/training/variant_training_pipeline.py --variant_id 001 --init_mode random

# Train student models via knowledge distillation
python code/scripts/training/train_multiple_student_sizes.py

# Run the full training pipeline (200+ variants)
python Importantcode/run_full_pipeline.py
```

---

## 🤖 Models

### Teacher Models (Variants)

We trained **200+ variant models** with different configurations:

| Component | Options |
|-----------|---------|
| **Base Architecture** | ECCV16, SIGGRAPH17 |
| **Attention** | SE blocks, CBAM, ILA |
| **Normalization** | BatchNorm, GroupNorm, LayerNorm |
| **Activation** | ReLU, LeakyReLU, GELU, Swish |
| **Initialization** | Random, Pretrained |

### Student Models (Distilled)

Lightweight models created via knowledge distillation:

| Model | Parameters | Size Reduction | PSNR | SSIM |
|-------|------------|----------------|------|------|
| Lightweight-2x | 1.2M | 2× smaller | 26.8 | 0.921 |
| Lightweight-4x | 0.6M | 4× smaller | 26.1 | 0.908 |
| Lightweight-8x | 0.3M | 8× smaller | 25.2 | 0.891 |
| MobileNet-2x | 1.4M | 2× smaller | 27.1 | 0.928 |
| MobileNet-4x | 0.7M | 4× smaller | 26.4 | 0.915 |
| MobileNet-8x | 0.35M | 8× smaller | 25.5 | 0.897 |

---

## 📊 Evaluation Metrics

We evaluate models using comprehensive metrics:

### Pixel-Level Metrics
- **PSNR** (Peak Signal-to-Noise Ratio) - Higher is better
- **SSIM** (Structural Similarity Index) - Higher is better

### Perceptual Metrics
- **LPIPS** (Learned Perceptual Image Patch Similarity) - Lower is better
- **FID** (Fréchet Inception Distance) - Lower is better
- **KID** (Kernel Inception Distance) - Lower is better

### Color Metrics
- **ΔE2000** (Color difference in CIELAB space) - Lower is better
- **FSIM** (Feature Similarity Index) - Higher is better

---

## 📂 Datasets

### CIFAR-10 (Auto-download)
- Downloads automatically via torchvision
- 60,000 32×32 color images
- Used for primary training and evaluation

### ImageNet (Manual download)
- Download from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
- Set environment variable:
  ```bash
  # Linux/Mac
  export IMAGENET_VAL_ROOT=/path/to/ILSVRC
  
  # Windows
  set IMAGENET_VAL_ROOT=C:\path\to\ILSVRC
  ```

---

## 💻 Hardware Requirements

| Task | RAM | GPU VRAM | Storage |
|------|-----|----------|---------|
| Inference | 4GB | 4GB | 1GB |
| Training (single) | 8GB | 8GB | 10GB |
| Full Pipeline | 16GB | 8GB+ | 100GB |

---

## 🔧 Troubleshooting

<details>
<summary><b>ImportError: No module named 'colorizers'</b></summary>

Run scripts from the project root directory, or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code/colorization"
```
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

Reduce batch size in training scripts or use CPU:
```python
device = 'cpu'  # Instead of 'cuda'
```
</details>

<details>
<summary><b>Model weights not found</b></summary>

1. Ensure models are in the correct location (`models/` folder)
2. Download from [Google Drive](https://drive.google.com/drive/folders/1ojJ07dq15GKh0d-W7HLnlD3LbqI48vgT?usp=sharing)
</details>

<details>
<summary><b>LPIPS/FID not available</b></summary>

Install optional dependencies:
```bash
pip install lpips pytorch-fid
```
</details>

---

## 📚 References

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM TOG (SIGGRAPH)},
  year={2017}
}
```

---

## 📄 License

This project is for academic and educational purposes as part of the Advanced Machine Learning course.

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ for Advanced Machine Learning

</div>
