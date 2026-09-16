# Image colorization with model compression

An Advanced Machine Learning course project exploring **automatic grayscale-to-color conversion**, ECCV16/SIGGRAPH17 backbones, architecture variants, and knowledge distillation into smaller student networks. The repository contains model-building, training, evaluation, and analysis code. **Project-trained checkpoints and the underlying experiment outputs are not tracked in Git**, so the historical numerical results cannot be independently verified from this checkout alone.

The project extends earlier colorization research rather than introducing the ECCV16 or SIGGRAPH17 architectures. See [original work and attribution](#research-and-attribution) below.

## Start here

| If you want to... | Go to... |
| --- | --- |
| Inspect a small, headless working example | [`scripts/colorize_image.py`](scripts/colorize_image.py) |
| Understand the image preprocessing and original backbones | [`code/colorization/colorizers/`](code/colorization/colorizers/) |
| Inspect the variant and distillation implementations | [`code/src/training/`](code/src/training/) |
| Inspect evaluation code and its assumptions | [`code/src/evaluation/`](code/src/evaluation/) |
| Find the larger experimental entry points | [`code/scripts/training/`](code/scripts/training/), [`code/scripts/evaluation/`](code/scripts/evaluation/), [`code/scripts/analysis/`](code/scripts/analysis/) |
| Read the historical project write-up | [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md), [`docs/TECHNICAL_DETAILS.md`](docs/TECHNICAL_DETAILS.md) |

## Minimal demonstration: one image

The new, noninteractive command below runs **the original upstream pretrained ECCV16 or SIGGRAPH17 backbone**; it does **not** load this project's teacher or distilled-student checkpoints. The original backbone weights are hosted outside this repository and download on first use. The explicit flag makes that external download visible.

```bash
git clone https://github.com/omarsaqr12/ImageColorization.git
cd ImageColorization
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

# CPU-only PyTorch and the small demo dependency set exercised by CI.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pillow matplotlib scikit-image 'ipython>=8.0'

# Supply your own local image. A network connection is required on first run
# to fetch original upstream pretrained model weights.
python scripts/colorize_image.py my_photo.jpg --output outputs/colorized.png --download-pretrained

# The other upstream backbone, still on CPU:
python scripts/colorize_image.py my_photo.jpg --output outputs/siggraph.png \
  --model siggraph17 --device cpu --download-pretrained

# Offline checks for the new interface; no pretrained weights are downloaded.
python -m unittest discover -s tests -v
```

The CPU dependency installation, model import, untrained-model forward/postprocessing and offline CLI tests were exercised by [GitHub Actions](https://github.com/omarsaqr12/ImageColorization/actions/runs/35085196971). **The pretrained weight download and the complete real-image command have not been exercised by CI**; those steps depend on external upstream availability and weight terms. To work on the wider training/evaluation pipeline, install [`requirements.txt`](requirements.txt) separately and verify the needed datasets/checkpoints first; the entire requirements set and research pipeline have not been validated as a fresh install.

The demo normalizes grayscale/RGBA inputs to RGB, performs the original colorizer preprocessing, saves the prediction, and does not require a graphical display. It is a **backbone demonstration**, not evidence for any of the historical student-versus-teacher metrics. If weights cannot be downloaded, inspect the architecture or use the original project-specific checkpoints as documented below; no hidden fallback to random model weights is performed by this CLI.

## Research pipeline and evidence boundaries

The source tree includes generators, optional attention/semantic modules, variants, GAN/perceptual/distillation losses, evaluation scripts, and training entry points. The historical project documentation describes a search over valid architecture configurations and student models with different width settings. The distinction between **configurations represented in code**, **models actually trained**, and **models evaluated on a particular split** matters; these are not interchangeable counts.

The [earlier README at the recorded baseline](https://github.com/omarsaqr12/ImageColorization/blob/29883287896b898f033edc1e7e964fb857640265/README.md) reports 23.04 dB teacher PSNR versus 24.68 dB for a small student and a 1467× checkpoint-size ratio, among other results. Those are **historical project-reported measurements, not results re-run or independently audited in this branch**. The corresponding project-trained checkpoints and the complete dataset/split and metric provenance are required before claiming that the students outperform their teacher, that a specific variant is best, or that results generalize to ImageNet. The separately hosted checkpoint folder is linked in the historical README; external access and file contents have not been verified here. The two older documents linked above retain additional historical numerical and comparative claims; treat those as unverified pending an artifact-backed audit.

To reproduce the original research rather than the backbone demo, first review the [technical notes](docs/TECHNICAL_DETAILS.md), obtain the project-trained weights and precisely identified datasets/splits, verify checkpoint/config mapping, and trace the appropriate evaluation entry point under [`code/scripts/evaluation/`](code/scripts/evaluation/). Some scripts require CIFAR-10 or a separately supplied ImageNet tree. **Do not launch the all-variants training pipeline as a quickstart**: it represents a substantial multi-run workload. Numerical comparisons should only be published with the exact checkpoints, preprocessing, sample counts, metric definitions, hardware/software versions, and evaluation logs.

## Scope and limitations

- The upstream demonstration may download third-party pretrained weights. The project's distilled-student weights are not bundled and are **not** used by the demo.
- The original research pipeline has many related scripts and historical alternatives. This branch does not assert that every combination is executable, that checkpoint links work, or that the full pipeline has been reproduced from a fresh installation.
- Tests that check tensor shapes or print model-loading exceptions do not by themselves verify colorization quality or evaluation claims. New tests cover demo input handling, CLI behavior, and forward/Lab-to-RGB postprocessing on synthetic inputs with random weights.
- Dataset usage and third-party pretrained-weight terms may differ from the repository's code license. Do not assume the MIT code license grants rights to redistribute external datasets or original weights.

## Research and attribution

The base colorization networks follow work by Richard Zhang and collaborators: [*Colorful Image Colorization* (ECCV 2016)](https://arxiv.org/abs/1603.08511) and [*Real-Time User-Guided Image Colorization with Learned Deep Priors* (SIGGRAPH 2017)](https://arxiv.org/abs/1705.02999). The `code/colorization/colorizers/` implementation contains pretrained-model URLs and should be evaluated with the original authors' license and attribution in mind. This repository is described in its prior documentation as an Advanced Machine Learning final project; no verified individual-contribution breakdown is available in the inspected materials. See [LICENSE](LICENSE) for the repository's code license.
