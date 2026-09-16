# Engineering review handoff · 2026-09-16

**Baseline:** `main` at `29883287896b898f033edc1e7e964fb857640265`. **Review branch:** `audit/reproducible-demo-and-claims-20260916`. [Draft PR #1](https://github.com/omarsaqr12/ImageColorization/pull/1). Neither main nor the original experiments/checkpoints were modified or removed.

## Calibration and file coverage

This is an ML research/course artifact, not a production inference service. The high-risk claims concern the **identity of the evaluated checkpoint**, dataset/split and metric provenance, comparisons of teacher versus students, and trained/evaluated counts. Their evidence needs executable metric code, accessible weights, fixed sample sets, evaluation logs and traced output files. A shape assertion or a colorful picture is not a quality benchmark.

The baseline recursive Git tree was enumerated. The README, both text documents under `docs/`, `requirements.txt`, the upstream ECCV16/SIGGRAPH17 definitions, shared image utilities, the existing demo, selected tests, and ILA module were read. This is **not an every-source-file audit**: the numerous training/evaluation/analysis scripts and variants were directory-inventoried but not all fully read. Dataset files, model checkpoints, and full experiment logs are not version-controlled; the external checkpoint folder was not retrieved. No reported CIFAR-10/ImageNet metric or checkpoint has been independently evaluated. This coverage limitation is material and must not be described as exhaustive.

## Claim-to-evidence boundaries

| Claim | Classification and evidence |
| --- | --- |
| Upstream ECCV16 and SIGGRAPH17 can instantiate and accept image tensors | **Tested:** [CPU CI run](https://github.com/omarsaqr12/ImageColorization/actions/runs/35085196971) imported both, ran synthetic 32×32 forward passes, and checked finite Lab-to-RGB output, with random weights. |
| New headless demo accepts grayscale/RGBA and produces clear errors for missing images | **Tested:** four offline CLI/input regression tests in the same CI run. |
| This repository's distilled models outperform the teacher and compress it 1467× | **Historical report only / unverified:** baseline README and historical technical notes quote these figures; no project-trained checkpoints, exact evaluation log or verified dataset split is tracked. |
| 200+ variants were fully trained and evaluated, best `variant_097` | **Unverified:** previous README and historical docs use different configuration/run descriptions; configuration count does not itself establish trained/evaluated model count. |
| Demo colorizes with upstream pretrained weights | **Partially supported:** pretrained model-loading code exists; actual external weight download, real-image inference and visual inspection were **not run**. |

## Changes and tests

New `scripts/colorize_image.py` exposes headless, image-file inference, explicit upstream-weight download consent, model/device selection and deterministic input validation. New `tests/test_colorize_image.py` and `.github/workflows/demo-smoke.yml` test CLI behavior and both original model forward paths; `requirements.txt` now declares IPython, which original model imports require. The README gives an actually CI-exercised CPU dependency subset, an architecture map, attribution, and an explicit distinction between this upstream demo and missing project-trained students. No historical artifacts or license were deleted.

**PASS:** Python 3.11.16 on Ubuntu 24.04; four offline regression tests; compilation of new script/tests; CPU-only PyTorch model imports, untrained forward passes and postprocessing for both backbones. The CI deliberately downloads **no pretrained weights**. **NOT RUN / BLOCKED:** original model checkpoint loading; full `requirements.txt` install; real pretrained-image demo; student inference; full training, ImageNet/CIFAR-10 evaluation, figure provenance; inspection of all historical source files; external model-asset license verification. No actual visual-quality claim follows from a synthetic untrained forward pass.

## Scope choices, recruiter description and next work

Accepted a small demo and smoke tests to make source code inspectable; rejected retraining dozens/hundreds of variants, new architecture features and fabricated benchmark improvements without datasets/weights/compute approval. Suggested repository description (GitHub metadata **not changed**): **“Image colorization research project exploring ECCV16/SIGGRAPH17 variants and knowledge distillation, with a headless demo and documented evaluation limits.”**

To finish the research audit, obtain and hash teacher/student checkpoints, configurations, split manifests, exact metric scripts and generated per-image outputs; trace metric direction/aggregation and reproduce table cells. Review every remaining readable tracked script and any accessible visual report, confirm contributor roles and third-party model/data rights, and test the pretrained CLI using an approved external download. Leave the PR draft and unmerged until those boundaries are acceptable.
