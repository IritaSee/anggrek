# Coelogyne Classification Results - Presentation Points

Date: 2026-07-13

## 1) Executive Summary (Non-Technical)

- Goal: classify 5 Coelogyne orchid species from photos for faster and more consistent identification.
- Best performing model family is EfficientNetV2S, with the strongest validation accuracy around 81.58% from fine-tuning logs.
- MobileNetV2 is competitive but lower (best around 69.74% in head training logs and 68.42% in fine-tuning logs).
- CSPDarknet53 and ViT-Tiny underperform on the current setup and dataset (below 50% and below 30% respectively in best validation accuracy).
- Practical takeaway: EfficientNetV2S is currently the most reliable candidate for deployment, with TFLite artifacts already available for mobile usage.

## 2) Project Scope and Data

- Task: 5-class image classification.
- Classes: Asperata, Dayana, Pandurata, Rumphii, Swaniana.
- Data layout follows directory-per-class format under data/.
- EDA notebook used unsupervised methods (variance map, gradient map, PCA, K-Means, t-SNE) to identify informative visual regions.

## 3) Models and Training Strategies (Technical)

- MobileNetV2:
  - Transfer learning with two stages: head-only training, then fine-tuning upper layers.
- EfficientNetV2S (main pipeline):
  - Head training and fine-tuning experiments in TensorBoard logs.
- EfficientNetV2S dedicated experiment script:
  - Dropout 0.3, fine-tune top 15 layers, label smoothing 0.1.
  - Stagnation guard for optional second round with class-weighted oversampling and stronger augmentation.
  - In the available run, oversampling was not triggered (oversampling=false in notes).
- CSPDarknet53:
  - Custom architecture path (from scratch) logged in train/validation event files.
- ViT-Tiny:
  - Transformer baseline logged in train/validation event files.

## 4) Consolidated Results (From Current Logs)

Notes:
- Values below are best validation metrics extracted from existing event logs and experiment CSV/JSON artifacts.
- For each model-stage, the best run among available event files was selected.

| Model Stage | Best Validation Accuracy | Best Validation Loss | Observation |
|---|---:|---:|---|
| EfficientNetV2S Fine-tune | 0.8158 | 0.7566 | Best overall in current tracked logs |
| EfficientNetV2S Head | 0.8026 | 0.6602 | Strong baseline before full fine-tune |
| EfficientNetV2S Dedicated Exp (Round 1 Fine-tune CSV) | 0.7895 | 1.1064 | Stable but below main EfficientNetV2S fine-tune peak |
| EfficientNetV2S Dedicated Exp (Round 1 Head CSV) | 0.7500 | 0.9605 | Good early-stage transfer performance |
| MobileNetV2 Head | 0.6974 | 1.3217 | Solid lightweight baseline |
| MobileNetV2 Fine-tune | 0.6842 | 1.9504 | Fine-tuning did not improve in best logged run |
| CSPDarknet53 | 0.4868 | 11.1342 | Significant optimization/generalization issues |
| ViT-Tiny | 0.2763 | 1.9591 | Not suitable with current data scale/setup |

## 5) Interpretation (Technical)

- Transfer learning with CNN backbones clearly outperforms from-scratch and transformer baseline in this project.
- EfficientNetV2S shows the strongest representation quality for this dataset and preprocessing pipeline.
- MobileNetV2 remains useful when model size and inference speed are prioritized, but with accuracy trade-off.
- CSPDarknet53 very high loss indicates training instability or mismatch between architecture/training recipe and dataset scale.
- ViT-Tiny likely data-hungry for this task; current dataset size/augmentation appears insufficient.

## 6) Interpretation (Non-Technical)

- If this system is used by growers, collectors, or researchers, EfficientNetV2S should give the most dependable identification among tested options.
- Lightweight option (MobileNetV2) can still be considered for lower-end devices, but users should expect lower hit rate.
- Current evidence suggests investment should focus on improving the EfficientNetV2S path rather than extending CSPDarknet53/ViT-Tiny right now.

## 7) Deployment Readiness

- Saved model artifacts already exist for major model families, including TFLite files for mobile deployment.
- Recommended deployment candidate today: EfficientNetV2S fine-tune model.
- Recommended fallback candidate: MobileNetV2 (if latency/size constraints are strict).

## 8) Risks and Gaps

- Current summary is based on logged validation metrics only; no independent hold-out or field-test set is documented here.
- Multiple event files exist per model; best-run reporting can hide run-to-run variance.
- Class-level precision/recall and confusion matrix outcomes are not consolidated in this summary and should be added before production decisions.

## 9) Recommended Next Actions

1. Standardize one benchmark split and report mean/std across repeated runs.
2. Add per-class metrics and confusion matrix summary in a single report artifact.
3. Run calibration and threshold analysis for real-world confidence handling.
4. Export final chosen EfficientNetV2S checkpoint with reproducible training metadata.
5. Validate on external photos (different lighting/background/device) before production release.

## 10) One-Line Closing

- Current results indicate EfficientNetV2S is the strongest and most deployment-ready direction for Coelogyne species classification in this repository.