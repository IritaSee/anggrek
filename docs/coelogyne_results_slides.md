---
marp: true
title: Coelogyne Classification Results
paginate: true
---

# Coelogyne Classification Results

Technical and non-technical review of current experiments

Date: 2026-07-13

---

# Problem and Objective

- Build an image classifier for 5 Coelogyne orchid species.
- Reduce manual identification effort and inconsistency.
- Evaluate multiple model families and identify the most deployable option.

Classes:
- Asperata, Dayana, Pandurata, Rumphii, Swaniana

---

# Data and Workflow

- Data organized with directory-per-class format.
- EDA used unsupervised analysis:
  - variance map
  - gradient map
  - PCA
  - K-Means patch clustering
  - t-SNE visualization
- Training/evaluation pipeline includes transfer learning, fine-tuning, and export.

---

# Models Evaluated

- MobileNetV2
  - Head training
  - Fine-tuning
- EfficientNetV2S
  - Head training
  - Fine-tuning
- EfficientNetV2S dedicated experiment
  - Dropout 0.3
  - Fine-tune top 15 layers
  - Label smoothing 0.1
  - Stagnation guard for optional oversampling round
- CSPDarknet53 (from scratch)
- ViT-Tiny baseline

---

# Headline Results

- Best overall model stage: EfficientNetV2S Fine-tune
  - Best validation accuracy: 0.8158
- EfficientNetV2S Head is also strong:
  - Best validation accuracy: 0.8026
- MobileNetV2 is moderate:
  - Best validation accuracy around 0.69
- CSPDarknet53 and ViT-Tiny are currently not competitive.

---

# Consolidated Metrics (Validation)

| Model Stage | Best Acc | Best Loss |
|---|---:|---:|
| EfficientNetV2S Fine-tune | 0.8158 | 0.7566 |
| EfficientNetV2S Head | 0.8026 | 0.6602 |
| EffNet Dedicated Exp R1 Fine-tune | 0.7895 | 1.1064 |
| EffNet Dedicated Exp R1 Head | 0.7500 | 0.9605 |
| MobileNetV2 Head | 0.6974 | 1.3217 |
| MobileNetV2 Fine-tune | 0.6842 | 1.9504 |
| CSPDarknet53 | 0.4868 | 11.1342 |
| ViT-Tiny | 0.2763 | 1.9591 |

---

# Technical Interpretation

- Transfer learning CNNs outperform other tested strategies.
- EfficientNetV2S gives the best feature quality for this dataset.
- MobileNetV2 remains a reasonable lightweight fallback.
- CSPDarknet53 shows optimization/generalization problems in current setup.
- ViT-Tiny likely underfits due to data scale and training regime.

---

# Non-Technical Interpretation

- For real use, EfficientNetV2S is currently the safest choice.
- Users can expect better identification reliability vs other tested models.
- MobileNetV2 can be used on constrained devices with lower accuracy expectations.
- Current evidence supports focusing team effort on one strong model path.

---

# Deployment Status

- Saved model artifacts are available, including TFLite variants.
- Practical recommendation:
  - Primary: EfficientNetV2S fine-tune
  - Fallback: MobileNetV2
- Next milestone: validate with external real-world photo samples.

---

# Risks and Open Gaps

- Results are based on logged validation runs, not independent field validation.
- Multiple run files per model indicate potential variance not yet summarized as mean/std.
- Class-level confusion and error modes are not consolidated in one decision report.

---

# Recommended Next Steps

1. Lock one benchmark split and run repeated seeds.
2. Publish mean/std and per-class precision/recall/F1.
3. Add confidence calibration for production decisions.
4. Finalize reproducible training metadata for chosen checkpoint.
5. Pilot on external images from varied conditions.

---

# Closing

EfficientNetV2S is the strongest current direction for Coelogyne classification in this repository, with clear room for reliability gains through standardized evaluation and external validation.