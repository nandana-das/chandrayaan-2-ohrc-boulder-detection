# Domain-Adaptive Transfer Learning for Rockfall-Related Feature Detection in Chandrayaan-2 OHRC Imagery

### M.Tech AI & Data Science Research Project — Alliance University, Bengaluru

---

## Overview

This repository contains the implementation and experimental results for a domain-adaptive transfer learning study on high-resolution Chandrayaan-2 Orbiter High Resolution Camera (OHRC) imagery.

The objective is to transfer a detector trained on labeled lunar boulder datasets to **unlabeled OHRC imagery**, where target-domain ground-truth annotations are not currently available. The study therefore separates two evaluation settings:

1. **Held-out source-domain evaluation**, where standard object-detection metrics can be computed against ground-truth annotations.
2. **Target-domain OHRC inference**, where results are reported as candidate detections, positive-tile rates, and confidence statistics rather than target-domain accuracy.

The current controlled experiment evaluates five YOLO-family detectors:

- YOLO26n
- YOLO26s
- YOLO26m
- YOLOv8n
- YOLOv5s

The main domain-adaptation step uses **averaged-reference histogram matching**, where a single reference image is constructed by averaging 20 randomly selected OHRC tiles. The adapted models are then fine-tuned with AdamW and cosine learning-rate decay.

---

## Current Experimental Configuration

### Source-domain data

The labeled source domain combines:

- **Prieur et al. lunar boulder dataset**
- **RMaM-2020**

The controlled split contains:

| Split | Images | Boulder instances |
|---|---:|---:|
| Train | 4,379 | 122,537 |
| Validation | 697 | 24,303 |
| Held-out test | 262 | 7,268 |
| **Total** | **5,338** | **154,108** |

The held-out test set is kept separate from training and validation and is used for the final source-domain comparison.

### OHRC target domain

The primary OHRC target set contains:

- **31,769 usable 640 × 640 tiles**
- calibrated Chandrayaan-2 OHRC products
- no target-domain ground-truth annotations

An additional ultra-deep polar set contains:

- **13,906 usable tiles**
- **20 calibrated OHRC products**
- extreme south-polar coverage

Because the OHRC target images are unlabeled, detections on these images are treated as **candidate detections**, not verified boulders or rockfalls.

---

## Methodology

The current controlled pipeline consists of two training stages followed by target-domain inference.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Source-Domain Transfer Learning                            │
│                                                                     │
│ Prieur + RMaM labeled source data                                  │
│ 4,379 train / 697 validation / 262 held-out test images            │
│                                                                     │
│ YOLO26n | YOLO26s | YOLO26m | YOLOv8n | YOLOv5s                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2: Domain Adaptation                                         │
│                                                                     │
│ 20 randomly selected OHRC tiles                                    │
│            ↓                                                        │
│ 20-tile averaged reference image                                  │
│            ↓                                                        │
│ Histogram matching of source-domain images                         │
│            ↓                                                        │
│ Full fine-tuning                                                   │
│ AdamW | lr₀ = 1×10⁻⁴ | cosine LR | 50 epochs                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3: Unlabeled OHRC Inference                                  │
│                                                                     │
│ 31,769 primary OHRC tiles + 13,906 ultra-deep polar tiles          │
│                                                                     │
│ Candidate detections | positive-tile rate | confidence statistics  │
└─────────────────────────────────────────────────────────────────────┘
```

### OHRC preprocessing

Calibrated OHRC PDS4 products are converted into 640 × 640 non-overlapping tiles. Edge tiles are zero-padded where necessary. Very dark tiles are removed using a mean-intensity threshold.

For visualization only, percentile normalization can be applied to generate human-readable previews. This visualization normalization is not used as the model input transformation.

---

## Held-Out Source-Domain Results

The final five-model benchmark is evaluated on the untouched 262-image source-domain test split.

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| **YOLO26m** | **0.6380** | **0.6172** | **0.6275** | **0.6463** | **0.2622** |
| YOLO26s | 0.6260 | 0.5691 | 0.5962 | 0.6023 | 0.2308 |
| YOLOv5s | 0.6108 | 0.5499 | 0.5788 | 0.5737 | 0.2121 |
| YOLO26n | 0.5873 | 0.5069 | 0.5442 | 0.5292 | 0.1896 |
| YOLOv8n | 0.5469 | 0.4798 | 0.5111 | 0.4813 | 0.1671 |

**YOLO26m is the strongest model among the five evaluated detectors on the held-out source-domain test set.**

The YOLO26 scaling experiment also shows a consistent source-domain improvement as model capacity increases from YOLO26n to YOLO26s to YOLO26m.

---

## Target-Domain OHRC Inference

Inference on the primary 31,769-tile OHRC set uses a confidence threshold of:

[
	au = 0.20
]

Since these images have no ground-truth annotations, the table below describes model behavior rather than target-domain accuracy.

| Model | Candidate detections | Positive tiles | Positive-tile rate | Mean confidence |
|---|---:|---:|---:|---:|
| YOLO26m | 1,484 | 443 | **1.39%** | 0.293 |
| YOLO26s | 14,102 | 2,050 | 6.45% | 0.295 |
| YOLO26n | 5,084 | 2,134 | 6.72% | 0.254 |
| YOLOv5s | 24,452 | 3,896 | 12.26% | 0.330 |
| YOLOv8n | 353,427 | 14,129 | 44.47% | 0.304 |

The large differences in candidate-detection density show that the models respond differently to the unlabeled OHRC target distribution. In particular, YOLOv8n produces substantially denser candidate detections than the other controlled Stage-2 models.

These rates should **not** be interpreted as precision, recall, false-positive rate, or target-domain accuracy.

---

## Regional Candidate-Detection Analysis

The 31,769 primary OHRC tiles are divided into two geographic regimes:

- South Pole: 16,308 tiles
- Equatorial/Northern: 15,461 tiles

At `τ = 0.20`, the current regional results are:

| Region | Model | Positive-tile rate |
|---|---|---:|
| South Pole | YOLO26m | 0.31% |
| South Pole | YOLO26s | 6.70% |
| South Pole | YOLO26n | 7.21% |
| South Pole | YOLOv5s | 13.70% |
| South Pole | YOLOv8n | 77.81% |
| Equatorial/Northern | YOLO26m | 2.54% |
| Equatorial/Northern | YOLO26s | 6.19% |
| Equatorial/Northern | YOLO26n | 6.20% |
| Equatorial/Northern | YOLOv5s | 10.75% |
| Equatorial/Northern | YOLOv8n | 9.31% |

Regional differences should be interpreted cautiously. They may be influenced by solar incidence angle, illumination, terrain morphology, and the distribution of available OHRC products. Without target-domain annotations, these differences do not establish intrinsic geological variation.

---

## Ultra-Deep Polar Evaluation

The repository also contains inference results for 20 calibrated ultra-deep polar products comprising 13,906 usable tiles.

| Model | Candidate detections | Positive tiles | Positive-tile rate | Mean confidence |
|---|---:|---:|---:|---:|
| YOLO26m | 1,447 | 309 | 2.22% | 0.3057 |
| YOLO26s | 8,128 | 2,124 | 15.27% | 0.2699 |
| YOLO26n | 10,085 | 1,173 | 8.44% | 0.2907 |
| YOLOv5s | 7,202 | 1,779 | 12.79% | 0.3214 |
| YOLOv8n | 71,001 | 6,047 | 43.48% | 0.2959 |

An earlier RT-DETR-L experiment is retained in the repository as a **historical baseline**, but it was trained under a different experimental protocol and is therefore not treated as a controlled comparison with the five current Stage-2 models.

---

## Model Complexity and Inference Latency

All complexity measurements use 640 × 640 input resolution.

| Model | Parameters (M) | GFLOPs | Model size (MB) |
|---|---:|---:|---:|
| YOLO26n | 2.504 | 2.89 | 5.14 |
| YOLO26s | 9.949 | 11.25 | 19.37 |
| YOLO26m | 21.774 | 37.36 | 41.98 |
| YOLOv8n | 3.011 | 4.10 | 5.96 |
| YOLOv5s | 9.123 | 12.02 | 17.66 |

Inference latency was measured on an NVIDIA GeForce RTX 3050 Laptop GPU at batch size 1 using 100 fixed OHRC tiles.

| Model | Mean latency (ms) | FPS |
|---|---:|---:|
| YOLO26n | 70.26 | 14.2 |
| YOLO26s | 54.52 | 18.3 |
| YOLO26m | 59.96 | 16.7 |
| YOLOv8n | 50.74 | 19.7 |
| YOLOv5s | 45.19 | 22.1 |

These measurements exclude disk I/O, image decoding, and large-scale tile stitching.

---

## Important Interpretation Constraint

The primary OHRC target set and ultra-deep polar set are **unlabeled**.

Therefore, the following claims are intentionally not made for target-domain inference:

- target-domain accuracy
- target-domain precision
- target-domain recall
- target-domain F1-score
- target-domain mAP
- verified false-positive counts
- verified true-positive counts

A detection on an unlabeled OHRC tile is referred to as a **candidate detection** or **candidate rockfall-related feature**.

The target-domain experiments are intended to characterize model behavior under domain shift and provide a baseline for future work using expert-annotated OHRC imagery.

---

## Repository Structure

```
chandrayaan-2-ohrc-boulder-detection/
├── scripts/
│   ├── paper_generation/
│   ├── ablation/
│   ├── evaluate_test_split.py
│   ├── infer_ohrc.py
│   ├── process_polar_batch.py
│   ├── analyze_regional_detections.py
│   └── ...
├── results/
│   ├── paper_figures/
│   ├── paper_tables/
│   ├── confidence_analysis/
│   ├── ohrc_inference/
│   └── polar_inference/
├── runs/
│   ├── stage2_yolo26n_combined_hm_cosine/
│   ├── stage2_yolo26s_combined_hm_cosine/
│   ├── stage2_yolo26m_combined_hm_cosine/
│   ├── stage2_yolov8n_combined_hm_cosine/
│   └── stage2_yolov5s_combined_hm_cosine/
├── data/
├── requirements.txt
└── README.md
```

The `results/paper_tables/` directory contains the consolidated tables used for the manuscript, while `results/paper_figures/` contains publication figures in PNG/PDF formats.

---

## Reproduction

### Installation

```bash
git clone https://github.com/nandana-das/chandrayaan-2-ohrc-boulder-detection.git
cd chandrayaan-2-ohrc-boulder-detection
pip install -r requirements.txt
```

### Main workflow

The exact scripts and configurations used for the current experiments are retained in the repository. The broad workflow is:

```text
Source datasets
      ↓
Label conversion and controlled split
      ↓
OHRC preprocessing and 640×640 tiling
      ↓
20-tile averaged-reference histogram matching
      ↓
Stage-2 cosine fine-tuning
      ↓
Held-out source-domain evaluation
      ↓
Primary OHRC inference
      ↓
Regional analysis
      ↓
Ultra-deep polar inference
      ↓
Publication figures and tables
```

See the individual scripts and the consolidated reports under `results/paper_tables/` for experiment-specific commands and outputs.

---

## Publication Figures and Tables

The repository contains the current manuscript figures, including:

- training curves
- held-out source-domain PR curves
- OHRC confidence distributions
- qualitative OHRC candidate detections
- cross-model candidate behavior
- regional candidate-detection rates
- ultra-deep polar results
- object-size distributions
- YOLO26 scaling metrics
- YOLO26 scaling complexity
- HM/no-HM ablation figures

The corresponding structured data are stored under `results/paper_tables/`.

---

## Data Sources

### Chandrayaan-2 OHRC

Calibrated OHRC products are obtained through ISRO's ISSDC PRADAN data portal.

The raw mission data are **not included in this repository**.

### Source-domain datasets

The source domain uses:

- Prieur et al. lunar boulder annotations
- RMaM-2020 labeled lunar imagery

Users must obtain the datasets from their respective sources and follow their licensing and attribution requirements.

---

## Hardware

The controlled experiments were performed on:

- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB VRAM
- CUDA-enabled PyTorch environment
- Windows development environment

The YOLO26m training run used batch size 4 because of GPU memory constraints; the smaller YOLO models used batch size 8 under the controlled Stage-2 configuration.

---

## Current Research Status

The repository represents the current controlled five-model Stage-2 experiment.

The main conclusions supported by the available evidence are:

1. Increasing YOLO26 model capacity improves performance on the labeled held-out source-domain test set.
2. YOLO26m is the strongest model among the five evaluated detectors on that held-out source-domain benchmark.
3. The five models exhibit substantially different candidate-detection densities when transferred to unlabeled OHRC imagery.
4. Target-domain candidate rates should not be interpreted as verified detection accuracy without OHRC ground-truth annotations.
5. Geographic differences in candidate rates require cautious interpretation because image acquisition and illumination conditions vary across products.
6. Expert-annotated OHRC data are needed for definitive target-domain evaluation.

---

## Citation

If you use this work, please cite:

```bibtex
@article{das2026ohrc,
  title   = {Domain-Adaptive Transfer Learning for Rockfall-Related
             Feature Detection in Chandrayaan-2 OHRC Imagery},
  author  = {Das, Nandana Narayan and Karpagalakshmi, R. C.},
  year    = {2026}
}
```

## Acknowledgement

The authors acknowledge the Indian Space Research Organisation (ISRO) and the Indian Space Science Data Centre (ISSDC) for access to Chandrayaan-2 OHRC data through the PRADAN portal.
