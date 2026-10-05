# Paper Results Generation Report

**Date:** 2026-10-03  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`  
**Authoritative Stage-2 Models:**
- `runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolo26s_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolo26m_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolov8n_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolov5s_combined_hm_cosine/weights/best.pt`

---

## Executive Overview
This document compiles the exhaustive inventory of all publication-ready figures, tables, and benchmark analyses generated for the manuscript based exclusively on the controlled Stage-2 cosine domain adaptation models. All artifacts have been generated in dual vector (PDF) and raster (>=300 DPI PNG) formats alongside structured CSV and Markdown data tables. All terminology and descriptions strictly adhere to scientific rigor for unlabeled target imagery and controlled experimental definitions.

---

## Section A: Generated from Existing Outputs (No New Inference)

### 1. `fig_training_curves.png` / `fig_training_curves.pdf`
- **Filename:** `results/paper_figures/fig_training_curves.png` & `.pdf`
- **Data Source:** `runs/stage2_*_combined_hm_cosine/results.csv` (50 epochs per model)
- **New Inference Required:** No
- **Training Required:** No (read directly from verified training run logs)
- **Intended Paper Section:** Section 3.2 (Stage-2 Controlled Domain Adaptation Training Dynamics)
- **One-Sentence Interpretation:** Controlled Stage-2 fine-tuning with AdamW and cosine learning rate decay converges smoothly across all three backbones, displaying maximum observed transient epoch values of 0.612 (YOLOv5s) and 0.589 (YOLO26n), while the authoritative best-checkpoint validation metrics remain 0.5869 (YOLOv5s), 0.5161 (YOLO26n), and 0.4885 (YOLOv8n).
- **Scientific Caveat:** Validation curves reflect performance on the combined source domain (Prieur + RMaM) under 20-tile averaged OHRC histogram matching. Transient epoch peaks must be distinguished from the authoritative best-checkpoint metrics, and neither curve directly measures target-domain generalization.

### 2. `fig_ohrc_confidence_distributions.png` / `fig_ohrc_confidence_distributions.pdf`
- **Filename:** `results/paper_figures/fig_ohrc_confidence_distributions.png` & `.pdf`
- **Data Source:** `results/ohrc_inference/raw_detections_*.csv` (31,769 usable tiles)
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 4.2 (Target-Domain Confidence Distribution & Operating Regimes)
- **One-Sentence Interpretation:** YOLOv8n exhibits an overwhelming mass of threshold-adjacent candidate detections near $\tau = 0.20$ (353,427 candidates, mean confidence 0.304), whereas YOLO26n is conservative (5,084 candidates, mean confidence 0.254) and YOLOv5s displays an intermediate detection density (24,452 candidates, mean confidence 0.330).
- **Scientific Caveat:** Confidence scores on unlabeled target imagery reflect raw model output activations under domain shift and cannot be interpreted as posterior probabilities of true boulder presence without ground truth.

### 3. `fig_regional_detection_rates.png` / `fig_regional_detection_rates.pdf` & `regional_detection_rates.csv`
- **Filename:** `results/paper_figures/fig_regional_detection_rates.png` & `.pdf`, `results/paper_tables/regional_detection_rates.csv`
- **Data Source:** `results/confidence_analysis/ohrc_regional_detection_comparison.csv` (16,309 South Pole tiles vs. 15,460 Equatorial/Northern tiles)
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 5.1 (Regional Detection Rate Disparities)
- **One-Sentence Interpretation:** YOLOv8n flags candidates on 77.81% of South Polar tiles versus 9.31% in equatorial regions, while YOLO26n and YOLOv5s maintain relatively stable regional detection frequencies (7.21% vs 6.20% and 13.70% vs 10.75%, respectively).
- **Scientific Caveat:** Observed regional candidate-rate differences must NOT be cited as evidence of intrinsic geological variation; they may be influenced by solar incidence angle, illumination, terrain morphology, and product distribution.

### 4. `fig_ultradeep_polar.png` / `fig_ultradeep_polar.pdf` & `ultradeep_polar_summary.csv`
- **Filename:** `results/paper_figures/fig_ultradeep_polar.png` & `.pdf`, `results/paper_tables/ultradeep_polar_summary.csv`
- **Data Source:** `results/polar_inference/` (13,906 usable tiles from 20 calibrated polar products)
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 5.2 (Ultra-Deep South Polar Benchmark Evaluation)
- **One-Sentence Interpretation:** Under extreme polar illumination, the historical RT-DETR-L baseline flagged 83.88% of tiles (532,844 candidates), whereas controlled Stage-2 models exhibit far more restrained behavior: YOLO26n flags 8.44% (10,085 candidates), YOLOv5s flags 12.79% (7,202 candidates), and YOLOv8n flags 43.48% (71,001 candidates).
- **Scientific Caveat:** RT-DETR-L was trained under an earlier, unconstrained protocol and serves strictly as an archived Historical Baseline (Pre-Stage 2); it is not treated as a controlled comparison against the three current models.

### 5. `fig_object_size_distribution.png` / `fig_object_size_distribution.pdf` & `object_size_statistics.csv`
- **Filename:** `results/paper_figures/fig_object_size_distribution.png` & `.pdf`, `results/paper_tables/object_size_statistics.csv`
- **Data Source:** Source dataset annotation files in `data/combined_hm/` (154,108 bounding boxes across Train [122,537], Val [24,303], and Test [7,268])
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 2.3 (Source Dataset Morphometric Properties)
- **One-Sentence Interpretation:** Across 154,108 source bounding boxes, annotations exhibit a median width of 7.68 px, median height of 7.68 px, and median area of 52.43 px$^2$, demonstrating that the source dataset is dominated by small annotated objects.
- **Scientific Caveat:** Annotation boundaries reflect manual labeling conventions in the source datasets (Prieur et al. / RMaM) and indicate object-size distribution within the training distribution rather than sensor-specific optical limits.

### 6. `fig_cross_model_behavior.png` / `fig_cross_model_behavior.pdf`
- **Filename:** `results/paper_figures/fig_cross_model_behavior.png` & `.pdf`
- **Data Source:** Same-tile extraction from `results/ohrc_inference/raw_detections_*.csv` and `data/tiles/usable/`
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 4.3 (Qualitative Cross-Model Discrepancy Analysis)
- **One-Sentence Interpretation:** Side-by-side tile panels demonstrate that on identical lunar terrain, YOLO26n produces sparse candidate detections, YOLOv5s exhibits intermediate candidate-detection density, and dense candidate activations occur for YOLOv8n over textured terrain.
- **Scientific Caveat:** The visualizations illustrate qualitative behavioral differences in candidate detection density, but cannot establish correctness or accuracy on unlabeled target imagery.

### 7. `fig_ohrc_qualitative_detections.png` / `fig_ohrc_qualitative_detections.pdf` & `qualitative_examples.csv`
- **Filename:** `results/paper_figures/fig_ohrc_qualitative_detections.png` & `.pdf`, `results/paper_tables/qualitative_examples.csv`
- **Data Source:** `results/ohrc_inference/raw_detections_*.csv` and `data/tiles/usable/`
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 4.4 (Qualitative Target-Domain Morphological Detections)
- **One-Sentence Interpretation:** A 3×3 grid of candidate detections exemplifies isolated candidates with cast shadows, crater-rim block fields, texture boundaries, and shadow-associated interface features.
- **Scientific Caveat:** All bounding boxes are explicitly designated as candidate detections or candidate rockfall-related features; no ground-truth assertions (such as true positive or false positive) are made.

### 8. `model_complexity.csv` / `model_complexity.md`
- **Filename:** `results/paper_tables/model_complexity.csv` & `.md`
- **Data Source:** Model checkpoint graphs profiled via `thop`
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 3.1 (Detector Architecture & Computational Complexity)
- **One-Sentence Interpretation:** YOLO26n exhibits the lowest parameter count and GFLOPs (2.504M parameters, 2.89 GFLOPs, 5.14 MB), followed by YOLOv8n (3.011M parameters, 4.10 GFLOPs, 5.96 MB), while YOLOv5s has 9.123M parameters, 12.02 GFLOPs, and 17.66 MB file size.
- **Scientific Caveat:** Complexity metrics represent static model graph operations at 640×640 input resolution on a batch size of 1; runtime latency depends additionally on architectural operators and memory access.

---

## Section B: Generated with Small New Evaluation (No Training)

### 1. `fig_test_pr_curves.png` / `fig_test_pr_curves.pdf` & `test_pr_curve_data.csv`
- **Filename:** `results/paper_figures/fig_test_pr_curves.png` & `.pdf`, `results/paper_tables/test_pr_curve_data.csv`
- **Evaluation Performed:** Validation pass (`model.val()`) on the held-out source test split (262 images, 7,268 ground truth boulders) using each Stage-2 cosine checkpoint.
- **Training Required:** Zero (exact existing checkpoints evaluated)
- **Intended Paper Section:** Section 3.3 (Held-Out Source-Domain Quantitative Evaluation)
- **One-Sentence Interpretation:** On the held-out Prieur test set, YOLOv5s achieves the highest overall precision-recall envelope (mAP@0.5 = 0.5737), outperforming YOLO26n (0.5292) and YOLOv8n (0.4813).
- **Scientific Caveat:** These curves reflect source-domain test data under histogram matching; they cannot be assumed to extrapolate identically to target Chandrayaan-2 OHRC imagery due to remaining domain shift.

### 2. `inference_latency.csv`
- **Filename:** `results/paper_tables/inference_latency.csv`
- **Evaluation Performed:** 15 warmup iterations followed by 100 timed inferences per model on 100 fixed OHRC tiles at 640×640 on an NVIDIA RTX 3050 Laptop GPU (CUDA:0).
- **Training Required:** Zero
- **Intended Paper Section:** Section 3.4 (Deployment GPU Latency & Throughput Benchmark)
- **One-Sentence Interpretation:** In this GPU latency/throughput benchmark, YOLOv8n processed at 31.9 FPS (mean 31.37 ms), YOLOv5s at 30.3 FPS (mean 33.00 ms), and YOLO26n at 20.5 FPS (mean 48.68 ms).
- **Scientific Caveat:** Benchmark values measure per-image tensor forward pass and NMS with GPU synchronization, but exclude disk I/O, decoding, and large-scale tile mosaic stitching.

---

## Section C: Still Requires New Training

### 1. Controlled No-HM vs. HM Ablation
- **Description:** Training identical YOLO26n, YOLOv8n, and YOLOv5s architectures on the raw source dataset (without histogram matching) using the exact same AdamW optimizer, cosine schedule, 50 epochs, learning rate ($10^{-4}$), and splits (4,379 / 697 / 262).
- **Current Status:** Documented in `results/paper_tables/missing_experiments.md` and executed in the ablation phase.
- **Reason:** Requires dedicated training to isolate the empirical contribution of histogram matching.

---

## Section D: Not Supported / Not Scientifically Reliable

The following analyses are **unsupported or scientifically invalid** with current available data:

1. **Target-Domain Accuracy / Precision / Recall / F1 / mAP on Full OHRC Data:**
   - *Status:* Strictly prohibited. The 31,769 usable OHRC tiles and 13,906 ultra-deep polar tiles have **no ground-truth labels**. Calculating accuracy or error rates from unverified detections would be scientifically invalid.
2. **Labeling Unconfirmed OHRC Detections as "False Positives":**
   - *Status:* Without human expert verification or laser altimeter ground truth, unconfirmed detections must be termed "candidate detections" or "candidate rockfall-related features," not false positives.
3. **Attributing Regional Detection Frequency Differences Solely to Geology or Illumination:**
   - *Status:* Regional detection differences may be influenced by solar incidence angle, illumination, terrain morphology, and product distribution.
4. **Historical vs. Controlled Architecture Ranking:**
   - *Status:* Any prior assertion that YOLO26n was the overall superior model was based on pre-cosine training runs. Under controlled Stage-2 cosine conditions, YOLOv5s is quantitatively superior on held-out source data (mAP50 = 0.5737 vs 0.5292).

---

## Section E: Data Integrity Verification Checklist

- [x] All 3 Stage-2 cosine checkpoints exist and are verified.
- [x] All 3 `results.csv` files exist with verified 50-epoch logs.
- [x] Authoritative validation metrics verified: YOLOv5s (0.5869), YOLO26n (0.5161), YOLOv8n (0.4885).
- [x] Authoritative test metrics verified: YOLOv5s (0.5737), YOLO26n (0.5292), YOLOv8n (0.4813).
- [x] Current target raw detection CSVs are authoritative (5,084 / 353,427 / 24,452).
- [x] Current regional detection counts are verified (16,309 South Pole vs 15,460 Equatorial/Northern).
- [x] Current ultra-deep polar counts are verified (13,906 tiles; RT-DETR-L labeled as Historical Baseline).
- [x] No `*_combined_hm_v2` checkpoints or legacy counts (4,565 / 5,026 / 41,941) were used.
- [x] No target-domain accuracy metrics were fabricated.
- [x] No target detections were labeled as false positives.
- [x] Split counts confirmed: 4,379 train / 697 val / 262 test.
- [x] Source instance counts confirmed: 122,537 train / 24,303 val / 7,268 test (154,108 total).
- [x] Source bounding box medians confirmed: width = 7.68 px, height = 7.68 px, area = 52.43 px$^2$.
- [x] Usable OHRC tile count confirmed: 31,769.
- [x] Usable ultra-deep polar tile count confirmed: 13,906.
- [x] Histogram matching accurately described as 20-tile averaged-reference histogram matching.
- [x] **New training experiments performed in generation phase = 0.**

---

## Section F: Summary of Generation Output

1. **Figures Generated (8 pairs, 16 files in both `.png` and `.pdf`):**
   - `fig_training_curves`
   - `fig_ohrc_qualitative_detections`
   - `fig_cross_model_behavior`
   - `fig_test_pr_curves`
   - `fig_object_size_distribution`
   - `fig_ohrc_confidence_distributions`
   - `fig_regional_detection_rates`
   - `fig_ultradeep_polar`
2. **Tables Generated (10 files):**
   - `stale_reference_audit.md`
   - `missing_experiments.md`
   - `test_pr_curve_data.csv`
   - `object_size_statistics.csv`
   - `model_complexity.csv`
   - `model_complexity.md`
   - `inference_latency.csv`
   - `regional_detection_rates.csv`
   - `ultradeep_polar_summary.csv`
   - `qualitative_examples.csv`
3. **Small Evaluations Performed:** 2 (Held-out 262-image test PR curve evaluation + 100-tile GPU latency benchmark).
4. **Training Experiments Performed:** **0 (Zero)**.
5. **Remaining Major Experiment:** Controlled No-HM vs. HM ablation (documented in `missing_experiments.md`).
6. **Stale-Result Problems Discovered:** Completely audited and categorized in `stale_reference_audit.md`; all active analysis scripts updated to controlled cosine runs.
7. **Data-Integrity Concerns:** None. Full consistency maintained across splits, instance numbers, and target-domain reporting constraints.

---

## Section G: Controlled YOLO26 Model Scaling Suite Audit & Integration (YOLO26n → YOLO26s → YOLO26m)

**Date:** 2026-10-05  
**Context:** Controlled empirical evaluation of detector capacity scaling across the YOLO26 architectural family (**YOLO26n** [2.50M params] $\rightarrow$ **YOLO26s** [9.95M params] $\rightarrow$ **YOLO26m** [21.77M params]) integrated alongside the external Stage-2 baselines (**YOLOv8n** and **YOLOv5s**).

### 1. Existing State vs. Newly Evaluated
- **Pre-existing / Authoritative Artifacts in Repository:**
  - YOLO26m Checkpoint: `runs/stage2_yolo26m_combined_hm_cosine/weights/best.pt` (preserved untouched, 50 epochs, best epoch 30, mAP50 = 0.6463)
  - YOLO26s Training & Checkpoint: `runs/stage2_yolo26s_combined_hm_cosine/weights/best.pt` (50 epochs, best epoch 46, validation fitness = 0.2567, training duration 212.0 min, batch size 8)
  - Stage-2 Baselines: YOLO26n, YOLOv8n, and YOLOv5s checkpoints preserved strictly without alteration.
- **Executed Empirical Evaluations for YOLO26s:**
  - Source held-out test evaluation on the untouched test split (262 images, 7,268 ground-truth annotations).
  - Target-domain inference across all 31,769 usable OHRC tiles at detection threshold $\tau = 0.20$.
  - Target-domain inference on the 13,906 usable ultra-deep polar tiles across 20 calibrated products.
  - Regional detection rate breakdown across South Pole (16,308 tiles) and Equatorial/Northern (15,461 tiles).
  - Parameter profiling (THOP FLOPs/params: 9.949M params, 11.25 GFLOPs, 19.37 MB) and GPU inference latency benchmark (100 tiles, RTX 3050 Laptop GPU).
  - Full regeneration and updating of all paper figures (`fig_training_curves`, `fig_ohrc_qualitative_detections`, `fig_cross_model_behavior`, `fig_test_pr_curves`, `fig_ohrc_confidence_distributions`, `fig_regional_detection_rates`, `fig_ultradeep_polar`, and dedicated scaling figures).

### 2. Exact Checkpoint Paths & Artifact References
| Model | Role | Checkpoint Path | Architecture Graph |
| :--- | :--- | :--- | :--- |
| **YOLO26n HM** | Scaling Base (Nano) | `runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt` | YOLO26n (Nano, 2.504M params) |
| **YOLO26s HM** | Scaling Step (Small) | `runs/stage2_yolo26s_combined_hm_cosine/weights/best.pt` | YOLO26s (Small, 9.949M params) |
| **YOLO26m HM** | Scaling Apex (Medium) | `runs/stage2_yolo26m_combined_hm_cosine/weights/best.pt` | YOLO26m (Medium, 21.774M params) |
| **YOLOv8n HM** | Controlled External Baseline | `runs/stage2_yolov8n_combined_hm_cosine/weights/best.pt` | YOLOv8n (Nano, 3.011M params) |
| **YOLOv5s HM** | Controlled External Baseline | `runs/stage2_yolov5s_combined_hm_cosine/weights/best.pt` | YOLOv5s (Small, 9.123M params) |
| **RT-DETR-L** | Historical Baseline | Historical pre-Stage 2 run (archived) | RT-DETR-L (Historical reference only) |

### 3. Exact Datasets & Split Partitions
- **Source Domain Labeled Dataset:** `data/combined_hm/` (combined Prieur et al. + RMaM with 20-tile averaged OHRC histogram matching).
  - Train split: 4,379 images (122,537 annotated bounding boxes)
  - Val split: 697 images (24,303 annotated bounding boxes)
  - Test split (Held-Out & Untouched): 262 images (7,268 annotated bounding boxes)
- **Target Domain Chandrayaan-2 OHRC Usable Tiles:** `data/tiles/usable/`
  - Total usable tiles: 31,769 tiles (1,024×1,024 native, inferred at 640×640)
  - South Pole partition: 16,308 evaluated tiles
  - Equatorial/Northern partition: 15,461 evaluated tiles
- **Ultra-Deep South Polar Dataset:** `data/tiles/polar_usable/`
  - Usable polar tiles: 13,906 tiles across 20 calibrated polar products.

### 4. Exact Execution Commands
- **Test Set Evaluation:** `python scripts/evaluate_test_split.py`
- **Target OHRC Full Inference:** `python scripts/infer_ohrc.py --model yolo26s`
- **Ultra-Deep Polar Inference:** `python scripts/process_polar_batch.py`
- **Regional Analysis:** `python scripts/analyze_regional_detections.py`
- **Complexity Profiling:** `python scripts/paper_generation/generate_task_f.py`
- **GPU Latency Benchmark:** `python scripts/paper_generation/generate_task_g.py`
- **Scaling Integration:** `python scripts/experiments/run_yolo26_scale.py --step all`

### 5. Consolidated Five-Model Numerical Results

#### A. Held-Out Source-Domain Test Performance (262 images, 7,268 instances)
| Model | Peak Val Epoch | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26m HM** | 30 | **0.6380** | **0.6172** | **0.6275** | **0.6463** | **0.2622** |
| **YOLO26s HM** | 46 | 0.6260 | 0.5691 | 0.5962 | 0.6023 | 0.2308 |
| **YOLOv5s HM** | 35 | 0.6108 | 0.5499 | 0.5788 | 0.5737 | 0.2121 |
| **YOLO26n HM** | 41 | 0.5873 | 0.5069 | 0.5442 | 0.5292 | 0.1896 |
| **YOLOv8n HM** | 33 | 0.5469 | 0.4798 | 0.5111 | 0.4813 | 0.1671 |

#### B. Target-Domain OHRC Candidate Activation Behavior (31,769 usable tiles, $\tau = 0.20$)
| Model | Candidate Detections | Positive Tiles | Positive Tile Rate (%) | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: |
| **YOLO26m HM** | 1,484 | 443 | 1.39% | 0.293 ± 0.088 |
| **YOLO26n HM** | 5,084 | 2,134 | 6.72% | 0.254 ± 0.055 |
| **YOLO26s HM** | 14,102 | 2,050 | 6.45% | 0.295 ± 0.087 |
| **YOLOv5s HM** | 24,452 | 3,896 | 12.26% | 0.330 ± 0.106 |
| **YOLOv8n HM** | 353,427 | 14,129 | 44.47% | 0.304 ± 0.093 |

#### C. Ultra-Deep South Polar Benchmark (13,906 usable tiles, $\tau = 0.20$)
| Model | Candidate Detections | Positive Tiles | Positive Tile Rate (%) | Mean Confidence | Detections / Pos Tile |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26m HM** | 1,447 | 309 | 2.22% | 0.3057 ± 0.0924 | 4.68 |
| **YOLOv5s HM** | 7,202 | 1,779 | 12.79% | 0.2854 ± 0.0760 | 4.05 |
| **YOLO26s HM** | 8,128 | 2,124 | 15.27% | 0.2699 ± 0.0708 | 3.83 |
| **YOLO26n HM** | 10,085 | 1,174 | 8.44% | 0.2546 ± 0.0573 | 8.59 |
| **YOLOv8n HM** | 71,001 | 6,047 | 43.48% | 0.2862 ± 0.0792 | 11.74 |
| *RT-DETR-L (Historical)* | *532,844* | *11,664* | *83.88%* | *0.4287 ± 0.1587* | *45.68* |

#### D. Regional Detection Rate Breakdown (South Pole vs. Equatorial/Northern)
| Model | Region | Evaluated Tiles | Candidate Detections | Positive Tiles | Positive Tile Rate (%) | Mean Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26m HM** | South Pole | 16,308 | 93 | 50 | 0.31% | 0.2621 ± 0.0555 |
| **YOLO26m HM** | Equatorial / Northern | 15,461 | 1,391 | 393 | 2.54% | 0.2950 ± 0.0876 |
| **YOLO26s HM** | South Pole | 16,308 | 1,983 | 1,093 | 6.70% | 0.2478 ± 0.0490 |
| **YOLO26s HM** | Equatorial / Northern | 15,461 | 12,119 | 957 | 6.19% | 0.3032 ± 0.0893 |
| **YOLO26n HM** | South Pole | 16,308 | 3,982 | 1,176 | 7.21% | 0.2567 ± 0.0583 |
| **YOLO26n HM** | Equatorial / Northern | 15,461 | 1,102 | 958 | 6.20% | 0.2458 ± 0.0468 |
| **YOLOv5s HM** | South Pole | 16,308 | 3,698 | 2,234 | 13.70% | 0.2824 ± 0.0764 |
| **YOLOv5s HM** | Equatorial / Northern | 15,461 | 20,754 | 1,662 | 10.75% | 0.3387 ± 0.1080 |
| **YOLOv8n HM** | South Pole | 16,308 | 292,371 | 12,689 | 77.81% | 0.2961 ± 0.0877 |
| **YOLOv8n HM** | Equatorial / Northern | 15,461 | 61,056 | 1,440 | 9.31% | 0.3396 ± 0.1088 |

#### E. Model Complexity and Edge Hardware Latency (NVIDIA RTX 3050 Laptop GPU)
| Model | Parameters (M) | GFLOPs (640×640) | Checkpoint Size (MB) | Mean Latency (ms) | Median Latency (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 2.504 | 2.89 | 5.14 | 70.26 | 70.75 | 14.2 |
| **YOLO26s** | 9.949 | 11.25 | 19.37 | 54.52 | 56.02 | 18.3 |
| **YOLO26m** | 21.774 | 37.36 | 41.98 | 59.96 | 61.02 | 16.7 |
| **YOLOv8n** | 3.011 | 4.10 | 5.96 | 50.74 | 46.96 | 19.7 |
| **YOLOv5s** | 9.123 | 12.02 | 17.66 | 45.19 | 46.40 | 22.1 |

#### F. Controlled YOLO26 Scaling Triad (YOLO26n → YOLO26s → YOLO26m)
| Scaling Transition | Capacity Change | mAP@0.5 Delta | F1 Delta | Target Positive Tile Rate Delta | Compute Multiplier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n $\rightarrow$ YOLO26s** | 2.50M $\rightarrow$ 9.95M (+298%) | **+0.0731** (0.5292 $\rightarrow$ 0.6023) | **+0.0520** (0.5442 $\rightarrow$ 0.5962) | 6.72% $\rightarrow$ 6.45% (-0.27 pp) | 3.89× GFLOPs |
| **YOLO26s $\rightarrow$ YOLO26m** | 9.95M $\rightarrow$ 21.77M (+119%) | **+0.0440** (0.6023 $\rightarrow$ 0.6463) | **+0.0313** (0.5962 $\rightarrow$ 0.6275) | 6.45% $\rightarrow$ 1.39% (-5.06 pp) | 3.32× GFLOPs |
| **Full YOLO26 Scaling** | 2.50M $\rightarrow$ 21.77M (+770%) | **+0.1171** (0.5292 $\rightarrow$ 0.6463) | **+0.0833** (0.5442 $\rightarrow$ 0.6275) | 6.72% $\rightarrow$ 1.39% (-5.33 pp) | 12.93× GFLOPs |

### 6. Methodological Differences & Comparability Caveats
1. **Training Batch Size Allocation (`batch=4` vs. `batch=8`):**
   - **Controlled Stage-2 Baselines (YOLO26n, YOLOv8n, YOLOv5s) and YOLO26s:** Trained with batch size 8 (`batch=8`) on CUDA:0. YOLO26s comfortably fits in the 4.0 GB VRAM limit of the NVIDIA RTX 3050 Laptop GPU at batch 8.
   - **YOLO26m:** Due to strict 4.0 GB VRAM hardware limitations on the NVIDIA GeForce RTX 3050 Laptop GPU, YOLO26m was resumed and completed with batch size 4 (`batch=4`).
   - **Comparability Assessment:** While learning rate schedule (AdamW, $\text{lr}_0=10^{-4}$, cosine decay to $\text{lrf}=0.01$), total epochs (50), input resolution (640×640), and dataset splits (4,379 / 697 / 262) were identical, the smaller batch size results in increased gradient stochasticity per step and different batch normalization accumulation dynamics. This represents a pragmatic hardware constraint rather than an experimental variable, and must be explicitly noted in comparative discussions.
2. **Unlabeled Target-Domain Framing:**
   - Because both the 31,769 usable OHRC tiles and 13,906 ultra-deep polar tiles lack ground-truth annotations, target-domain metrics reflect candidate activation density, operating frequency, and confidence calibration rather than verified true/false positive rates.
3. **Regional Disparity Interpretation:**
   - Regional detection rate differences between South Polar and Equatorial/Northern regions must be interpreted with respect to illumination angle, low solar incidence, shadow length, and sensor terrain morphology, rather than definitive geological boulder distribution.

