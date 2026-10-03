# Paper Results Generation Report

**Date:** 2026-10-03  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`  
**Authoritative Stage-2 Models:**
- `runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt`
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
