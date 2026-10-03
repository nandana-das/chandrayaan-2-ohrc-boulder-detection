# Paper Results Generation Report

**Date:** 2026-10-03  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`  
**Authoritative Stage-2 Models:**
- `runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolov8n_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolov5s_combined_hm_cosine/weights/best.pt`

---

## Executive Overview
This document compiles the exhaustive inventory of all publication-ready figures, tables, and benchmark analyses generated for the manuscript based exclusively on the controlled Stage-2 cosine domain adaptation models. All artifacts have been generated in dual vector (PDF) and raster (>=300 DPI PNG) formats alongside structured CSV and Markdown data tables.

---

## Section A: Generated from Existing Outputs (No New Inference)

### 1. `fig_training_curves.png` / `fig_training_curves.pdf`
- **Filename:** `results/paper_figures/fig_training_curves.png` & `.pdf`
- **Data Source:** `runs/stage2_*_combined_hm_cosine/results.csv` (50 epochs per model)
- **New Inference Required:** No
- **Training Required:** No (read from existing run logs)
- **Intended Paper Section:** Section 3.2 (Stage-2 Controlled Domain Adaptation Training Dynamics)
- **One-Sentence Interpretation:** Controlled Stage-2 fine-tuning with AdamW and cosine learning rate decay converges smoothly across all three backbones, with YOLOv5s reaching the highest validation mAP@0.5 (0.612) and YOLO26n reaching 0.589 while maintaining low bounding box loss.
- **Scientific Caveat:** Validation curves reflect performance on the combined source domain (Prieur + RMaM) under 20-tile averaged OHRC histogram matching and do not directly measure target-domain generalization.

### 2. `fig_ohrc_confidence_distributions.png` / `fig_ohrc_confidence_distributions.pdf`
- **Filename:** `results/paper_figures/fig_ohrc_confidence_distributions.png` & `.pdf`
- **Data Source:** `results/ohrc_inference/raw_detections_*.csv` (31,769 usable tiles)
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 4.2 (Target-Domain Confidence Calibration & Operating Regimes)
- **One-Sentence Interpretation:** YOLOv8n exhibits an overwhelming mass of threshold-adjacent detections near $\tau = 0.20$ (353,427 candidates, mean 0.304), whereas YOLO26n is highly conservative (5,084 candidates, mean 0.254) and YOLOv5s shows balanced confidence decay (24,452 candidates, mean 0.330).
- **Scientific Caveat:** Confidence scores on unlabeled target imagery reflect network output activations under domain shift and cannot be interpreted as posterior probabilities of true boulder presence without ground truth.

### 3. `fig_regional_detection_rates.png` / `fig_regional_detection_rates.pdf` & `regional_detection_rates.csv`
- **Filename:** `results/paper_figures/fig_regional_detection_rates.png` & `.pdf`, `results/paper_tables/regional_detection_rates.csv`
- **Data Source:** `results/confidence_analysis/ohrc_regional_detection_comparison.csv` (16,309 South Pole tiles vs. 15,460 Equatorial/Northern tiles)
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 5.1 (Regional Detection Rate Disparities)
- **One-Sentence Interpretation:** YOLOv8n triggers on 77.81% of South Polar tiles versus only 9.31% in equatorial regions, while YOLO26n and YOLOv5s maintain relatively stable regional detection frequencies (7.21% vs 6.20% and 13.70% vs 10.75%, respectively).
- **Scientific Caveat:** Differences in regional candidate rates must NOT be cited as evidence of intrinsic geological variation, as polar OHRC imagery features extreme grazing solar incidence angles ($>75^\circ$), long shadows, and phase-angle disparities that strongly confound detector feature maps.

### 4. `fig_ultradeep_polar.png` / `fig_ultradeep_polar.pdf` & `ultradeep_polar_summary.csv`
- **Filename:** `results/paper_figures/fig_ultradeep_polar.png` & `.pdf`, `results/paper_tables/ultradeep_polar_summary.csv`
- **Data Source:** `results/polar_inference/` (13,906 usable tiles from 20 calibrated polar products)
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 5.2 (Ultra-Deep South Polar Benchmark Evaluation)
- **One-Sentence Interpretation:** Under extreme polar illumination, the historical RT-DETR-L baseline flagged 83.88% of tiles (532,844 candidates), whereas controlled Stage-2 models exhibit far more restrained behavior: YOLO26n flags 8.44% (10,085 candidates), YOLOv5s flags 12.79% (7,202 candidates), and YOLOv8n flags 43.48% (71,001 candidates).
- **Scientific Caveat:** RT-DETR-L was trained under an earlier, unconstrained protocol and serves strictly as a historical high-density sensitivity baseline; it is not directly comparable as a controlled ablation.

### 5. `fig_object_size_distribution.png` / `fig_object_size_distribution.pdf` & `object_size_statistics.csv`
- **Filename:** `results/paper_figures/fig_object_size_distribution.png` & `.pdf`, `results/paper_tables/object_size_statistics.csv`
- **Data Source:** Source dataset annotation files in `data/combined_hm/` (154,108 bounding boxes across Train [122,537], Val [24,303], and Test [7,268])
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 2.3 (Source Dataset Morphometric Properties)
- **One-Sentence Interpretation:** Boulder annotations in the source domain are heavily skewed toward sub-30-pixel features (median width 7.1 px, median height 7.4 px, median area 52.8 px$^2$), demonstrating that small-object feature resolution dominates the learning objective.
- **Scientific Caveat:** Annotation boundaries reflect manual labeling conventions in the source datasets (Prieur et al. / RMaM) and may omit boulders near or below the optical Rayleigh resolution limit.

### 6. `fig_cross_model_behavior.png` / `fig_cross_model_behavior.pdf`
- **Filename:** `results/paper_figures/fig_cross_model_behavior.png` & `.pdf`
- **Data Source:** Same-tile extraction from `results/ohrc_inference/raw_detections_*.csv` and `data/tiles/usable/`
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 4.3 (Qualitative Cross-Model Discrepancy Analysis)
- **One-Sentence Interpretation:** Side-by-side tile panels demonstrate that on identical lunar terrain, YOLO26n isolates single prominent boulders, YOLOv5s captures coherent boulder clusters, and YOLOv8n triggers dense bounding boxes over low-contrast ejecta textures.
- **Scientific Caveat:** The visualizations illustrate qualitative behavioral differences in sensitivity and false alarm rates, but cannot establish correctness on unlabeled imagery.

### 7. `fig_ohrc_qualitative_detections.png` / `fig_ohrc_qualitative_detections.pdf` & `qualitative_examples.csv`
- **Filename:** `results/paper_figures/fig_ohrc_qualitative_detections.png` & `.pdf`, `results/paper_tables/qualitative_examples.csv`
- **Data Source:** `results/ohrc_inference/raw_detections_*.csv` and `data/tiles/usable/`
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 4.4 (Qualitative Target-Domain Morphological Detections)
- **One-Sentence Interpretation:** A 3×3 grid of candidate detections exemplifies isolated boulders with cast shadows, crater-rim block fields, texture boundaries, and shadow-associated interface features.
- **Scientific Caveat:** All bounding boxes are explicitly designated as candidate detections; no true positive or false positive claims are asserted.

### 8. `model_complexity.csv` / `model_complexity.md`
- **Filename:** `results/paper_tables/model_complexity.csv` & `.md`
- **Data Source:** Model checkpoint graphs profiled via `thop`
- **New Inference Required:** No
- **Training Required:** No
- **Intended Paper Section:** Section 3.1 (Detector Architecture & Computational Efficiency)
- **One-Sentence Interpretation:** YOLO26n achieves the highest computational efficiency (2.50M parameters, 2.89 GFLOPs, 5.14 MB), followed by YOLOv8n (3.01M parameters, 4.10 GFLOPs, 5.96 MB), while YOLOv5s requires over 3× the parameters and FLOPs (9.12M parameters, 12.02 GFLOPs, 17.66 MB).
- **Scientific Caveat:** FLOPs are computed at 640×640 input resolution on a batch size of 1; actual deployment throughput also depends on memory bandwidth and postprocessing (NMS).

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
- **Intended Paper Section:** Section 3.4 (Deployment Latency & Throughput Benchmark)
- **One-Sentence Interpretation:** All three models achieve real-time throughput on mobile GPU hardware, with YOLOv8n processing at 31.9 FPS (mean 31.37 ms), YOLOv5s at 30.3 FPS (mean 33.00 ms), and YOLO26n at 20.5 FPS (mean 48.68 ms).
- **Scientific Caveat:** Benchmark times include preprocessing and PyTorch model execution with GPU synchronization, but exclude disk I/O and large-scale tile mosaic stitching.

---

## Section C: Still Requires New Training

### 1. Controlled No-HM vs. HM Ablation
- **Description:** Training identical YOLO26n, YOLOv8n, and YOLOv5s architectures on the raw source dataset (without histogram matching) using the exact same AdamW optimizer, cosine schedule, 50 epochs, learning rate ($10^{-4}$), and splits (4,379 / 697 / 262).
- **Current Status:** **DEFERRED / PENDING.** Documented in `results/paper_tables/missing_experiments.md`.
- **Reason:** Requires dedicated GPU training hours and was explicitly excluded from the current diagnostic and reporting phase.

---

## Section D: Not Supported / Not Scientifically Reliable

The following analyses were investigated and determined to be **unsupported or scientifically invalid** with current available data:

1. **Target-Domain Accuracy / Precision / Recall / F1 / mAP on Full OHRC Data:**
   - *Status:* Strictly prohibited. The 31,769 usable OHRC tiles and 13,906 ultra-deep polar tiles have **no ground-truth labels**. Calculating pseudo-accuracy from unverified detections would be scientifically dishonest.
2. **Labeling Unconfirmed OHRC Detections as "False Positives":**
   - *Status:* Without human expert geological verification or laser altimeter ground truth, unconfirmed detections must be termed "candidate detections" or "candidate rockfall features," not false positives.
3. **Attributing Regional Detection Frequency Differences to Geological Phenomena:**
   - *Status:* The 10× difference in YOLOv8n polar detection rates versus equatorial rates is primarily driven by photometric and incidence angle differences, not verifiable boulder density differences.
4. **Historical vs. Controlled Architecture Ranking:**
   - *Status:* Any prior assertion that YOLO26n was the overall superior model was based on pre-cosine training runs. Under controlled Stage-2 cosine conditions, YOLOv5s is quantitatively superior on held-out source data (mAP50 = 0.5737 vs 0.5292).

---

## Section E: Data Integrity Verification Checklist

- [x] All 3 Stage-2 cosine checkpoints exist and are verified.
- [x] All 3 `results.csv` files exist with verified 50-epoch logs.
- [x] Current target raw detection CSVs are authoritative (5,084 / 353,427 / 24,452).
- [x] Current test metrics are verified (YOLOv5s: 0.5737, YOLO26n: 0.5292, YOLOv8n: 0.4813).
- [x] Current regional detection counts are verified (16,309 South Pole vs 15,460 Equatorial/Northern).
- [x] Current ultra-deep polar counts are verified (13,906 tiles; RT-DETR-L labeled as historical baseline).
- [x] No `*_combined_hm_v2` checkpoints or legacy counts (4,565 / 5,026 / 41,941) were used.
- [x] No target-domain accuracy metrics were fabricated.
- [x] No target detections were labeled as false positives.
- [x] Split counts confirmed: 4,379 train / 697 val / 262 test.
- [x] Source instance counts confirmed: 122,537 train / 24,303 val / 7,268 test (154,108 total).
- [x] Usable OHRC tile count confirmed: 31,769.
- [x] Usable ultra-deep polar tile count confirmed: 13,906.
- [x] Histogram matching accurately described as 20-tile averaged-reference histogram matching.
- [x] **New training experiments performed = 0.**

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
