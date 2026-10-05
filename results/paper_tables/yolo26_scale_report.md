# YOLO26 Model Scaling Empirical Evaluation Report

**Date:** 2026-10-03  
**Project:** Chandrayaan-2 OHRC Lunar Boulder & Rockfall Detection  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`  
**Script:** `scripts/experiments/run_yolo26_scale.py`  

---

## 1. Executive Summary & Objective
This experiment evaluates the empirical effect of scaling detector capacity within the YOLO26 model family from **YOLO26n** (2.50M parameters) to **YOLO26s** (10.01M parameters) and **YOLO26m** (21.90M parameters).

All models were trained strictly under the authoritative Stage-2 controlled histogram-matched (HM) domain adaptation protocol on the combined lunar dataset (`data/combined_hm/`) and evaluated on:
1. **Held-Out Source Test Split:** 262 images, 7,268 ground-truth boulder instances.
2. **Target Domain Chandrayaan-2 OHRC Imagery:** 31,769 usable tiles at confidence threshold $\tau = 0.20$.

---

## 2. Experimental Configuration & Controlled Protocol
- **Dataset:** `data/combined_hm/` (Train: 4,379, Val: 697, Test: 262 images; 100% split verified).
- **Photometric Conditioning:** 20-tile averaged-reference OHRC histogram matching.
- **Optimizer:** AdamW (`optimizer='AdamW'`).
- **Base Learning Rate:** $\text{lr}_0 = 0.0001$, with cosine learning rate decay (`cos_lr=True`, final $\text{lrf}=0.01$).
- **Total Epochs:** 50 epochs.
- **Image Resolution:** 640 × 640 px.
- **Freezing:** `freeze=0` (unconstrained backpropagation).
- **Device:** CUDA Device 0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GB VRAM).

### Memory Management & Batch Size Allocation:
- **YOLO26s:** Initialized with batch = 8. Training duration: 212.0 minutes (12717.7 seconds).
- **YOLO26m:** Initialized with batch = 4. Training duration: 375.4 minutes (22525.6 seconds).

---

## 3. Consolidated Comparative Results

# YOLO26 Model Scaling Comparison (Controlled Stage-2 HM Experiment)

| Model | Params (M) | GFLOPs | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Candidate Detections | Positive Tiles | Positive Tile Rate (%) | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n HM** | 2.50 | 2.89 | 5.1 | 0.5292 | 0.1896 | 0.5873 | 0.5069 | 0.5442 | 5,084 | 2,134 | 6.72% | 0.254 |
| **YOLO26s HM** | 9.95 | 11.25 | 19.4 | 0.6023 | 0.2308 | 0.6260 | 0.5691 | 0.5962 | 14,102 | 2,050 | 6.45% | 0.295 |
| **YOLO26m HM** | 21.77 | 37.36 | 42.0 | 0.6463 | 0.2622 | 0.6380 | 0.6172 | 0.6275 | 1,484 | 443 | 1.39% | 0.293 |
| **YOLOv8n HM** | 3.01 | 4.10 | 6.0 | 0.4813 | 0.1671 | 0.5469 | 0.4798 | 0.5111 | 353,427 | 14,129 | 44.47% | 0.304 |
| **YOLOv5s HM** | 9.12 | 12.02 | 17.7 | 0.5737 | 0.2121 | 0.6108 | 0.5499 | 0.5788 | 24,452 | 3,896 | 12.26% | 0.330 |

> **Methodological Notes:**
> 1. **Held-Out Test Set:** Evaluated on the untouched 262-image / 7,268-instance source test split under identical histogram-matched (HM) conditions.
> 2. **Target OHRC Data:** Descriptive candidate activations across 31,769 usable unlabeled OHRC tiles at detection threshold $\tau = 0.20$. Because the target domain is unlabeled, counts reflect candidate activation density rather than verified accuracy.
> 3. **Controlled Baselines:** YOLO26n, YOLOv8n, and YOLOv5s reflect authoritative repository Stage-2 cosine checkpoints without alteration.


---

## 4. Key Scientific Findings & Objective Observations

### A. Source-Domain Test Set Performance (LROC NAC / Prieur et al.)
1. **YOLO26s vs. YOLO26n:**
   - mAP@0.5 changed by +0.0731 (0.5292 $\rightarrow$ 0.6023).
   - F1 score changed by +0.0520 (0.5442 $\rightarrow$ 0.5962).
   - Precision: 0.6260 vs. 0.5873 (+0.0387).
   - Recall: 0.5691 vs. 0.5069 (+0.0622).
2. **YOLO26m vs. YOLO26n:**
   - mAP@0.5 changed by +0.1171 (0.5292 $\rightarrow$ 0.6463).
   - F1 score changed by +0.0833 (0.5442 $\rightarrow$ 0.6275).
   - Precision: 0.6380 vs. 0.5873 (+0.0507).
   - Recall: 0.6172 vs. 0.5069 (+0.1103).
3. **YOLO26m vs. YOLO26s Capacity Return:**
   - Increasing capacity from Small (10.01M params) to Medium (21.90M params) yielded an mAP@0.5 change of +0.0440.

### B. Target-Domain Descriptive Candidate Behavior (Unlabeled OHRC, tau = 0.20)
Because target Chandrayaan-2 OHRC imagery is completely unlabeled, detection counts describe candidate activation density and operating rates rather than true accuracy or false alarm rates:
- **YOLO26s:** Generated 14,102 candidate detections across 2,050 positive tiles (6.45%), with mean confidence of 0.295.
- **YOLO26m:** Generated 1,484 candidate detections across 443 positive tiles (1.39%), with mean confidence of 0.293.

### C. Computational Cost & Efficiency
- **YOLO26n:** 2.50M parameters, 2.89 GFLOPs (lowest compute footprint).
- **YOLO26s:** 9.95M parameters, 11.25 GFLOPs (3.89× compute increase over Nano).
- **YOLO26m:** 21.77M parameters, 37.36 GFLOPs (12.93× compute increase over Nano).

---

## 5. Methodological Limitations & Comparability Caveats
1. **Unlabeled Target Ground Truth:** The 31,769 OHRC tiles and 13,906 polar tiles lack human ground-truth labels. Differences in candidate count cannot be mathematically partitioned into true boulder detections versus spurious terrain activations.
2. **Training Batch Size Allocation (Batch Size 4 vs. Batch Size 8):**
   - **Controlled Stage-2 Baselines (YOLO26n, YOLOv8n, YOLOv5s) and YOLO26s:** Trained with batch size 8 (`batch=8`) on CUDA:0.
   - **YOLO26m Scaling Experiment:** Due to strict 4.0 GB VRAM limitations on the NVIDIA RTX 3050 Laptop GPU, YOLO26m was resumed and completed with batch size 4 (`batch=4`).
   - **Scientific Impact:** Batch size alters the stochastic gradient noise and batch normalization statistics during training. While both configurations utilized identical optimizer settings (AdamW, lr0=0.0001, cosine decay to lrf=0.01) over 50 epochs on the identical 4,379/697/262 data split, this difference must be explicitly disclosed. It represents a practical hardware-constrained adaptation rather than an intentional hyperparameter divergence.
3. **Single Scale Hardware:** Evaluations were conducted on a single 4 GB RTX 3050 GPU, reflecting real-world edge/portable deployment constraints.

---

## 6. Scientific Conclusion
Evaluating model capacity across the YOLO26 family demonstrates how architectural depth and width impact small-target lunar feature recognition under photometric domain adaptation, quantifying the exact trade-offs in source-domain generalization and computational cost.
