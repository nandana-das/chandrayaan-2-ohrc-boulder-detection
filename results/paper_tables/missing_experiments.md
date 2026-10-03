# Missing Experiments Documentation

**Date:** 2026-10-03  
**Project:** Lunar OHRC Boulder & Rockfall Detection  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`

---

## 1. Major Remaining Experiment: Controlled No-HM vs. HM Ablation

### Motivation
The current controlled Stage-2 models were trained on the combined dataset with 20-tile averaged OHRC histogram matching (HM) applied to bridge the photometric domain shift between the source domains (Prieur / RMaM) and target Chandrayaan-2 OHRC imagery. 

To isolate and rigorously quantify the exact empirical contribution of histogram matching toward domain alignment, a controlled **No-HM vs. HM ablation** is required.

### Required Experimental Protocol
The No-HM ablation models must be trained under identical conditions to the authoritative Stage-2 cosine checkpoints:

1. **Dataset Split:** Exactly identical train (4,379), validation (697), and test (262) image splits.
2. **Backbones:** YOLO26n, YOLOv8n, YOLOv5s.
3. **Initialization:** Exact Stage-1 `best.pt` weights.
4. **Optimizer:** AdamW (`optimizer='AdamW'`).
5. **Initial Learning Rate:** `lr0 = 0.0001`.
6. **Learning Rate Schedule:** Cosine decay (`cos_lr = True`).
7. **Epochs:** 50 epochs.
8. **Batch Size:** 8.
9. **Image Size:** 640×640.
10. **Hardware & Device:** CUDA device 0 (NVIDIA RTX 3050).
11. **Freezing:** `freeze = 0` (full model fine-tuning).
12. **Ablated Condition:** Raw source images (no histogram matching) vs. 20-tile averaged OHRC histogram-matched images.

### Planned Downstream Comparative Evaluations
Once trained, the No-HM models will be compared directly against the current controlled HM models on:
1. Held-out source test set (262 images, 7,268 instances) to determine if HM introduces source-domain representation degradation.
2. Target-domain OHRC candidate detection density and spatial distribution across the 31,769 usable tiles.
3. Regional and ultra-deep polar distribution consistency.
4. Diagnostic evaluation on the independent external OHRC benchmark (`data/roboflow_boulder_test/`).

### Current Status
**Deferred.** In accordance with project instructions, no new training was performed in the current diagnostic and reporting phase. This experiment stands as the primary pending empirical task for subsequent execution.
