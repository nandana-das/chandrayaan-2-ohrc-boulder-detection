# Diagnostic Evaluation Report: External Labelled OHRC Boulder Benchmark

**Date:** 2026-10-03  
**Target Benchmark:** External Chandrayaan-2 OHRC Lunar Crater Dataset (Roboflow v4i Test Split)  
**Location:** `data/roboflow_boulder_test/` (Isolated test split)  
**Evaluated Checkpoints:**
- **YOLO26n:** `runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt`
- **YOLOv8n:** `runs/stage2_yolov8n_combined_hm_cosine/weights/best.pt`
- **YOLOv5s:** `runs/stage2_yolov5s_combined_hm_cosine/weights/best.pt`

---

## 1. Exact Preprocessing Used in Each Experiment

### Experiment A — Original External Images
- **Input Images:** 10 test images from `data/roboflow_boulder_test/images/` evaluated in their exact as-exported state.
- **Native Image Specifications:** 512×512 resolution, 3-channel RGB (created during Roboflow grayscale export from identical channels), 8-bit unsigned integer (`uint8`).
- **External Dataset Provenance:** Exported by external creators with adaptive histogram equalization / auto-contrast, grayscale phosphor conversion, and EXIF orientation stripping. Original PDS4 / Chandrayaan-2 radiometric calibration metadata is absent.
- **Inference Preprocessing:** Images passed directly to standard Ultralytics inference pipeline at `imgsz=640` with standard letterboxing (padding to square aspect ratio with gray 114 border). No geometric transformation, augmentations, or photometric adjustments applied.

### Experiment B — Project-Style Image Representation
- **Diagnostic Directory:** `results/external_ohrc_diagnostic/project_preprocessed/`
- **Reference Profile:** 20-tile averaged OHRC reference profile (`results/ohrc_reference_tiles_20.txt`, generated deterministically with `seed=42` from usable OHRC tiles in `data/tiles/usable`, matching `scripts/histogram_matching.py`).
- **Applied Transformation:**
  1. Each of the 10 external 512×512 images was converted to single-channel grayscale (`uint8`).
  2. Exact histogram matching was computed using `skimage.exposure.match_histograms` against the 20-tile averaged OHRC reference cumulative distribution function.
  3. Matched single-channel images were saved as lossless 8-bit PNGs and replicated across 3 RGB channels to match model input requirements without altering geometry.
  4. Labels and coordinate files were copied directly, preserving exact 512×512 normalized bounding box coordinates (`xywh`).
- **Inference Preprocessing:** Identical to Experiment A (`imgsz=640`, letterboxed, `device=0`).

---

## 2. Results for Original Images

Evaluated on the 10 original external test images (338 ground-truth boulder annotations across 7 positive images, 3 negative images):

| Model | mAP50 (Val) | mAP50-95 (Val) | Preds ($\tau=0.20$) | TP | FP | FN | Precision ($\tau=0.20$) | Recall ($\tau=0.20$) | F1 ($\tau=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 0.0009 | 0.0001 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 |
| **YOLOv8n** | 0.0000 | 0.0000 | 218 | 0 | 218 | 338 | 0.0000 | 0.0000 | 0.0000 |
| **YOLOv5s** | 0.0001 | 0.0000 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 |

### Observations:
- At the standard project operational threshold ($\tau = 0.20$), **YOLO26n** and **YOLOv5s** produce zero detections across all 10 images. The zero FP count reflects complete prediction silence rather than precision.
- **YOLOv8n** generates 218 candidate predictions at $\tau=0.20$, but every single prediction fails to match any ground-truth boulder box at IoU $\ge 0.50$ (0 TP, 218 FP, 338 FN).
- Standard Ultralytics validation (`model.val()`) reports near-zero mAP50 ($0.0009$ for YOLO26n, $0.0000$ for YOLOv8n, $0.0001$ for YOLOv5s).

---

## 3. Results for Project-Preprocessed Images

Evaluated on the 10 histogram-matched external test images:

| Model | mAP50 (Val) | mAP50-95 (Val) | Preds ($\tau=0.20$) | TP | FP | FN | Precision ($\tau=0.20$) | Recall ($\tau=0.20$) | F1 ($\tau=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 0.0001 | 0.0000 | 83 | 0 | 83 | 338 | 0.0000 | 0.0000 | 0.0000 |
| **YOLOv8n** | 0.0003 | 0.0000 | 216 | 2 | 214 | 336 | 0.0093 | 0.0059 | 0.0072 |
| **YOLOv5s** | 0.0003 | 0.0000 | 30 | 0 | 30 | 338 | 0.0000 | 0.0000 | 0.0000 |

### Observations:
- Project-style histogram matching increases candidate predictions in the $\tau \ge 0.20$ regime for YOLO26n (from 0 to 83) and YOLOv5s (from 0 to 30), while YOLOv8n remains steady (216 vs 218).
- For YOLOv8n, 2 true positive matches are achieved at $\tau=0.20$ (Recall = 0.59%).
- Despite this slight shift in prediction activations, overall **mAP50 remains essentially zero** ($\le 0.0003$ across all models).
- **Critical finding:** Preprocessing mismatch alone does NOT explain the near-zero performance.

---

## 4. Threshold Comparison

Discrete diagnostic evaluations conducted at $\text{IoU} \ge 0.50$ across fixed confidence thresholds: $[0.20, 0.05, 0.01, 0.005, 0.001]$.

| Model | Preprocessing | Threshold | Predictions | TP | FP | FN | Precision | Recall | F1 | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | Original External | 0.200 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | 0.0001 |
| YOLO26n | Original External | 0.050 | 11 | 0 | 11 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | 0.0001 |
| YOLO26n | Original External | 0.010 | 241 | 1 | 240 | 337 | 0.0041 | 0.0030 | 0.0035 | 0.0009 | 0.0001 |
| YOLO26n | Original External | 0.005 | 838 | 3 | 835 | 335 | 0.0036 | 0.0089 | 0.0051 | 0.0009 | 0.0001 |
| YOLO26n | Original External | 0.001 | 2,695 | 24 | 2,671 | 314 | 0.0089 | 0.0710 | 0.0158 | 0.0009 | 0.0001 |
| **YOLOv8n** | Original External | 0.200 | 218 | 0 | 218 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| YOLOv8n | Original External | 0.050 | 1,332 | 0 | 1,332 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| YOLOv8n | Original External | 0.010 | 2,984 | 1 | 2,983 | 337 | 0.0003 | 0.0030 | 0.0005 | 0.0000 | 0.0000 |
| YOLOv8n | Original External | 0.005 | 3,000 | 1 | 2,999 | 337 | 0.0003 | 0.0030 | 0.0005 | 0.0000 | 0.0000 |
| YOLOv8n | Original External | 0.001 | 3,000 | 1 | 2,999 | 337 | 0.0003 | 0.0030 | 0.0005 | 0.0000 | 0.0000 |
| **YOLOv5s** | Original External | 0.200 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| YOLOv5s | Original External | 0.050 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| YOLOv5s | Original External | 0.010 | 12 | 0 | 12 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| YOLOv5s | Original External | 0.005 | 28 | 0 | 28 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| YOLOv5s | Original External | 0.001 | 398 | 4 | 394 | 334 | 0.0101 | 0.0118 | 0.0109 | 0.0001 | 0.0000 |
| **YOLO26n** | Project Preprocessed (HM) | 0.200 | 83 | 0 | 83 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| YOLO26n | Project Preprocessed (HM) | 0.050 | 300 | 1 | 299 | 337 | 0.0033 | 0.0030 | 0.0031 | 0.0001 | 0.0000 |
| YOLO26n | Project Preprocessed (HM) | 0.010 | 607 | 3 | 604 | 335 | 0.0049 | 0.0089 | 0.0063 | 0.0001 | 0.0000 |
| YOLO26n | Project Preprocessed (HM) | 0.005 | 829 | 3 | 826 | 335 | 0.0036 | 0.0089 | 0.0051 | 0.0001 | 0.0000 |
| YOLO26n | Project Preprocessed (HM) | 0.001 | 1,558 | 7 | 1,551 | 331 | 0.0045 | 0.0207 | 0.0074 | 0.0001 | 0.0000 |
| **YOLOv8n** | Project Preprocessed (HM) | 0.200 | 216 | 2 | 214 | 336 | 0.0093 | 0.0059 | 0.0072 | 0.0003 | 0.0000 |
| YOLOv8n | Project Preprocessed (HM) | 0.050 | 793 | 4 | 789 | 334 | 0.0050 | 0.0118 | 0.0070 | 0.0003 | 0.0000 |
| YOLOv8n | Project Preprocessed (HM) | 0.010 | 1,781 | 9 | 1,772 | 329 | 0.0051 | 0.0266 | 0.0086 | 0.0003 | 0.0000 |
| YOLOv8n | Project Preprocessed (HM) | 0.005 | 2,124 | 10 | 2,114 | 328 | 0.0047 | 0.0296 | 0.0081 | 0.0003 | 0.0000 |
| YOLOv8n | Project Preprocessed (HM) | 0.001 | 2,901 | 12 | 2,889 | 326 | 0.0041 | 0.0355 | 0.0074 | 0.0003 | 0.0000 |
| **YOLOv5s** | Project Preprocessed (HM) | 0.200 | 30 | 0 | 30 | 338 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0000 |
| YOLOv5s | Project Preprocessed (HM) | 0.050 | 184 | 1 | 183 | 337 | 0.0054 | 0.0030 | 0.0039 | 0.0003 | 0.0000 |
| YOLOv5s | Project Preprocessed (HM) | 0.010 | 591 | 3 | 588 | 335 | 0.0051 | 0.0089 | 0.0065 | 0.0003 | 0.0000 |
| YOLOv5s | Project Preprocessed (HM) | 0.005 | 912 | 5 | 907 | 333 | 0.0055 | 0.0148 | 0.0080 | 0.0003 | 0.0000 |
| YOLOv5s | Project Preprocessed (HM) | 0.001 | 1,947 | 12 | 1,935 | 326 | 0.0062 | 0.0355 | 0.0106 | 0.0003 | 0.0000 |

---

## 5. Ground-Truth Box-Size Comparison

To test the hypothesis that external boulders fail detection due to small-object scale truncation, geometric dimensions of all 338 external annotations were audited and compared against the 122,537 bounding boxes in the project's Stage 2 training dataset (`data/combined_hm`):

| Dataset / Configuration | Total Instances | Mean Width (px) | Median Width (px) | Min Width (px) | Max Width (px) | Mean Height (px) | Median Height (px) | Min Height (px) | Max Height (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **External OHRC (Native 512×512)** | 338 | 17.52 | 15.25 | 5.13 | 85.66 | 16.94 | 14.73 | 4.82 | 115.02 |
| **External OHRC (Rescaled 640×640)** | 338 | 21.90 | 19.06 | 6.41 | 107.07 | 21.17 | 18.41 | 6.03 | 143.77 |
| **Project Combined HM Training Set (640×640)** | 122,537 | 8.96 | 7.68 | 1.92 | 279.46 | 9.20 | 7.68 | 0.64 | 322.99 |

### Scale Analysis Findings:
- The external benchmark boulders have a median size of **15.25 × 14.73 pixels** at native 512×512, and **19.06 × 18.41 pixels** when letterboxed to model input resolution (640×640).
- By contrast, the training dataset has a median bounding box size of **7.68 × 7.68 pixels** at 640×640.
- **The external ground-truth boulders are ~2.5× LARGER than the median objects in the project training set.**
- Therefore, the near-zero performance is **not** caused by the external boulders being too small or falling below the network's receptive field or stride limit.

---

## 6. Image-Statistics Comparison

Pixel intensity distributions were computed across the 10 external benchmark images and 1,000 representative usable calibrated OHRC tiles (`data/tiles/usable`):

| Dataset Group | Sample Count | Channel Configuration | Mean Intensity | Std Intensity | Min Intensity | Max Intensity | Percentile 2 (p2) | Percentile 50 (Median) | Percentile 98 (p98) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **External OHRC Benchmark** | 10 | 3 identical channels (8-bit RGB) | 112.16 | 67.90 | 0 | 255 | 8.0 | 113.0 | 237.0 |
| **Project Calibrated OHRC Usable Tiles** | 1,000 | Single-channel lossless PNG (8-bit) | 118.16 | 60.23 | 0 | 255 | 0.0 | 118.0 | 255.0 |

### Observed Statistical Differences:
- Global central tendencies are remarkably close: mean intensity 112.16 vs 118.16, and median 113.0 vs 118.0.
- However, the external images exhibit an elevated standard deviation (67.90 vs 60.23) and compressed dynamic extremes (p2=8.0, p98=237.0 compared to p2=0.0, p98=255.0).
- This statistical profile is consistent with the external creator's documented use of adaptive contrast equalization / auto-contrast, which locally stretches midtone contrasts while clipping/shifting extremes.

---

## 7. Confidence-Distribution Analysis

Maximum detection confidences across all 10 images and counts of raw prediction candidates by confidence threshold:

| Model | Maximum Confidence | Preds ($\ge 0.20$) | Preds ($\ge 0.05$) | Preds ($\ge 0.01$) | Preds ($\ge 0.005$) | Preds ($\ge 0.001$) | Total Raw Candidates | Highest-Confidence Image |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26n** | **0.0893** | 0 | 11 | 241 | 838 | 2,695 | 2,695 | `383_png...` (Negative tile) |
| **YOLOv8n** | **0.5955** | 218 | 1,332 | 2,984 | 3,000 | 3,000 | 3,000 | `383_png...` (Negative tile) |
| **YOLOv5s** | **0.0271** | 0 | 0 | 12 | 28 | 398 | 398 | `201_png...` (Negative tile) |

### Key Diagnostic Points:
1. **YOLO26n and YOLOv5s peak well below 0.20:** YOLO26n never exceeds confidence 0.0893, and YOLOv5s never exceeds 0.0271 across the entire external dataset.
2. **False alarms dominate highest activations:** For all three models, the single highest-confidence detection occurs on image `383_png...` or `201_png...`, which are **negative images containing zero ground-truth boulders**. The models activate on sharp crater rim topography or high-contrast sunlit ridges, rather than boulder relief.
3. **Emergence of True Positives at extremely low confidence:**
   - On original images, YOLO26n recovers 24 true positives at $\tau=0.001$ (Recall 7.1%), but generates 2,671 false positives (Precision 0.89%).
   - On project-preprocessed images, YOLOv8n and YOLOv5s recover 12 true positives each at $\tau=0.001$ (Recall 3.55%), accompanied by ~1,900 to 2,900 false positives (Precision ~0.4–0.6%).
   - This demonstrates that while weak feature activations exist at the locations of some external ground-truth boulders, their confidence scores are suppressed by 1–2 orders of magnitude below the operational decision boundary.

---

## 8. Visual Observations

Detailed visual inspection was performed using the ground-truth contact sheet (`contact_sheet_gt_boulders.png`) and individual model overlay images in `results/external_ohrc_diagnostic/original/` and `project_preprocessed/`:

1. **Topographic Context and Geological Setting:**
   - The 7 positive external images (`2_png`, `20_png`, `24_png`, `26_png`, `41_png`, `65_png`, `321_png`) depict steep inner crater walls, slumped crater rims, and dense scree/talus fields.
   - Ground-truth boxes in these scenes are tightly clustered along crater crests and steep slopes, marking contiguous rocky patches, fractured bedrock outcrops, and aggregates of rubble.
   - This contrasts sharply with the Prieur and RMaM training datasets, which primarily feature discrete, isolated ejecta boulders casting clear, elongated shadows across relatively planar regolith.
2. **Shadow and Morphology Disconnect:**
   - In steep crater terrain, shadow direction and aspect vary widely due to local slope orientation. The characteristic high-albedo sunlit crest paired with a sharp, elongated down-sun shadow—which the Stage-2 models learned as the primary boulder signature—is absent or distorted in these scree clusters.
3. **Model Prediction Behaviors:**
   - **YOLOv8n** responds aggressively to continuous high-contrast linear edges, predicting long chains of false-positive bounding boxes along the illuminated rims of craters.
   - **YOLO26n** and **YOLOv5s** have learned stronger spatial suppression for continuous linear rims, resulting in near-total silence at $\tau \ge 0.05$.
   - When histogram matching is applied (Experiment B), the overall contrast is softened, which dampens some false alarms on crater rims and produces scattered low-confidence detections on rocky patches, yielding isolated true matches (e.g., 2 TP for YOLOv8n at $\tau=0.20$, and 1 TP for YOLO26n/YOLOv5s at $\tau=0.05$). However, predicted box boundaries frequently misalign with the dense, overlapping external annotations, failing the IoU $\ge 0.50$ criterion.

---

## 9. Cautious Conclusion

In accordance with strict scientific diagnostic standards, the findings are categorized into established facts, plausible explanations, and unestablished claims:

### A. Observed Facts
1. All three controlled Stage-2 models achieve near-zero mAP ($\text{mAP50} \le 0.0009$) under standard evaluation on the external Roboflow OHRC benchmark.
2. Applying the project's exact histogram matching pipeline (Experiment B) alters activation levels slightly (producing 2 TPs at $\tau=0.20$ for YOLOv8n and 1 TP at $\tau=0.05$ for YOLO26n/YOLOv5s), but **does not resolve the performance gap** ($\text{mAP50} \le 0.0003$).
3. The external ground-truth boulders are **not** undersized relative to the training set; their median width is 15.25 px (19.06 px at 640×640) compared to 7.68 px for the training dataset.
4. True positive detections do exist within the model networks at sub-operational confidence thresholds ($\tau = 0.01\text{--}0.001$), reaching up to 24 TPs for YOLO26n, but are accompanied by thousands of false positives (precision $< 1\%$).
5. The highest-confidence detections across all three models occur on negative images containing zero boulders, specifically triggering on illuminated crater rim crests.

### B. Plausible Explanations
1. **Geological Context & Morphological Shift:** The primary divergence appears to be the terrain type. The external benchmark focuses heavily on dense crater-rim boulder fields and scree slopes, where boulders appear as jagged, overlapping rock fragments without clear, isolated planar shadows. The models were trained predominantly on isolated ejecta boulders situated on open lunar plains.
2. **Annotation Granularity Mismatch:** The external dataset annotator outlined dense clusters of rock fragments along crater walls, where boundaries between individual boulders and fractured bedrock are ambiguous. Differences in annotation criteria between independent labeling teams can cause substantial IoU degradation.
3. **Loss of Radiometric Calibration:** The external dataset creator applied non-linear adaptive histogram equalization, auto-contrast, and 8-bit RGB export without PDS4 photometric metadata. This alters the local gradient signatures (sunlit-to-shadow ratios) that convolutional filters rely upon to distinguish 3D topographic relief from surface albedo variations.

### C. Things That CANNOT Yet Be Established
1. **It CANNOT be concluded that preprocessing mismatch is the primary cause of failure.** Experiment B demonstrates that mapping the external images to the project's calibrated OHRC reference histogram does not restore mAP.
2. **It CANNOT be concluded that the external annotations are incorrect.** The external labels reflect a legitimate interpretation of rocky crater-wall scree, representing an alternative annotation standard rather than an error.
3. **It CANNOT be concluded that the models are incapable of boulder detection in Chandrayaan-2 OHRC imagery.** The models achieve strong, consistent performance on the 7,268-instance Prieur test set ($\text{mAP50} \sim 0.63\text{--}0.65$) and produce geologically coherent spatial distributions across 31,769 calibrated OHRC tiles. The near-zero result on this benchmark reflects a domain and annotation boundary between isolated planar ejecta and clustered crater-rim scree, rather than general model failure.
