# Independent External OHRC Boulder Benchmark Evaluation

**Evaluation Date:** October 2026  
**Dataset Reference:** External labelled OHRC boulder test set (`data/roboflow_boulder_test/`)  
**Task:** One-Class Boulder Detection Evaluation (`class 0: boulder`)  
**Checkpoints Evaluated:** Controlled Stage-2 Cosine / AdamW Models (Untouched, No Fine-Tuning)

---

## 1. Executive Summary & Comparison Table

Each model was evaluated under an identical protocol using Ultralytics standard validation (`model.val`) for COCO mAP metrics and evaluated at the project operating threshold ($\tau = 0.20$, $\text{IoU} = 0.50$) for discrete classification counts:

| Model | Images | GT Boulders | Predicted | TP | FP | FN | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Inference Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 10 | 338 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 | **0.0009** | **0.0001** | 14.3 ms/img |
| **YOLOv8n** | 10 | 338 | 218 | 0 | 218 | 338 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | **0.0000** | 5.6 ms/img |
| **YOLOv5s** | 10 | 338 | 0 | 0 | 0 | 338 | 0.0000 | 0.0000 | 0.0000 | **0.0001** | **0.0000** | 10.9 ms/img |

*Note: Peak PR-curve metrics from continuous threshold sweep ($0.001 \le \tau \le 0.999$):*
- **YOLO26n**: Peak Recall = 0.0562 (19 true positives at $\tau \approx 0.002$), Peak Precision = 0.0083, Peak F1 = 0.0144, mAP@0.5 = 0.000902.
- **YOLOv8n**: Peak Recall = 0.0000, Peak Precision = 0.0000, Peak F1 = 0.0000, mAP@0.5 = 0.000000.
- **YOLOv5s**: Peak Recall = 0.0059 (2 true positives at $\tau \approx 0.002$), Peak Precision = 0.0088, Peak F1 = 0.0071, mAP@0.5 = 0.000078.

---

## 2. Evaluation Protocol & Dataset Provenance

1. **Dataset Independence**:
   - The test set consists of 10 images with 338 ground-truth boulder annotations.
   - An exhaustive pre-evaluation audit confirmed **zero filename overlap** and **zero SHA-256 hash overlap** with any training, validation, or unlabelled tile set in the repository (`data/tiles/usable`, `data/tiles/polar_usable`, `data/combined_hm`, `data/rmam`, and `data/prieur`).
   - The dataset was neither used for training, validation, fine-tuning, nor hyperparameter / threshold tuning.
2. **One-Class Isolation**:
   - The original multi-class labels (`Crater`, `boulder`, `large-crater`, `medium-crater`, `plane`, `small crater`) were filtered strictly to `boulder` and mapped to class `0`.
   - 7 images contain boulders; 3 images contain 0 boulders (background tiles).
3. **Resolution & Preprocessing**:
   - The native test images are $512 \times 512$ RGB.
   - For consistency with the Stage-2 training and inference pipeline, evaluation was executed at `imgsz=640` with standard Ultralytics aspect-preserving letterboxing.
   - No photometric or geometric transformations were applied.
4. **Attribution Notice**:
   - This benchmark is termed an **external labelled OHRC boulder test set** or **external OHRC boulder benchmark**. It is an independent community-labelled dataset and is not claimed to be an official ISRO-released ground-truth set.

---

## 3. Per-Image Detailed Breakdown ($\tau = 0.20$, $\text{IoU} = 0.50$)

| Image Filename | GT | YOLO26n Pred (TP / FP / FN) | YOLOv8n Pred (TP / FP / FN) | YOLOv5s Pred (TP / FP / FN) |
| :--- | :---: | :---: | :---: | :---: |
| `201_png.rf.e7ca0463...jpg` | 0 | 0 (0 / 0 / 0) | 1 (0 / 1 / 0) | 0 (0 / 0 / 0) |
| `20_png.rf.b37894c8...jpg` | 34 | 0 (0 / 0 / 34) | 0 (0 / 0 / 34) | 0 (0 / 0 / 34) |
| `24_png.rf.0774f007...jpg` | 54 | 0 (0 / 0 / 54) | 0 (0 / 0 / 54) | 0 (0 / 0 / 54) |
| `26_png.rf.ce6c874d...jpg` | 69 | 0 (0 / 0 / 69) | 11 (0 / 11 / 69) | 0 (0 / 0 / 69) |
| `2_png.rf.ffd89eab...jpg` | 75 | 0 (0 / 0 / 75) | 8 (0 / 8 / 75) | 0 (0 / 0 / 75) |
| `321_png.rf.fa2e57d0...jpg` | 8 | 0 (0 / 0 / 8) | 104 (0 / 104 / 8) | 0 (0 / 0 / 8) |
| `383_png.rf.06dc11f1...jpg` | 0 | 0 (0 / 0 / 0) | 70 (0 / 70 / 0) | 0 (0 / 0 / 0) |
| `41_png.rf.1913d757...jpg` | 97 | 0 (0 / 0 / 97) | 2 (0 / 2 / 97) | 0 (0 / 0 / 97) |
| `54_png.rf.76c491e7...jpg` | 0 | 0 (0 / 0 / 0) | 8 (0 / 8 / 0) | 0 (0 / 0 / 0) |
| `65_png.rf.a29ecf29...jpg` | 1 | 0 (0 / 0 / 1) | 14 (0 / 14 / 1) | 0 (0 / 0 / 1) |
| **Total** | **338** | **0 (0 / 0 / 338)** | **218 (0 / 218 / 338)** | **0 (0 / 0 / 338)** |

---

## 4. Visual Verification Outputs

Visual verification images displaying ground-truth bounding boxes (green) and predicted bounding boxes (magenta with confidence scores) have been exported for all 10 images to:
- `results/external_ohrc_eval/yolo26n/`
- `results/external_ohrc_eval/yolov8n/`
- `results/external_ohrc_eval/yolov5s/`

---

## 5. Factual Interpretation of Numerical Results

1. **Conservative vs. Over-Detection Tendencies**:
   - At the operational operating threshold ($\tau = 0.20$), **YOLO26n** and **YOLOv5s** produced zero detections across the 10 test tiles.
   - In contrast, **YOLOv8n** produced 218 candidate detections at $\tau = 0.20$, consistent with its known high target-domain triggering rate observed on the unlabelled OHRC full set.
2. **IoU Localization vs. Annotation Style**:
   - None of YOLOv8n's 218 predictions overlapped the 338 ground-truth annotations with $\text{IoU} \ge 0.50$, yielding $\text{TP} = 0$, $\text{FP} = 218$, and $\text{Precision} = 0.0000$. Visual inspection indicates that YOLOv8n detections on these images predominantly triggered on small crater rims and high-contrast shadow fringes.
3. **Continuous PR-Curve Sensitivity**:
   - Under standard COCO PR-curve validation down to $\tau = 0.001$, **YOLO26n** achieved the highest mAP@0.5 ($0.000902$) and highest peak recall ($0.0562$, capturing 19 ground-truth boulders), followed by **YOLOv5s** (mAP@0.5 = $0.000078$, peak recall $0.0059$, capturing 2 boulders). **YOLOv8n** achieved mAP@0.5 = $0.000000$.
4. **Domain and Preprocessing Discrepancies**:
   - The external Roboflow dataset was subjected to creator-applied contrast equalization and CRT phosphor grayscale conversion (`README.roboflow.txt`), which alters the radiometric profile compared to the native calibrated PDS4 percentile stretch ($p_2 - p_{98}$) used in the project's Stage-1 / Stage-2 pipeline.
