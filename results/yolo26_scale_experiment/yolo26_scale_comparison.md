# YOLO26 Model Scaling Comparison (Controlled Stage-2 HM Experiment)

| Model | Params (M) | GFLOPs | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Candidate Detections | Positive Tiles | Positive Tile Rate (%) | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n HM** | 2.50 | 2.89 | 5.1 | 0.5292 | 0.1896 | 0.5873 | 0.5069 | 0.5442 | 5,084 | 2,134 | 6.72% | 0.254 |
| **YOLO26m HM** | 21.77 | 37.36 | 42.0 | 0.6463 | 0.2622 | 0.6380 | 0.6172 | 0.6275 | 1,484 | 443 | 1.39% | 0.293 |
| **YOLOv8n HM** | 3.01 | 4.10 | 6.0 | 0.4813 | 0.1671 | 0.5469 | 0.4798 | 0.5111 | 353,427 | 14,129 | 44.47% | 0.304 |
| **YOLOv5s HM** | 9.12 | 12.02 | 17.7 | 0.5737 | 0.2121 | 0.6108 | 0.5499 | 0.5788 | 24,452 | 3,896 | 12.26% | 0.330 |

> **Methodological Notes:**
> 1. **Held-Out Test Set:** Evaluated on the untouched 262-image / 7,268-instance source test split under identical histogram-matched (HM) conditions.
> 2. **Target OHRC Data:** Descriptive candidate activations across 31,769 usable unlabeled OHRC tiles at detection threshold $\tau = 0.20$. Because the target domain is unlabeled, counts reflect candidate activation density rather than verified accuracy.
> 3. **Controlled Baselines:** YOLO26n, YOLOv8n, and YOLOv5s reflect authoritative repository Stage-2 cosine checkpoints without alteration.
