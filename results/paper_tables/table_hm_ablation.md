# Controlled Histogram Matching (HM vs. No-HM) Ablation on Held-Out Test Split

| Model | Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Best Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | HM | 0.5292 | 0.1896 | 0.5873 | 0.5069 | 0.5442 | 33 |
| **YOLO26n** | No-HM | 0.6736 | 0.2664 | 0.6898 | 0.6172 | 0.6515 | 39 |
| **YOLOv8n** | HM | 0.4813 | 0.1671 | 0.5469 | 0.4798 | 0.5111 | 21 |
| **YOLOv8n** | No-HM | 0.6455 | 0.2488 | 0.6756 | 0.5930 | 0.6316 | 35 |
| **YOLOv5s** | HM | 0.5737 | 0.2121 | 0.6108 | 0.5499 | 0.5788 | 47 |
| **YOLOv5s** | No-HM | 0.7122 | 0.2908 | 0.7207 | 0.6475 | 0.6821 | 35 |

## Numerical Differences (Delta = HM - No-HM)

| Model | Delta mAP@0.5 | Delta mAP@0.5:0.95 | Delta Precision | Delta Recall | Delta F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | -0.1444 | -0.0768 | -0.1025 | -0.1103 | -0.1073 |
| **YOLOv8n** | -0.1642 | -0.0817 | -0.1287 | -0.1132 | -0.1205 |
| **YOLOv5s** | -0.1385 | -0.0787 | -0.1099 | -0.0976 | -0.1033 |

> **Note:** Delta represents numerical difference (HM - No-HM) without normative value judgment.
