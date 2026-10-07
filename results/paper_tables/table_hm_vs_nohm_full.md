# Controlled Histogram Matching (HM vs. No-HM) Full Comparison Table

**Source Test Set:** Untouched 262-image held-out source split (7,268 ground truth instances).  
**Target OHRC Set:** 31,769 primary usable tiles (unlabeled; operational detection threshold $\tau = 0.20$).  
**Note:** Where No-HM experiments were not conducted (YOLO26s, YOLO26m), values are explicitly recorded as NA.

| Model | Condition | Test mAP50 | Test mAP50-95 | Test Precision | Test Recall | Test F1 | OHRC detections | OHRC positive tiles | OHRC positive-tile rate | Mean confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | HM | 0.5292 | 0.1896 | 0.5873 | 0.5069 | 0.5442 | 5,084 | 2,134 | 6.72% | 0.254 |
| **YOLO26n** | No-HM | 0.6736 | 0.2664 | 0.6898 | 0.6172 | 0.6515 | 2,429,131 | 26,956 | 84.85% | 0.347 |
| **YOLO26s** | HM | 0.6023 | 0.2308 | 0.626 | 0.5691 | 0.5962 | 14,102 | 2,050 | 6.45% | 0.295 |
| **YOLO26s** | No-HM | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| **YOLO26m** | HM | 0.6463 | 0.2622 | 0.638 | 0.6172 | 0.6275 | 1,484 | 443 | 1.39% | 0.293 |
| **YOLO26m** | No-HM | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| **YOLOv8n** | HM | 0.4813 | 0.1671 | 0.5469 | 0.4798 | 0.5111 | 353,427 | 14,129 | 44.47% | 0.304 |
| **YOLOv8n** | No-HM | 0.6455 | 0.2488 | 0.6756 | 0.593 | 0.6316 | 2,448,714 | 26,755 | 84.22% | 0.331 |
| **YOLOv5s** | HM | 0.5737 | 0.2121 | 0.6108 | 0.5499 | 0.5788 | 24,452 | 3,896 | 12.26% | 0.33 |
| **YOLOv5s** | No-HM | 0.7122 | 0.2908 | 0.7207 | 0.6475 | 0.6821 | 3,356,769 | 30,152 | 94.91% | 0.402 |

### Empirical Takeaways:
1. **Source Representation Trade-Off:** Across all architectures evaluated with No-HM (YOLO26n, YOLOv8n, YOLOv5s), the No-HM condition achieves higher in-distribution test metrics on the native source distribution ($\Delta \text{mAP@0.5} \approx -0.14$ to $-0.16$ under HM). This is because HM deliberately warps source histograms to match the target OHRC CDF, creating a synthetic domain divergence from the native source test distribution.
2. **Target Activation Regularization:** In the unlabeled target OHRC domain, No-HM models suffer from massive over-activation (positive-tile rates of 84.22%–94.91%, with 2.43M to 3.36M detections). HM stabilizes the detectors, reducing YOLO26n positive tiles from 84.85% to 6.72% (5,084 detections) and YOLOv5s from 94.91% to 12.26% (24,452 detections).
3. **Absence of Ground Truth:** Because target OHRC imagery is completely unlabeled, target candidate reductions must NOT be described as an accuracy improvement, but rather as evidence of photometric domain stabilization preventing widespread background false-triggering.
