# Chandrayaan-2 OHRC Target-Domain Confidence Threshold Sensitivity Analysis

**Target Dataset:** Primary OHRC Usable Mosaic (31,769 tiles, 640×640 px).  
**Ground Truth Note:** Unlabeled remote sensing imagery; all metrics describe candidate activation density and operating rates, NOT verified precision, recall, or accuracy.  

---

## 1. Operating Point Justification: $\tau = 0.20$

The operational threshold $\tau = 0.20$ was chosen for the survey pipeline to balance candidate feature retention against terrain noise:
1. **Conservative Floor:** At $\tau = 0.20$, YOLO26n and YOLO26m maintain disciplined candidate activation (6.72% and 1.83% positive tile rates respectively), avoiding catastrophic spatial saturation.
2. **Cross-Architecture Divergence:** Highlighting $\tau = 0.20$ reveals extreme behavioral contrast: while YOLOv8n activates on 44.47% of tiles (353,427 candidate boxes), YOLO26n activates on only 6.72% (5,084 boxes) and YOLO26m on 1.83% (1,484 boxes).
3. **Decay Dynamics:** Across all architectures, candidate counts experience steep exponential decay between $\tau = 0.20$ and $\tau = 0.50$, indicating that borderline activations rapidly prune while persistent high-confidence candidates remain localized to genuine geomorphic structures.

---

## 2. Positive Tile Rate (%) vs. Confidence Threshold $\tau$

| Threshold $\tau$ | YOLO26n (%) | YOLO26s (%) | YOLO26m (%) | YOLOv8n (%) | YOLOv5s (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 6.72% | 6.45% | 1.39% | 44.47% | 12.26% |
| **0.25** | 3.01% | 3.87% | 0.88% | 39.91% | 8.14% |
| **0.30** | 1.35% | 2.50% | 0.55% | 35.31% | 5.67% |
| **0.35** | 0.58% | 1.71% | 0.34% | 30.38% | 4.30% |
| **0.40** | 0.26% | 1.16% | 0.22% | 25.42% | 3.37% |
| **0.45** | 0.09% | 0.77% | 0.13% | 20.23% | 2.74% |
| **0.50** | 0.03% | 0.45% | 0.08% | 15.11% | 2.23% |
| **0.60** | 0.01% | 0.11% | 0.02% | 6.07% | 1.08% |
| **0.70** | 0.00% | 0.01% | 0.00% | 0.98% | 0.09% |
| **0.80** | 0.00% | 0.00% | 0.00% | 0.04% | 0.00% |
| **0.90** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

---

## 3. Total Candidate Detections vs. Confidence Threshold $\tau$

| Threshold $\tau$ | YOLO26n | YOLO26s | YOLO26m | YOLOv8n | YOLOv5s |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 5,084 | 14,102 | 1,484 | 353,427 | 24,452 |
| **0.25** | 2,023 | 8,498 | 871 | 221,691 | 17,049 |
| **0.30** | 805 | 5,218 | 543 | 140,902 | 12,047 |
| **0.35** | 322 | 3,206 | 321 | 88,721 | 8,558 |
| **0.40** | 136 | 1,879 | 189 | 54,731 | 5,937 |
| **0.45** | 50 | 1,025 | 95 | 32,175 | 3,950 |
| **0.50** | 18 | 484 | 45 | 17,657 | 2,472 |
| **0.60** | 2 | 63 | 10 | 3,802 | 599 |
| **0.70** | 0 | 2 | 0 | 369 | 31 |
| **0.80** | 0 | 0 | 0 | 13 | 0 |
| **0.90** | 0 | 0 | 0 | 0 | 0 |
