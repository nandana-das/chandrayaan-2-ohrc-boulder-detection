# Table IV: Chandrayaan-2 OHRC Regional Detection Comparison

Cross-regional boulder detection performance comparing the existing baseline regions against the newly added Ultra-Deep South Pole batch ($\le -89.5^\circ\text{S}$).

| region | model | detection_count | tiles_with_detection | total_evaluated_tiles | tile_detection_rate_pct | mean_confidence | std_confidence | detections_per_positive_tile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| South Pole (lat ≈ -70°S) | YOLO26n | 3982 | 1176 | 16308 | 7.21 | 0.2567 | 0.0564 | 3.39 |
| South Pole (lat ≈ -70°S) | YOLOv8n | 292371 | 12689 | 16308 | 77.81 | 0.2961 | 0.0879 | 23.04 |
| South Pole (lat ≈ -70°S) | YOLOv5s | 3698 | 2234 | 16308 | 13.7 | 0.2824 | 0.0799 | 1.66 |
| Equatorial (lat ≈ +60°N) | YOLO26n | 1102 | 958 | 15461 | 6.2 | 0.2458 | 0.0452 | 1.15 |
| Equatorial (lat ≈ +60°N) | YOLOv8n | 61056 | 1440 | 15461 | 9.31 | 0.3396 | 0.1156 | 42.4 |
| Equatorial (lat ≈ +60°N) | YOLOv5s | 20754 | 1662 | 15461 | 10.75 | 0.3387 | 0.1139 | 12.49 |
| Ultra-Deep South Pole (lat ≤ -89.5°S) | YOLO26n | 10085 | 1173 | 13906 | 8.44 | 0.2907 | 0.082 | 8.6 |
| Ultra-Deep South Pole (lat ≤ -89.5°S) | YOLOv8n | 71001 | 6047 | 13906 | 43.48 | 0.2959 | 0.088 | 11.74 |
| Ultra-Deep South Pole (lat ≤ -89.5°S) | YOLOv5s | 7202 | 1779 | 13906 | 12.79 | 0.3214 | 0.107 | 4.05 |
| Ultra-Deep South Pole (lat ≤ -89.5°S) | RT-DETR-L | 532844 | 11664 | 13906 | 83.88 | 0.3258 | 0.1063 | 45.68 |
