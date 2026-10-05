# Model Complexity & Architectural Properties (Authoritative Stage-2 Models)

| Model | Architecture Base | Input Resolution | Parameters (M) | FLOPs (G) | File Size (MB) | Total Modules |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | YOLOv11-Nano (Custom Head) | 640 × 640 | 2.504M (2,504,190) | 2.89 G | 5.14 MB | 454 |
| **YOLO26s** | YOLOv11-Small (Custom Head) | 640 × 640 | 9.949M (9,948,638) | 11.25 G | 19.37 MB | 454 |
| **YOLO26m** | YOLOv11-Medium (Custom Head) | 640 × 640 | 21.774M (21,774,430) | 37.36 G | 41.98 MB | 490 |
| **YOLOv8n** | YOLOv8-Nano | 640 × 640 | 3.011M (3,011,043) | 4.10 G | 5.96 MB | 225 |
| **YOLOv5s** | YOLOv5-Small | 640 × 640 | 9.123M (9,122,579) | 12.02 G | 17.66 MB | 262 |

> **Note:** Parameter counts and GFLOPs were directly measured using `thop` profiling on a single 1×3×640×640 tensor input on the PyTorch execution graph. File size represents the serialized TorchScript/PyTorch checkpoint on disk.
