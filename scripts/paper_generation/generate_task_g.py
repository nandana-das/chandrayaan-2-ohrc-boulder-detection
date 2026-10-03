"""Task G: Inference Latency and Throughput (FPS) Benchmark on Chandrayaan-2 OHRC Tiles.
Evaluates the three Stage-2 models on a fixed, representative sample of 100 usable OHRC tiles.
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA:0).
Batch size: 1, Image size: 640x640, Precision: FP16 / Native PyTorch inference.
Includes 15 warmup inferences per model, followed by 100 measured iterations with GPU synchronization.
Saves:
- results/paper_tables/inference_latency.csv
"""
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
TILES_DIR = BASE / "data/tiles/usable"
RUNS = BASE / "runs"
RESULTS = BASE / "results"
OUT_TAB = RESULTS / "paper_tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

MODELS = {
    'YOLO26n': RUNS / 'stage2_yolo26n_combined_hm_cosine/weights/best.pt',
    'YOLOv8n': RUNS / 'stage2_yolov8n_combined_hm_cosine/weights/best.pt',
    'YOLOv5s': RUNS / 'stage2_yolov5s_combined_hm_cosine/weights/best.pt',
}

def benchmark_model(model_path, sample_tiles, warmup=15):
    model = YOLO(str(model_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Warmup runs
    for i in range(min(warmup, len(sample_tiles))):
        _ = model.predict(str(sample_tiles[i]), imgsz=640, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    times_ms = []
    for tile_path in sample_tiles:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        _ = model.predict(str(tile_path), imgsz=640, device=0, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        
    times = np.array(times_ms)
    mean_t = np.mean(times)
    median_t = np.median(times)
    std_t = np.std(times)
    min_t = np.min(times)
    max_t = np.max(times)
    fps = 1000.0 / mean_t if mean_t > 0 else 0.0
    
    return {
        'mean_ms': round(float(mean_t), 2),
        'median_ms': round(float(median_t), 2),
        'std_ms': round(float(std_t), 2),
        'min_ms': round(float(min_t), 2),
        'max_ms': round(float(max_t), 2),
        'fps': round(float(fps), 1),
    }

def main():
    print("=" * 60)
    print("Task G: Benchmarking Inference Latency & FPS on 100 OHRC Tiles")
    print("=" * 60)
    
    # Select 100 fixed, reproducible tiles (sorted by name)
    all_tiles = sorted(list(TILES_DIR.glob("*.png")))
    assert len(all_tiles) >= 100, f"Insufficient tiles found in {TILES_DIR}: {len(all_tiles)}"
    sample_tiles = all_tiles[:100]
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"Device: {device_name}")
    print(f"Sample Size: {len(sample_tiles)} tiles")
    print(f"Resolution: 640x640, Batch: 1")
    
    rows = []
    for name, p in MODELS.items():
        print(f"\nBenchmarking {name}...")
        metrics = benchmark_model(p, sample_tiles)
        print(f"  Mean: {metrics['mean_ms']} ms | Median: {metrics['median_ms']} ms | Std: {metrics['std_ms']} ms | FPS: {metrics['fps']}")
        rows.append({
            'Model': name,
            'Device': device_name,
            'Batch_Size': 1,
            'Image_Size': '640x640',
            'Tiles_Evaluated': len(sample_tiles),
            'Mean_Latency_ms': metrics['mean_ms'],
            'Median_Latency_ms': metrics['median_ms'],
            'Std_Latency_ms': metrics['std_ms'],
            'Min_Latency_ms': metrics['min_ms'],
            'Max_Latency_ms': metrics['max_ms'],
            'FPS': metrics['fps'],
        })
        
    df = pd.DataFrame(rows)
    csv_path = OUT_TAB / "inference_latency.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Done] Saved latency benchmark table: {csv_path}")

if __name__ == '__main__':
    main()
