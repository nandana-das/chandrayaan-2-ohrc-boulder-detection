"""Run inference on all OHRC tiles (31,769 usable tiles) using Stage 2 cosine models."""
import argparse
import time
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
TILES = str(BASE / "data/tiles/usable")
RUNS = BASE / "runs"
RESULTS_DIR = BASE / "results"
OHRC_INFERENCE_DIR = RESULTS_DIR / "ohrc_inference"

MODELS = {
    'yolo26n': RUNS / 'stage2_yolo26n_combined_hm_cosine' / 'weights' / 'best.pt',
    'yolo26m': RUNS / 'stage2_yolo26m_combined_hm_cosine' / 'weights' / 'best.pt',
    'yolov8n': RUNS / 'stage2_yolov8n_combined_hm_cosine' / 'weights' / 'best.pt',
    'yolov5s': RUNS / 'stage2_yolov5s_combined_hm_cosine' / 'weights' / 'best.pt',
}

def run_inference(model_name: str, weights: Path, conf: float = 0.20):
    print("\n" + "=" * 70)
    print(f"OHRC INFERENCE: {model_name.upper()} (conf={conf})")
    print(f"Weights: {weights}")
    print(f"Tiles Directory: {TILES}")
    print("=" * 70)

    out_img = Path(f'C:/ohrc_detections/{model_name}/images')
    out_lbl = Path(f'C:/ohrc_detections/{model_name}/labels')
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    OHRC_INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OHRC_INFERENCE_DIR / f"raw_detections_{model_name}.csv"

    model = YOLO(str(weights))
    all_detections = []
    total = 0
    hits = 0
    scanned = 0
    t0 = time.time()

    for r in model.predict(source=TILES, imgsz=640, conf=conf,
                          stream=True, device=0, verbose=False):
        scanned += 1
        tile_name = Path(r.path).name
        n_boxes = len(r.boxes)
        if n_boxes > 0:
            hits += 1
            total += n_boxes
            r.save(filename=str(out_img / tile_name))
            r.save_txt(str(out_lbl / (Path(r.path).stem + '.txt')))

            for b in r.boxes:
                c = float(b.conf[0])
                coords = b.xyxy[0].tolist()
                all_detections.append({
                    'tile': tile_name,
                    'conf': round(c, 4),
                    'x1': round(coords[0], 2),
                    'y1': round(coords[1], 2),
                    'x2': round(coords[2], 2),
                    'y2': round(coords[3], 2)
                })

        if scanned % 5000 == 0 or scanned == 31769:
            elapsed = time.time() - t0
            fps = scanned / elapsed if elapsed > 0 else 0
            print(f"[{scanned}/31769] Tiles processed ({fps:.1f} tiles/sec) | Detections: {total:,} | Positive Tiles: {hits:,}")

    det_df = pd.DataFrame(all_detections)
    det_df.to_csv(out_csv, index=False)
    print(f"\n[Saved CSV] {out_csv} ({len(det_df):,} detections)")

    mean_conf = det_df['conf'].mean() if len(det_df) > 0 else 0.0
    hit_rate = (hits / scanned * 100) if scanned > 0 else 0.0

    print(f"\nSummary for {model_name}:")
    print(f"  Scanned Tiles:    {scanned:,}")
    print(f"  Total Detections: {total:,}")
    print(f"  Positive Tiles:   {hits:,}")
    print(f"  Tile Hit Rate:    {hit_rate:.2f}%")
    print(f"  Mean Confidence:  {mean_conf:.3f}\n")

    return {
        'Model': model_name,
        'Total Detections': total,
        'Positive Tiles': hits,
        'Scanned Tiles': scanned,
        'Hit Rate (%)': round(hit_rate, 2),
        'Mean Confidence': round(mean_conf, 3)
    }

def main():
    parser = argparse.ArgumentParser(description="OHRC Target Domain Inference")
    parser.add_argument('--conf', type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'yolo26n', 'yolo26m', 'yolov8n', 'yolov5s'],
                        help="Specific model to run or 'all'")
    args = parser.parse_args()

    target_models = MODELS if args.model == 'all' else {args.model: MODELS[args.model]}
    results = []
    for name, weights in target_models.items():
        res = run_inference(name, weights, conf=args.conf)
        results.append(res)

    print("\n" + "=" * 70)
    print("FINAL OHRC INFERENCE COMPARISON (conf=0.20)")
    print("=" * 70)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    main()
