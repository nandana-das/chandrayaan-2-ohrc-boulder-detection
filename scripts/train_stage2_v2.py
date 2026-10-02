"""Stage 2 training — all 3 models on combined HM data with explicit AdamW and cosine LR schedule."""
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA = str(BASE / "data/combined_hm/dataset_combined_hm.yaml")
RUNS = BASE / "runs"

MODELS = ['yolo26n', 'yolov8n', 'yolov5s']

TRAIN_CONFIG = {
    'epochs': 50,
    'imgsz': 640,
    'batch': 8,
    'lr0': 0.0001,
    'optimizer': 'AdamW',
    'cos_lr': True,
    'freeze': 0,
    'device': 0,
}

def train_model(model_name: str):
    weights = RUNS / f'stage1_{model_name}_combined' / 'weights' / 'best.pt'
    run_name = f'stage2_{model_name}_combined_hm_cosine'
    
    print("\n" + "=" * 70)
    print(f"STAGE 2 TRAINING: {model_name.upper()}")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Stage 1 Weights: {weights}")
    print(f"Weights Exist: {weights.exists()} ({weights.stat().st_size if weights.exists() else 0} bytes)")
    print(f"Dataset YAML: {DATA}")
    print(f"Run Directory: {RUNS / run_name}")
    print("Training Configuration Parameters:")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")
    
    assert weights.exists(), f"Stage 1 weights file not found: {weights}"
    
    model = YOLO(str(weights))
    results = model.train(
        data=DATA,
        project=str(RUNS),
        name=run_name,
        **TRAIN_CONFIG
    )
    print(f"\nCompleted Stage 2 training for {model_name}.\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Training with AdamW and Cosine LR")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'yolo26n', 'yolov8n', 'yolov5s'],
                        help="Model to train ('all' or specific model)")
    args = parser.parse_args()
    
    target_models = MODELS if args.model == 'all' else [args.model]
    for model_name in target_models:
        train_model(model_name)

if __name__ == '__main__':
    main()
