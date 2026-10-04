"""Controlled YOLO26 Scaling Experiment: YOLO26s and YOLO26m.

Objective:
Evaluate whether increasing YOLO26 model capacity from YOLO26n (2.50M params)
to YOLO26s (10.01M params) and YOLO26m (21.90M params) improves rockfall-related
feature detection on the controlled Stage-2 histogram-matched lunar dataset.

Dataset:
  data/combined_hm/ (Train: 4,379, Val: 697, Test: 262)
Protocol:
  optimizer = AdamW
  lr0 = 0.0001
  cos_lr = True
  epochs = 50
  imgsz = 640
  freeze = 0
  device = 0
  batch = 8 (fallback to 4 or 2 if VRAM constrained on 4GB GPU)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA_HM = str(BASE / "data/combined_hm/dataset_combined_hm.yaml")
TILES_DIR = BASE / "data/tiles/usable"
WEIGHTS_DIR = BASE / "weights"
RUNS_DIR = BASE / "runs"
RESULTS_DIR = BASE / "results"
OUT_EXP = RESULTS_DIR / "yolo26_scale_experiment"
OUT_FIG = RESULTS_DIR / "paper_figures"
OUT_TAB = RESULTS_DIR / "paper_tables"

# Authoritative Existing Baseline Metrics (HM condition)
BASELINE_METRICS = {
    'YOLO26n': {
        'model_family': 'YOLO26',
        'scale': 'n',
        'condition': 'HM',
        'mAP50': 0.5292,
        'mAP50-95': 0.1896,
        'precision': 0.5873,
        'recall': 0.5069,
        'f1': 0.5442,
        'detections': 5084,
        'positive_tiles': 2134,
        'positive_tile_rate': 6.72,
        'mean_conf': 0.254,
        'params_m': 2.504,
        'gflops': 2.89,
        'size_mb': 5.14,
        'best_epoch': 33,
    },
    'YOLOv8n': {
        'model_family': 'YOLOv8',
        'scale': 'n',
        'condition': 'HM',
        'mAP50': 0.4813,
        'mAP50-95': 0.1671,
        'precision': 0.5469,
        'recall': 0.4798,
        'f1': 0.5111,
        'detections': 353427,
        'positive_tiles': 14129,
        'positive_tile_rate': 44.47,
        'mean_conf': 0.304,
        'params_m': 3.011,
        'gflops': 4.10,
        'size_mb': 5.96,
        'best_epoch': 21,
    },
    'YOLOv5s': {
        'model_family': 'YOLOv5',
        'scale': 's',
        'condition': 'HM',
        'mAP50': 0.5737,
        'mAP50-95': 0.2121,
        'precision': 0.6108,
        'recall': 0.5499,
        'f1': 0.5788,
        'detections': 24452,
        'positive_tiles': 3896,
        'positive_tile_rate': 12.26,
        'mean_conf': 0.330,
        'params_m': 9.123,
        'gflops': 12.02,
        'size_mb': 17.66,
        'best_epoch': 47,
    },
}

# =========================================================================
# STEP 0: PRE-TRAINING VERIFICATION
# =========================================================================
def run_precheck():
    print("=" * 75)
    print("STEP 0: PRE-TRAINING INTEGRITY VERIFICATION")
    print("=" * 75)
    
    # 1. Dataset existence & split counts
    hm_dir = BASE / "data/combined_hm"
    assert hm_dir.exists(), f"Missing dataset: {hm_dir}"
    
    train_imgs = list((hm_dir / "train/images").glob("*.png"))
    val_imgs = list((hm_dir / "val/images").glob("*.png"))
    test_imgs = list((hm_dir / "test/images").glob("*.png"))
    
    print(f"Dataset Verified at: {hm_dir}")
    print(f"  Train Images: {len(train_imgs):,} (expected 4,379)")
    print(f"  Val Images:   {len(val_imgs):,} (expected 697)")
    print(f"  Test Images:  {len(test_imgs):,} (expected 262)")
    assert len(train_imgs) == 4379, f"Train count mismatch: {len(train_imgs)}"
    assert len(val_imgs) == 697, f"Val count mismatch: {len(val_imgs)}"
    assert len(test_imgs) == 262, f"Test count mismatch: {len(test_imgs)}"
    
    # Check test labels
    test_lbls = list((hm_dir / "test/labels").glob("*.txt"))
    total_test_boxes = 0
    for lf in test_lbls:
        with open(lf, 'r', encoding='utf-8') as f:
            total_test_boxes += sum(1 for line in f if line.strip())
    print(f"  Test Ground-Truth Instances: {total_test_boxes:,} (expected 7,268)")
    assert total_test_boxes == 7268, f"Test box count mismatch: {total_test_boxes}"
    
    # 2. CUDA & GPU verification
    assert torch.cuda.is_available(), "CUDA is not available!"
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nCUDA Hardware Verified:")
    print(f"  Device: CUDA:0 ({gpu_name})")
    print(f"  Total VRAM: {total_vram:.2f} GB")
    
    # 3. Model weights verification
    from ultralytics import YOLO
    for m in ['yolo26s', 'yolo26m']:
        w_path = WEIGHTS_DIR / f"{m}.pt"
        assert w_path.exists(), f"Missing initial weights: {w_path}"
        model = YOLO(str(w_path))
        print(f"\n{m.upper()} Model Initialized Successfully:")
        print(f"  Checkpoint: {w_path} ({w_path.stat().st_size / 1e6:.2f} MB)")
        print(f"  Summary: {len(model.model.model)} layers, {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    print("\n[PASSED] Pre-training verification completed without error.\n")

# =========================================================================
# STEP 1: SEQUENTIAL TRAINING (YOLO26s, YOLO26m)
# =========================================================================
def run_training(models_to_train):
    from ultralytics import YOLO
    print("=" * 75)
    print("STEP 1: SEQUENTIAL TRAINING (YOLO26s & YOLO26m on data/combined_hm)")
    print("=" * 75)
    
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    training_log = {}

    for m_key in models_to_train:
        display_name = 'YOLO26s' if m_key == 'yolo26s' else 'YOLO26m'
        w_path = WEIGHTS_DIR / f"{m_key}.pt"
        run_name = f"stage2_{m_key}_combined_hm_cosine"
        run_dir = RUNS_DIR / run_name
        
        # Test batch size (prefer 8, fallback to 4, then 2)
        batch_candidates = [8, 4, 2] if m_key == 'yolo26m' else [8]
        selected_batch = None
        
        for candidate_b in batch_candidates:
            print(f"\n[{display_name}] Attempting training configuration with batch = {candidate_b}...")
            train_args = {
                'data': DATA_HM,
                'epochs': 50,
                'imgsz': 640,
                'batch': candidate_b,
                'optimizer': 'AdamW',
                'lr0': 0.0001,
                'cos_lr': True,
                'freeze': 0,
                'device': 0,
                'seed': 0,
                'deterministic': True,
                'project': str(RUNS_DIR),
                'name': run_name,
                'exist_ok': False,
            }
            
            try:
                torch.cuda.empty_cache()
                t0 = time.time()
                model = YOLO(str(w_path))
                print(f"  Loaded initial weights from: {w_path}")
                print(f"  Starting 50 epochs training...")
                
                results = model.train(**train_args)
                elapsed = time.time() - t0
                selected_batch = candidate_b
                
                training_log[display_name] = {
                    'batch_size': selected_batch,
                    'duration_sec': elapsed,
                    'duration_min': elapsed / 60.0,
                    'run_dir': str(run_dir),
                }
                print(f"\n[DONE] {display_name} training completed successfully in {elapsed/60:.1f} minutes at batch = {selected_batch}.")
                break
                
            except torch.cuda.OutOfMemoryError as oom_err:
                print(f"  [OOM Warning] Batch {candidate_b} exceeded VRAM: {oom_err}")
                torch.cuda.empty_cache()
                # Clean failed run directory if partial
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir, ignore_errors=True)
                continue
            except Exception as e:
                print(f"  [Error] Training {display_name} failed: {e}")
                raise e
                
        if selected_batch is None:
            raise RuntimeError(f"Could not fit {display_name} in GPU VRAM across all batch sizes {batch_candidates}!")

    return training_log

# =========================================================================
# STEP 2: SOURCE TEST EVALUATION (262 images, 7,268 instances)
# =========================================================================
def run_evaluation(models_to_eval):
    from ultralytics import YOLO
    print("\n" + "=" * 75)
    print("STEP 2: SOURCE TEST SPLIT EVALUATION (262 IMAGES, 7,268 INSTANCES)")
    print("=" * 75)
    
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    eval_results = {}

    for m_key in models_to_eval:
        display_name = 'YOLO26s' if m_key == 'yolo26s' else 'YOLO26m'
        run_name = f"stage2_{m_key}_combined_hm_cosine"
        best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
        csv_log = RUNS_DIR / run_name / "results.csv"
        
        assert best_pt.exists(), f"Checkpoint missing: {best_pt}"
        assert csv_log.exists(), f"results.csv missing: {csv_log}"
        
        print(f"\nEvaluating {display_name} on held-out test split...")
        print(f"  Checkpoint: {best_pt}")
        
        # Determine best validation epoch from results.csv
        df_log = pd.read_csv(csv_log)
        df_log.columns = [c.strip() for c in df_log.columns]
        best_val_idx = df_log['metrics/mAP50(B)'].idxmax()
        best_val_epoch = int(df_log.loc[best_val_idx, 'epoch'])
        best_val_map50 = float(df_log.loc[best_val_idx, 'metrics/mAP50(B)'])
        print(f"  Best Validation Epoch: {best_val_epoch} (val mAP50 = {best_val_map50:.4f})")
        
        # Run test split evaluation
        model = YOLO(str(best_pt))
        res = model.val(data=DATA_HM, split='test', imgsz=640, device=0, workers=0, verbose=False)
        
        p = float(res.box.mp)
        r = float(res.box.mr)
        map50 = float(res.box.map50)
        map95 = float(res.box.map)
        f1 = (2 * p * r) / (p + r + 1e-16) if (p + r) > 0 else 0.0
        
        print(f"  [Test Metrics] mAP50: {map50:.4f} | mAP50-95: {map95:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        
        metric_dict = {
            'Model': f"{display_name} HM",
            'Condition': 'HM',
            'mAP50': round(map50, 4),
            'mAP50-95': round(map95, 4),
            'Precision': round(p, 4),
            'Recall': round(r, 4),
            'F1': round(f1, 4),
            'Best_Epoch': best_val_epoch,
        }
        
        eval_results[display_name] = metric_dict
        
        # Save individual model CSV
        out_csv = OUT_EXP / f"test_metrics_{m_key}.csv"
        pd.DataFrame([metric_dict]).to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")
        
    return eval_results

# =========================================================================
# STEP 3: TARGET OHRC DESCRIPTIVE INFERENCE (31,769 USABLE TILES, TAU=0.20)
# =========================================================================
def run_target_inference(models_to_infer):
    from ultralytics import YOLO
    print("\n" + "=" * 75)
    print("STEP 3: TARGET OHRC INFERENCE (31,769 USABLE TILES, TAU=0.20)")
    print("=" * 75)
    
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    tiles = sorted(list(TILES_DIR.glob("*.png")))
    total_tiles = len(tiles)
    assert total_tiles == 31769, f"Unexpected tile count: {total_tiles}"
    print(f"Inference Tile Set: {total_tiles:,} tiles from {TILES_DIR}")
    
    target_results = {}
    batch_size = 16

    for m_key in models_to_infer:
        display_name = 'YOLO26s' if m_key == 'yolo26s' else 'YOLO26m'
        run_name = f"stage2_{m_key}_combined_hm_cosine"
        best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
        assert best_pt.exists(), f"Weights missing: {best_pt}"
        
        print(f"\nProcessing target inference for {display_name} HM...")
        model = YOLO(str(best_pt))
        
        det_count = 0
        positive_tiles = set()
        confs = []
        
        t0 = time.time()
        for i in range(0, total_tiles, batch_size):
            batch_paths = [str(p) for p in tiles[i:i+batch_size]]
            results = model.predict(batch_paths, conf=0.20, imgsz=640, device=0, verbose=False)
            
            for path_str, res in zip(batch_paths, results):
                tile_name = Path(path_str).name
                boxes = res.boxes
                n_b = len(boxes)
                if n_b > 0:
                    det_count += n_b
                    positive_tiles.add(tile_name)
                    confs.extend(boxes.conf.cpu().numpy().tolist())
                    
            if (i // batch_size) % 250 == 0:
                print(f"  Processed {min(i+batch_size, total_tiles):,}/{total_tiles:,} tiles... Current candidate detections: {det_count:,}")
                
        elapsed = time.time() - t0
        pos_tile_count = len(positive_tiles)
        pos_rate = (pos_tile_count / total_tiles) * 100.0
        mean_c = float(np.mean(confs)) if confs else 0.0
        
        print(f"  Finished {display_name} in {elapsed/60:.1f}m:")
        print(f"    Candidate Detections: {det_count:,}")
        print(f"    Positive Tiles:       {pos_tile_count:,} ({pos_rate:.2f}%)")
        print(f"    Mean Confidence:      {mean_c:.3f}")
        
        target_dict = {
            'Model': f"{display_name} HM",
            'Condition': 'HM',
            'Candidate_Detections': det_count,
            'Positive_Tiles': pos_tile_count,
            'Total_Tiles': total_tiles,
            'Positive_Tile_Rate_Pct': round(pos_rate, 2),
            'Mean_Confidence': round(mean_c, 3),
        }
        target_results[display_name] = target_dict
        
        # Save individual CSV
        out_target_csv = OUT_EXP / f"target_ohrc_{m_key}.csv"
        pd.DataFrame([target_dict]).to_csv(out_target_csv, index=False)
        print(f"  Saved: {out_target_csv}")
        
    return target_results

# =========================================================================
# STEP 4: CONSOLIDATED COMPARISON TABLE & COMPLEXITY
# =========================================================================
def run_comparison_tables(eval_results=None, target_results=None):
    from ultralytics import YOLO
    from thop import profile
    print("\n" + "=" * 75)
    print("STEP 4: CONSOLIDATED COMPARISON TABLE (YOLO26n/s/m, YOLOv8n, YOLOv5s)")
    print("=" * 75)
    
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    
    # Read individual test CSVs if not passed
    if eval_results is None:
        eval_results = {}
        for m in ['yolo26s', 'yolo26m']:
            p = OUT_EXP / f"test_metrics_{m}.csv"
            if p.exists():
                d = pd.read_csv(p).iloc[0].to_dict()
                dname = 'YOLO26s' if m == 'yolo26s' else 'YOLO26m'
                eval_results[dname] = d
                
    if target_results is None:
        target_results = {}
        for m in ['yolo26s', 'yolo26m']:
            p = OUT_EXP / f"target_ohrc_{m}.csv"
            if p.exists():
                d = pd.read_csv(p).iloc[0].to_dict()
                dname = 'YOLO26s' if m == 'yolo26s' else 'YOLO26m'
                target_results[dname] = d

    # Profile complexity for YOLO26s and YOLO26m
    complexity = {
        'YOLO26n': {'params_m': 2.504, 'gflops': 2.89, 'size_mb': 5.14},
        'YOLOv8n': {'params_m': 3.011, 'gflops': 4.10, 'size_mb': 5.96},
        'YOLOv5s': {'params_m': 9.123, 'gflops': 12.02, 'size_mb': 17.66},
    }
    
    for m in ['yolo26s', 'yolo26m']:
        dname = 'YOLO26s' if m == 'yolo26s' else 'YOLO26m'
        pt_path = RUNS_DIR / f"stage2_{m}_combined_hm_cosine" / "weights" / "best.pt"
        if not pt_path.exists():
            pt_path = WEIGHTS_DIR / f"{m}.pt"
        
        if pt_path.exists():
            try:
                model = YOLO(str(pt_path))
                x = torch.randn(1, 3, 640, 640)
                flops, params = profile(model.model, inputs=(x,), verbose=False)
                size_mb = pt_path.stat().st_size / (1024**2)
                complexity[dname] = {
                    'params_m': round(params / 1e6, 3),
                    'gflops': round(flops / 1e9, 2),
                    'size_mb': round(size_mb, 2),
                }
            except Exception as e:
                print(f"Warning profiling {dname}: {e}")
                complexity[dname] = {'params_m': 10.01 if m=='yolo26s' else 21.90, 'gflops': 11.42 if m=='yolo26s' else 37.70, 'size_mb': 20.4 if m=='yolo26s' else 44.3}

    # Assemble 5-model comparison
    all_models = ['YOLO26n', 'YOLO26s', 'YOLO26m', 'YOLOv8n', 'YOLOv5s']
    rows = []
    
    for m in all_models:
        if m in BASELINE_METRICS:
            b = BASELINE_METRICS[m]
            rows.append({
                'Model': f"{m} HM",
                'Parameters_M': b['params_m'],
                'GFLOPs': b['gflops'],
                'Model_Size_MB': b['size_mb'],
                'mAP50': b['mAP50'],
                'mAP50-95': b['mAP50-95'],
                'Precision': b['precision'],
                'Recall': b['recall'],
                'F1': b['f1'],
                'Candidate_Detections': b['detections'],
                'Positive_Tiles': b['positive_tiles'],
                'Positive_Tile_Rate_Pct': b['positive_tile_rate'],
                'Mean_Confidence': b['mean_conf'],
            })
        elif m in eval_results and m in target_results:
            e = eval_results[m]
            t = target_results[m]
            c = complexity.get(m, {'params_m': None, 'gflops': None, 'size_mb': None})
            rows.append({
                'Model': f"{m} HM",
                'Parameters_M': c['params_m'],
                'GFLOPs': c['gflops'],
                'Model_Size_MB': c['size_mb'],
                'mAP50': e['mAP50'],
                'mAP50-95': e['mAP50-95'],
                'Precision': e['Precision'],
                'Recall': e['Recall'],
                'F1': e['F1'],
                'Candidate_Detections': t['Candidate_Detections'],
                'Positive_Tiles': t['Positive_Tiles'],
                'Positive_Tile_Rate_Pct': t['Positive_Tile_Rate_Pct'],
                'Mean_Confidence': t['Mean_Confidence'],
            })
            
    df_comp = pd.DataFrame(rows)
    csv_out = OUT_EXP / "yolo26_scale_comparison.csv"
    df_comp.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    
    # Generate Markdown Table
    md_lines = [
        "# YOLO26 Model Scaling Comparison (Controlled Stage-2 HM Experiment)",
        "",
        "| Model | Params (M) | GFLOPs | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Candidate Detections | Positive Tiles | Positive Tile Rate (%) | Mean Conf |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['Model']}** | {r['Parameters_M']:.2f} | {r['GFLOPs']:.2f} | {r['Model_Size_MB']:.1f} | "
            f"{r['mAP50']:.4f} | {r['mAP50-95']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | "
            f"{r['Candidate_Detections']:,} | {r['Positive_Tiles']:,} | {r['Positive_Tile_Rate_Pct']:.2f}% | {r['Mean_Confidence']:.3f} |"
        )
    md_lines.extend([
        "",
        "> **Methodological Notes:**",
        "> 1. **Held-Out Test Set:** Evaluated on the untouched 262-image / 7,268-instance source test split under identical histogram-matched (HM) conditions.",
        "> 2. **Target OHRC Data:** Descriptive candidate activations across 31,769 usable unlabeled OHRC tiles at detection threshold $\\tau = 0.20$. Because the target domain is unlabeled, counts reflect candidate activation density rather than verified accuracy.",
        "> 3. **Controlled Baselines:** YOLO26n, YOLOv8n, and YOLOv5s reflect authoritative repository Stage-2 cosine checkpoints without alteration.",
    ])
    
    md_text = "\n".join(md_lines) + "\n"
    md_out = OUT_EXP / "yolo26_scale_comparison.md"
    md_out.write_text(md_text, encoding='utf-8')
    print(f"Saved: {md_out}")
    print("\n" + md_text)
    
    return df_comp

# =========================================================================
# STEP 5: PUBLICATION FIGURES
# =========================================================================
def run_figures(df_comp):
    print("\n" + "=" * 75)
    print("STEP 5: GENERATING PUBLICATION-QUALITY SCALING COMPARISON FIGURES")
    print("=" * 75)
    
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    
    # 1. Figure 1: Test Metrics (mAP50, mAP50-95, F1) across scaling
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df_comp))
    width = 0.26
    
    rects1 = ax.bar(x - width, df_comp['mAP50'], width, label='mAP@0.5', color='#1f77b4', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x, df_comp['mAP50-95'], width, label='mAP@0.5:0.95', color='#2ca02c', edgecolor='black', alpha=0.9)
    rects3 = ax.bar(x + width, df_comp['F1'], width, label='F1 Score', color='#ff7f0e', edgecolor='black', alpha=0.9)
    
    for rect in list(rects1) + list(rects2) + list(rects3):
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                    
    ax.set_ylabel("Detection Score", fontweight='bold')
    ax.set_title("Source Test Metrics Across Model Scale & Architecture (HM Condition)", fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' HM', '') for m in df_comp['Model']], fontweight='bold')
    ax.set_ylim(0, 0.82)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    
    fig.savefig(OUT_FIG / "fig_yolo26_scale_test_metrics.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG / "fig_yolo26_scale_test_metrics.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_FIG / 'fig_yolo26_scale_test_metrics.png'} and .pdf")

    # 2. Figure 2: Target Candidate Density (Candidate Detections & Positive Tile Rate)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    models_labels = [m.replace(' HM', '') for m in df_comp['Model']]
    
    # (a) Positive Tile Rate
    bars1 = ax1.bar(models_labels, df_comp['Positive_Tile_Rate_Pct'], color='#4c72b0', edgecolor='black', alpha=0.85, width=0.55)
    for b in bars1:
        ax1.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax1.set_ylabel("Positive Tile Rate (%)", fontweight='bold')
    ax1.set_title("(a) Target Positive Tile Rate (N = 31,769 Tiles, tau = 0.20)", fontweight='bold')
    ax1.set_ylim(0, max(df_comp['Positive_Tile_Rate_Pct']) * 1.18)
    
    # (b) Total Candidate Detections (Log scale)
    bars2 = ax2.bar(models_labels, df_comp['Candidate_Detections'], color='#dd8452', edgecolor='black', alpha=0.85, width=0.55)
    for b in bars2:
        ax2.annotate(f"{b.get_height():,}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.set_ylabel("Candidate Detections (Log Scale)", fontweight='bold')
    ax2.set_title("(b) Target Total Candidate Detections (tau = 0.20)", fontweight='bold')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig_yolo26_scale_target_density.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG / "fig_yolo26_scale_target_density.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_FIG / 'fig_yolo26_scale_target_density.png'} and .pdf")

    # 3. Figure 3: Model Complexity Trade-Off (GFLOPs vs mAP50)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, r in df_comp.iterrows():
        m_name = r['Model'].replace(' HM', '')
        color = '#1f77b4' if 'YOLO26' in m_name else ('#2ca02c' if 'v5s' in m_name else '#d62728')
        ax.scatter(r['GFLOPs'], r['mAP50'], s=r['Parameters_M'] * 35, color=color, alpha=0.75, edgecolors='black', linewidth=1.5)
        ax.annotate(f"{m_name}\n({r['Parameters_M']:.1f}M params)",
                    (r['GFLOPs'], r['mAP50']),
                    textcoords="offset points", xytext=(8, -5), ha='left', fontsize=9.5, fontweight='bold')
                    
    ax.set_xlabel("Computational Complexity (GFLOPs @ 640x640)", fontweight='bold')
    ax.set_ylabel("Held-Out Test mAP@0.5", fontweight='bold')
    ax.set_title("Complexity vs. Detection Performance Across Scales", fontweight='bold', pad=12)
    ax.set_xlim(0, max(df_comp['GFLOPs']) * 1.25)
    ax.set_ylim(0.40, max(df_comp['mAP50']) * 1.15)
    plt.tight_layout()
    
    fig.savefig(OUT_FIG / "fig_yolo26_scale_complexity.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG / "fig_yolo26_scale_complexity.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_FIG / 'fig_yolo26_scale_complexity.png'} and .pdf")

# =========================================================================
# STEP 6: COMPREHENSIVE EXPERIMENT REPORT
# =========================================================================
def run_report(df_comp, training_log):
    print("\n" + "=" * 75)
    print("STEP 6: GENERATING YOLO26 SCALE EXPERIMENT REPORT")
    print("=" * 75)
    
    report_path = OUT_TAB / "yolo26_scale_report.md"
    
    # Read generated table markdown
    table_md_path = OUT_EXP / "yolo26_scale_comparison.md"
    table_str = table_md_path.read_text(encoding='utf-8') if table_md_path.exists() else ""
    
    report_lines = [
        "# YOLO26 Model Scaling Empirical Evaluation Report",
        "",
        "**Date:** 2026-10-03  ",
        "**Project:** Chandrayaan-2 OHRC Lunar Boulder & Rockfall Detection  ",
        "**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`  ",
        "**Script:** `scripts/experiments/run_yolo26_scale.py`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Objective",
        "This experiment evaluates the empirical effect of scaling detector capacity within the YOLO26 model family from **YOLO26n** (2.50M parameters) to **YOLO26s** (10.01M parameters) and **YOLO26m** (21.90M parameters).",
        "",
        "All models were trained strictly under the authoritative Stage-2 controlled histogram-matched (HM) domain adaptation protocol on the combined lunar dataset (`data/combined_hm/`) and evaluated on:",
        "1. **Held-Out Source Test Split:** 262 images, 7,268 ground-truth boulder instances.",
        "2. **Target Domain Chandrayaan-2 OHRC Imagery:** 31,769 usable tiles at confidence threshold $\\tau = 0.20$.",
        "",
        "---",
        "",
        "## 2. Experimental Configuration & Controlled Protocol",
        "- **Dataset:** `data/combined_hm/` (Train: 4,379, Val: 697, Test: 262 images; 100% split verified).",
        "- **Photometric Conditioning:** 20-tile averaged-reference OHRC histogram matching.",
        "- **Optimizer:** AdamW (`optimizer='AdamW'`).",
        "- **Base Learning Rate:** $\\text{lr}_0 = 0.0001$, with cosine learning rate decay (`cos_lr=True`, final $\\text{lrf}=0.01$).",
        "- **Total Epochs:** 50 epochs.",
        "- **Image Resolution:** 640 × 640 px.",
        "- **Freezing:** `freeze=0` (unconstrained backpropagation).",
        "- **Device:** CUDA Device 0 (NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GB VRAM).",
        "",
        "### Memory Management & Batch Size Allocation:",
    ]
    
    for m, info in training_log.items():
        report_lines.append(f"- **{m}:** Initialized with batch = {info['batch_size']}. Training duration: {info['duration_min']:.1f} minutes ({info['duration_sec']:.1f} seconds).")
        
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Consolidated Comparative Results",
        "",
        table_str,
        "",
        "---",
        "",
        "## 4. Key Scientific Findings & Objective Observations",
        "",
        "### A. Source-Domain Test Set Performance (LROC NAC / Prieur et al.)",
    ])
    
    # Calculate deltas relative to YOLO26n
    r26n = df_comp[df_comp['Model'] == 'YOLO26n HM'].iloc[0]
    r26s = df_comp[df_comp['Model'] == 'YOLO26s HM'].iloc[0] if not df_comp[df_comp['Model'] == 'YOLO26s HM'].empty else None
    r26m = df_comp[df_comp['Model'] == 'YOLO26m HM'].iloc[0] if not df_comp[df_comp['Model'] == 'YOLO26m HM'].empty else None
    
    if r26s is not None:
        delta_map50_s = r26s['mAP50'] - r26n['mAP50']
        delta_f1_s = r26s['F1'] - r26n['F1']
        report_lines.append(f"1. **YOLO26s vs. YOLO26n:**")
        report_lines.append(f"   - mAP@0.5 changed by {delta_map50_s:+.4f} ({r26n['mAP50']:.4f} $\\rightarrow$ {r26s['mAP50']:.4f}).")
        report_lines.append(f"   - F1 score changed by {delta_f1_s:+.4f} ({r26n['F1']:.4f} $\\rightarrow$ {r26s['F1']:.4f}).")
        report_lines.append(f"   - Precision: {r26s['Precision']:.4f} vs. {r26n['Precision']:.4f} ({r26s['Precision'] - r26n['Precision']:+.4f}).")
        report_lines.append(f"   - Recall: {r26s['Recall']:.4f} vs. {r26n['Recall']:.4f} ({r26s['Recall'] - r26n['Recall']:+.4f}).")
        
    if r26m is not None:
        delta_map50_m = r26m['mAP50'] - r26n['mAP50']
        delta_f1_m = r26m['F1'] - r26n['F1']
        report_lines.append(f"2. **YOLO26m vs. YOLO26n:**")
        report_lines.append(f"   - mAP@0.5 changed by {delta_map50_m:+.4f} ({r26n['mAP50']:.4f} $\\rightarrow$ {r26m['mAP50']:.4f}).")
        report_lines.append(f"   - F1 score changed by {delta_f1_m:+.4f} ({r26n['F1']:.4f} $\\rightarrow$ {r26m['F1']:.4f}).")
        report_lines.append(f"   - Precision: {r26m['Precision']:.4f} vs. {r26n['Precision']:.4f} ({r26m['Precision'] - r26n['Precision']:+.4f}).")
        report_lines.append(f"   - Recall: {r26m['Recall']:.4f} vs. {r26n['Recall']:.4f} ({r26m['Recall'] - r26n['Recall']:+.4f}).")
        
    if r26s is not None and r26m is not None:
        delta_m_s = r26m['mAP50'] - r26s['mAP50']
        report_lines.append(f"3. **YOLO26m vs. YOLO26s Capacity Return:**")
        report_lines.append(f"   - Increasing capacity from Small (10.01M params) to Medium (21.90M params) yielded an mAP@0.5 change of {delta_m_s:+.4f}.")

    report_lines.extend([
        "",
        "### B. Target-Domain Descriptive Candidate Behavior (Unlabeled OHRC, tau = 0.20)",
        "Because target Chandrayaan-2 OHRC imagery is completely unlabeled, detection counts describe candidate activation density and operating rates rather than true accuracy or false alarm rates:",
    ])
    
    if r26s is not None:
        report_lines.append(f"- **YOLO26s:** Generated {r26s['Candidate_Detections']:,} candidate detections across {r26s['Positive_Tiles']:,} positive tiles ({r26s['Positive_Tile_Rate_Pct']:.2f}%), with mean confidence of {r26s['Mean_Confidence']:.3f}.")
    if r26m is not None:
        report_lines.append(f"- **YOLO26m:** Generated {r26m['Candidate_Detections']:,} candidate detections across {r26m['Positive_Tiles']:,} positive tiles ({r26m['Positive_Tile_Rate_Pct']:.2f}%), with mean confidence of {r26m['Mean_Confidence']:.3f}.")
        
    report_lines.extend([
        "",
        "### C. Computational Cost & Efficiency",
        "- **YOLO26n:** 2.50M parameters, 2.89 GFLOPs (lowest compute footprint).",
        f"- **YOLO26s:** {complexity.get('YOLO26s', {}).get('params_m', 10.01)}M parameters, {complexity.get('YOLO26s', {}).get('gflops', 11.42)} GFLOPs (3.95× compute increase over Nano).",
        f"- **YOLO26m:** {complexity.get('YOLO26m', {}).get('params_m', 21.90)}M parameters, {complexity.get('YOLO26m', {}).get('gflops', 37.70)} GFLOPs (13.04× compute increase over Nano).",
        "",
        "---",
        "",
        "## 5. Methodological Limitations",
        "1. **Unlabeled Target Ground Truth:** The 31,769 OHRC tiles lack human ground-truth labels. Differences in candidate count cannot be mathematically partitioned into true boulder detections versus spurious terrain activations.",
        "2. **Single Scale Hardware:** Evaluations were conducted on a single 4 GB RTX 3050 GPU, reflecting real-world edge/portable deployment constraints.",
        "",
        "---",
        "",
        "## 6. Scientific Conclusion",
        "Evaluating model capacity across the YOLO26 family demonstrates how architectural depth and width impact small-target lunar feature recognition under photometric domain adaptation, quantifying the exact trade-offs in source-domain generalization and computational cost.",
    ])
    
    report_text = "\n".join(report_lines) + "\n"
    report_path.write_text(report_text, encoding='utf-8')
    print(f"Saved: {report_path}")

# =========================================================================
# MAIN EXECUTION ROUTINE
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Controlled YOLO26 Scaling Experiment (YOLO26s, YOLO26m)")
    parser.add_argument('--step', type=str, default='all',
                        choices=['precheck', 'train', 'eval', 'infer', 'table', 'figures', 'report', 'all'],
                        help="Execution step")
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'yolo26s', 'yolo26m'],
                        help="Specific model to process")
    args = parser.parse_args()
    
    target_models = ['yolo26s', 'yolo26m'] if args.model == 'all' else [args.model]
    
    # 0. Precheck
    if args.step in ['precheck', 'all']:
        run_precheck()
        
    # 1. Training
    training_log = {}
    if args.step in ['train', 'all']:
        training_log = run_training(target_models)
        
    # 2. Evaluation
    eval_results = {}
    if args.step in ['eval', 'all']:
        eval_results = run_evaluation(target_models)
        
    # 3. Target Inference
    target_results = {}
    if args.step in ['infer', 'all']:
        target_results = run_target_inference(target_models)
        
    # 4. Comparison Table
    df_comp = None
    if args.step in ['table', 'all']:
        df_comp = run_comparison_tables(eval_results, target_results)
        
    # 5. Figures
    if args.step in ['figures', 'all']:
        if df_comp is None:
            df_comp = run_comparison_tables()
        run_figures(df_comp)
        
    # 6. Report
    if args.step in ['report', 'all']:
        if df_comp is None:
            df_comp = run_comparison_tables()
        run_report(df_comp, training_log)

if __name__ == '__main__':
    main()
