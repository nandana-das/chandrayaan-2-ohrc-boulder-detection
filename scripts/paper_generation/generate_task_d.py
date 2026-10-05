"""Task D: Held-out Test Precision-Recall (PR) Curves.
Evaluates the three Stage-2 cosine checkpoints on the untouched 262-image Prieur test set.
Saves:
- results/paper_figures/fig_test_pr_curves.png
- results/paper_figures/fig_test_pr_curves.pdf
- results/paper_tables/test_pr_curve_data.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA = str(BASE / "data/combined_hm/dataset_combined_hm.yaml")
RUNS = BASE / "runs"
RESULTS = BASE / "results"
OUT_FIG = RESULTS / "paper_figures"
OUT_TAB = RESULTS / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

MODELS = {
    'YOLO26n': RUNS / 'stage2_yolo26n_combined_hm_cosine/weights/best.pt',
    'YOLO26m': RUNS / 'stage2_yolo26m_combined_hm_cosine/weights/best.pt',
    'YOLOv8n': RUNS / 'stage2_yolov8n_combined_hm_cosine/weights/best.pt',
    'YOLOv5s': RUNS / 'stage2_yolov5s_combined_hm_cosine/weights/best.pt',
}

colors = {
    'YOLO26n': '#1f77b4',
    'YOLO26m': '#9467bd',
    'YOLOv8n': '#d95f02',
    'YOLOv5s': '#2ca02c',
}

def main():
    print("=" * 60)
    print("Task D: Evaluating Held-Out Test PR Curves")
    print("=" * 60)
    
    pr_data = {}
    metrics_summary = {}
    
    for name, weights_path in MODELS.items():
        print(f"\nEvaluating {name} ({weights_path.name})...")
        model = YOLO(str(weights_path))
        val_res = model.val(
            data=DATA,
            split='test',
            imgsz=640,
            device=0,
            workers=0,
            verbose=False
        )
        
        # In Ultralytics, curves_results[0] is Precision-Recall curve
        # item[0] = recall (1000,), item[1] = precision array (nc, 1000)
        cr = val_res.box.curves_results[0]
        recall = np.array(cr[0])
        precision = np.array(cr[1][0]) # class 0 (boulder)
        
        map50 = float(val_res.box.map50)
        map50_95 = float(val_res.box.map)
        mp = float(val_res.box.mp)
        mr = float(val_res.box.mr)
        
        pr_data[name] = {
            'recall': recall,
            'precision': precision,
            'map50': map50,
            'map50_95': map50_95,
            'mp': mp,
            'mr': mr,
        }
        metrics_summary[name] = {'map50': map50, 'map50_95': map50_95, 'mp': mp, 'mr': mr}
        print(f"  {name}: mAP@0.5 = {map50:.4f}, mAP@0.5:0.95 = {map50_95:.4f}, P = {mp:.4f}, R = {mr:.4f}")

    # Build CSV of raw PR curves (interpolated over 100 evenly spaced recall points for clean tabular representation)
    eval_recalls = np.linspace(0.01, 1.0, 100)
    csv_rows = []
    for r in eval_recalls:
        row = {'Recall': round(r, 4)}
        for name in MODELS:
            rec = pr_data[name]['recall']
            prec = pr_data[name]['precision']
            # Find precision at this recall
            interp_p = np.interp(r, rec, prec)
            row[f'{name}_Precision'] = round(float(interp_p), 4)
        csv_rows.append(row)
        
    df_pr = pd.DataFrame(csv_rows)
    csv_path = OUT_TAB / "test_pr_curve_data.csv"
    df_pr.to_csv(csv_path, index=False)
    print(f"\n[Done] Saved curve data: {csv_path}")

    # Plotting Publication-Quality PR Curves
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10.5,
        'lines.linewidth': 2.2,
        'grid.alpha': 0.5,
    })

    fig, ax = plt.subplots(figsize=(8, 6.5))

    for name in MODELS:
        rec = pr_data[name]['recall']
        prec = pr_data[name]['precision']
        map50 = pr_data[name]['map50']
        c = colors[name]
        
        ax.plot(rec, prec, color=c, lw=2.4, 
                label=f"{name} (mAP@0.5 = {map50:.4f})")
        
        # Mark operating point (overall test P, R)
        op_p = pr_data[name]['mp']
        op_r = pr_data[name]['mr']
        ax.scatter([op_r], [op_p], color=c, s=55, zorder=5, edgecolor='black', lw=0.8)
        ax.annotate(f"{name} (P={op_p:.2f}, R={op_r:.2f})", 
                    (op_r, op_p),
                    xytext=(-35, 12 if name != 'YOLO26n' else -18), 
                    textcoords="offset points", 
                    fontsize=8.5, fontweight='bold', color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.0))

    ax.set_xlabel("Recall", fontweight='bold')
    ax.set_ylabel("Precision", fontweight='bold')
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Held-Out Source Test Set Precision-Recall Curves (Prieur, 262 images)", 
                 fontweight='bold', pad=12)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, shadow=False)

    plt.tight_layout()
    png_path = OUT_FIG / "fig_test_pr_curves.png"
    pdf_path = OUT_FIG / "fig_test_pr_curves.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] Saved PR curve figure: {png_path} and {pdf_path}")

if __name__ == '__main__':
    main()
