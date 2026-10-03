import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
RUNS = BASE / "runs"
OUT_FIG = BASE / "results/paper_figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

csv_files = {
    'YOLO26n': RUNS / 'stage2_yolo26n_combined_hm_cosine' / 'results.csv',
    'YOLOv8n': RUNS / 'stage2_yolov8n_combined_hm_cosine' / 'results.csv',
    'YOLOv5s': RUNS / 'stage2_yolov5s_combined_hm_cosine' / 'results.csv',
}

# Clean column names (strip whitespace)
dfs = {}
for name, path in csv_files.items():
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    dfs[name] = df

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.5,
})

colors = {
    'YOLO26n': '#1f77b4',  # blue
    'YOLOv8n': '#ff7f0e',  # orange
    'YOLOv5s': '#2ca02c',  # green
}

# 3x3 Plot: Columns = Models, Rows = Metrics
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

models = ['YOLO26n', 'YOLOv8n', 'YOLOv5s']

for col_idx, model in enumerate(models):
    df = dfs[model]
    epochs = df['epoch']
    c = colors[model]
    
    # Row 0: Validation mAP@0.5
    ax0 = axes[0, col_idx]
    ax0.plot(epochs, df['metrics/mAP50(B)'], color=c, label=f'{model} Val mAP50')
    ax0.set_title(f"{model}", fontweight='bold', pad=8)
    if col_idx == 0:
        ax0.set_ylabel("Validation mAP@0.5", fontweight='bold')
    ax0.set_ylim(0, 0.75)
    ax0.legend(loc='lower right', frameon=True)
    
    # Best mAP50 annotation
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    best_val = df.loc[best_idx, 'metrics/mAP50(B)']
    ax0.scatter([best_epoch], [best_val], color='red', s=40, zorder=5)
    ax0.annotate(f"Peak: {best_val:.3f}\n(ep {int(best_epoch)})", 
                 (best_epoch, best_val), 
                 textcoords="offset points", xytext=(-25, -28), 
                 fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.2))

    # Row 1: Validation mAP@0.5:0.95
    ax1 = axes[1, col_idx]
    ax1.plot(epochs, df['metrics/mAP50-95(B)'], color=c, linestyle='--', label=f'{model} Val mAP50-95')
    if col_idx == 0:
        ax1.set_ylabel("Val mAP@0.5:0.95", fontweight='bold')
    ax1.set_ylim(0, 0.35)
    ax1.legend(loc='lower right', frameon=True)
    
    best_idx95 = df['metrics/mAP50-95(B)'].idxmax()
    best_epoch95 = df.loc[best_idx95, 'epoch']
    best_val95 = df.loc[best_idx95, 'metrics/mAP50-95(B)']
    ax1.scatter([best_epoch95], [best_val95], color='red', s=40, zorder=5)
    ax1.annotate(f"Peak: {best_val95:.3f}\n(ep {int(best_epoch95)})", 
                 (best_epoch95, best_val95), 
                 textcoords="offset points", xytext=(-25, -28), 
                 fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.2))

    # Row 2: Box Loss (Train & Val)
    ax2 = axes[2, col_idx]
    ax2.plot(epochs, df['train/box_loss'], color=c, label='Train Box Loss', alpha=0.9)
    ax2.plot(epochs, df['val/box_loss'], color=c, linestyle=':', label='Val Box Loss', alpha=0.9, lw=2.2)
    if col_idx == 0:
        ax2.set_ylabel("Bounding Box Loss", fontweight='bold')
    ax2.set_xlabel("Epoch", fontweight='bold')
    ax2.set_xlim(1, 50)
    ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
png_path = OUT_FIG / "fig_training_curves.png"
pdf_path = OUT_FIG / "fig_training_curves.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[Done] Saved: {png_path} and {pdf_path}")
