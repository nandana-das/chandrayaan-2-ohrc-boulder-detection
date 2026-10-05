"""Task H: OHRC Confidence Distributions for YOLO26n, YOLOv8n, and YOLOv5s.
Operational threshold tau = 0.20 marked, plus consensus-proxy operating points.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
RESULTS = BASE / "results"
OUT_FIG = RESULTS / "paper_figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Load detection CSVs
files = {
    'YOLO26n': RESULTS / 'ohrc_inference/raw_detections_yolo26n.csv',
    'YOLO26s': RESULTS / 'ohrc_inference/raw_detections_yolo26s.csv',
    'YOLO26m': RESULTS / 'ohrc_inference/raw_detections_yolo26m.csv',
    'YOLOv8n': RESULTS / 'ohrc_inference/raw_detections_yolov8n.csv',
    'YOLOv5s': RESULTS / 'ohrc_inference/raw_detections_yolov5s.csv',
}

data = {}
for name, fpath in files.items():
    df = pd.read_csv(fpath)
    data[name] = df['conf'].values

colors = {
    'YOLO26n': '#1f77b4',  # steel blue
    'YOLO26s': '#17becf',  # cyan / teal
    'YOLO26m': '#9467bd',  # purple
    'YOLOv8n': '#d95f02',  # burnt orange
    'YOLOv5s': '#2ca02c',  # forest green
}

# Proxy operating points from consensus analysis (NOT ground truth)
proxy_points = {
    'YOLO26n': {'tau': 0.20, 'f1': 0.1347, 'p': 0.2909, 'r': 0.0877},
    'YOLO26s': {'tau': 0.20, 'f1': 0.0000, 'p': 0.0000, 'r': 0.0000},
    'YOLO26m': {'tau': 0.20, 'f1': 0.0000, 'p': 0.0000, 'r': 0.0000},
    'YOLOv8n': {'tau': 0.46, 'f1': 0.2201, 'p': 0.1749, 'r': 0.2970},
    'YOLOv5s': {'tau': 0.20, 'f1': 0.7473, 'p': 0.6314, 'r': 0.9153},
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.4,
})

fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

bins = np.linspace(0.20, 1.0, 41)  # 0.02 step from 0.20 to 1.00

for i, (name, confs) in enumerate(data.items()):
    ax = axes[i]
    c = colors[name]
    
    # Normalized density / percentage of detections
    weights = np.ones_like(confs) / len(confs) * 100.0
    n, b, patches = ax.hist(confs, bins=bins, weights=weights, color=c, alpha=0.75, edgecolor='black', lw=0.6)
    
    # Mean and median lines
    mean_val = np.mean(confs)
    median_val = np.median(confs)
    
    ax.axvline(mean_val, color='darkblue', linestyle='--', lw=1.8, 
               label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='purple', linestyle=':', lw=1.8, 
               label=f'Median: {median_val:.3f}')
    
    # Operational threshold tau = 0.20
    ax.axvline(0.20, color='red', linestyle='-', lw=2.0, 
               label=r'Operational $\tau = 0.20$')
    
    # Proxy threshold marker if distinct from 0.20
    tau_proxy = proxy_points[name]['tau']
    if abs(tau_proxy - 0.20) > 0.01:
        ax.axvline(tau_proxy, color='#7570b3', linestyle='-.', lw=2.0,
                   label=rf'Consensus Proxy $\tau^* = {tau_proxy:.2f}$')
    
    # Annotations
    det_count = len(confs)
    ax.set_title(f"{name} (N = {det_count:,})", fontweight='bold', pad=10)
    ax.set_xlabel("Confidence Score", fontweight='bold')
    if i == 0:
        ax.set_ylabel("Detection Frequency (% of total)", fontweight='bold')
    
    ax.set_xlim(0.18, 1.0)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
png_path = OUT_FIG / "fig_ohrc_confidence_distributions.png"
pdf_path = OUT_FIG / "fig_ohrc_confidence_distributions.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[Done] Saved: {png_path} and {pdf_path}")
