"""Task J: Ultra-Deep Polar Analysis on 20 calibrated Chandrayaan-2 OHRC products (13,906 usable tiles).
Compares current Stage-2 models against the historical RT-DETR-L baseline (clearly labeled as historical baseline).
Saves:
- results/paper_figures/fig_ultradeep_polar.png
- results/paper_figures/fig_ultradeep_polar.pdf
- results/paper_tables/ultradeep_polar_summary.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
RESULTS = BASE / "results"
OUT_FIG = RESULTS / "paper_figures"
OUT_TAB = RESULTS / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# Data dictionary from authoritative prompt
records = [
    {
        'Model': 'YOLO26n',
        'Model_Status': 'Stage-2 Cosine (Current)',
        'Detections': 10085,
        'Positive_Tiles': 1173,
        'Total_Tiles': 13906,
        'Positive_Tile_Rate_Pct': 8.44,
        'Mean_Confidence': 0.2907,
    },
    {
        'Model': 'YOLO26m',
        'Model_Status': 'Stage-2 Cosine (Current)',
        'Detections': 1447,
        'Positive_Tiles': 309,
        'Total_Tiles': 13906,
        'Positive_Tile_Rate_Pct': 2.22,
        'Mean_Confidence': 0.3057,
    },
    {
        'Model': 'YOLOv8n',
        'Model_Status': 'Stage-2 Cosine (Current)',
        'Detections': 71001,
        'Positive_Tiles': 6047,
        'Total_Tiles': 13906,
        'Positive_Tile_Rate_Pct': 43.48,
        'Mean_Confidence': 0.2959,
    },
    {
        'Model': 'YOLOv5s',
        'Model_Status': 'Stage-2 Cosine (Current)',
        'Detections': 7202,
        'Positive_Tiles': 1779,
        'Total_Tiles': 13906,
        'Positive_Tile_Rate_Pct': 12.79,
        'Mean_Confidence': 0.3214,
    },
    {
        'Model': 'RT-DETR-L',
        'Model_Status': 'Historical Baseline (Pre-Stage 2)',
        'Detections': 532844,
        'Positive_Tiles': 11664,
        'Total_Tiles': 13906,
        'Positive_Tile_Rate_Pct': 83.88,
        'Mean_Confidence': 0.3258,
    },
]

df = pd.DataFrame(records)

# Save CSV table
csv_path = OUT_TAB / "ultradeep_polar_summary.csv"
df.to_csv(csv_path, index=False)
print(f"[Done] Saved table: {csv_path}")

# Plotting publication figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.alpha': 0.5,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

models = ['YOLO26n', 'YOLO26m', 'YOLOv8n', 'YOLOv5s', 'RT-DETR-L\n(Hist. Baseline)']
model_keys = ['YOLO26n', 'YOLO26m', 'YOLOv8n', 'YOLOv5s', 'RT-DETR-L']

bar_colors = [
    '#1f77b4',  # blue
    '#9467bd',  # purple
    '#d95f02',  # orange
    '#2ca02c',  # green
    '#7570b3',  # slate purple for historical baseline
]

# (a) Positive Tile Rate (%)
rates = df['Positive_Tile_Rate_Pct'].values
rects1 = ax1.bar(models, rates, color=bar_colors, width=0.55, edgecolor='black', alpha=0.85, lw=0.6)
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax1.set_ylabel("Positive Tile Rate (%)", fontweight='bold')
ax1.set_title("(a) Ultra-Deep Polar Positive Tile Rate (N = 13,906 tiles)", fontweight='bold', pad=10)
ax1.set_ylim(0, 100)

# (b) Total Candidate Detections (Log scale)
counts = df['Detections'].values
rects2 = ax2.bar(models, counts, color=bar_colors, width=0.55, edgecolor='black', alpha=0.85, lw=0.6)
for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax2.set_ylabel("Total Candidate Detections (Log Scale)", fontweight='bold')
ax2.set_title("(b) Ultra-Deep Polar Detection Counts (Log Scale)", fontweight='bold', pad=10)
ax2.set_yscale('log')
ax2.set_ylim(1000, 1500000)

plt.tight_layout()
png_path = OUT_FIG / "fig_ultradeep_polar.png"
pdf_path = OUT_FIG / "fig_ultradeep_polar.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[Done] Saved figure: {png_path} and {pdf_path}")
