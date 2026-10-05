"""Task I: Regional Detection Rates Comparison between South Pole and Equatorial/Northern regions.
Includes grouped bar chart and CSV table.
Scientific caveat: Differences reflect observation geometry, incidence angles, and tile sampling rather than intrinsic geological density.
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

# Authoritative regional summary data
data = [
    {
        'Region': 'South Pole',
        'Model': 'YOLO26n',
        'Detections': 3982,
        'Positive_Tiles': 1176,
        'Total_Tiles': 16308,
        'Positive_Tile_Rate_Pct': 7.21,
    },
    {
        'Region': 'South Pole',
        'Model': 'YOLO26m',
        'Detections': 93,
        'Positive_Tiles': 50,
        'Total_Tiles': 16308,
        'Positive_Tile_Rate_Pct': 0.31,
    },
    {
        'Region': 'South Pole',
        'Model': 'YOLOv8n',
        'Detections': 292371,
        'Positive_Tiles': 12689,
        'Total_Tiles': 16308,
        'Positive_Tile_Rate_Pct': 77.81,
    },
    {
        'Region': 'South Pole',
        'Model': 'YOLOv5s',
        'Detections': 3698,
        'Positive_Tiles': 2234,
        'Total_Tiles': 16308,
        'Positive_Tile_Rate_Pct': 13.70,
    },
    {
        'Region': 'Equatorial / Northern',
        'Model': 'YOLO26n',
        'Detections': 1102,
        'Positive_Tiles': 958,
        'Total_Tiles': 15461,
        'Positive_Tile_Rate_Pct': 6.20,
    },
    {
        'Region': 'Equatorial / Northern',
        'Model': 'YOLO26m',
        'Detections': 1391,
        'Positive_Tiles': 393,
        'Total_Tiles': 15461,
        'Positive_Tile_Rate_Pct': 2.54,
    },
    {
        'Region': 'Equatorial / Northern',
        'Model': 'YOLOv8n',
        'Detections': 61056,
        'Positive_Tiles': 1440,
        'Total_Tiles': 15461,
        'Positive_Tile_Rate_Pct': 9.31,
    },
    {
        'Region': 'Equatorial / Northern',
        'Model': 'YOLOv5s',
        'Detections': 20754,
        'Positive_Tiles': 1662,
        'Total_Tiles': 15461,
        'Positive_Tile_Rate_Pct': 10.75,
    },
]

df = pd.DataFrame(data)

# Save CSV table
csv_path = OUT_TAB / "regional_detection_rates.csv"
df.to_csv(csv_path, index=False)
print(f"[Done] Saved table: {csv_path}")

# Plotting
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

regions = ['South Pole', 'Equatorial / Northern']
models = ['YOLO26n', 'YOLO26m', 'YOLOv8n', 'YOLOv5s']
x = np.arange(len(regions))
width = 0.18

colors = {
    'YOLO26n': '#1f77b4',
    'YOLO26m': '#9467bd',
    'YOLOv8n': '#d95f02',
    'YOLOv5s': '#2ca02c',
}

# Left plot: Positive Tile Rate (%)
for i, m in enumerate(models):
    sub = df[df['Model'] == m]
    rates = [sub[sub['Region'] == r]['Positive_Tile_Rate_Pct'].values[0] for r in regions]
    offset = (i - 1.5) * width
    rects = ax1.bar(x + offset, rates, width, label=m, color=colors[m], alpha=0.85, edgecolor='black', lw=0.6)
    for rect in rects:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_ylabel("Positive Tile Rate (%)", fontweight='bold')
ax1.set_title("(a) Tile Detection Frequency by Region", fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(regions, fontweight='bold')
ax1.set_ylim(0, 90)
ax1.legend(title="Model", frameon=True, facecolor='white', framealpha=0.9)

# Right plot: Total Candidate Detections (Log scale)
for i, m in enumerate(models):
    sub = df[df['Model'] == m]
    counts = [sub[sub['Region'] == r]['Detections'].values[0] for r in regions]
    offset = (i - 1.5) * width
    rects = ax2.bar(x + offset, counts, width, label=m, color=colors[m], alpha=0.85, edgecolor='black', lw=0.6)
    for rect in rects:
        h = rect.get_height()
        ax2.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_ylabel("Total Candidate Detections (Log Scale)", fontweight='bold')
ax2.set_title("(b) Candidate Count by Region", fontweight='bold', pad=10)
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(regions, fontweight='bold')
ax2.set_ylim(500, 1000000)
ax2.legend(title="Model", frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
png_path = OUT_FIG / "fig_regional_detection_rates.png"
pdf_path = OUT_FIG / "fig_regional_detection_rates.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[Done] Saved figure: {png_path} and {pdf_path}")
