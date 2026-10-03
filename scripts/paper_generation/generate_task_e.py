"""Task E: Source Object-Size Analysis across training, validation, and test splits.
Analyzes bounding box width, height, area, and normalized area.
Calculates summary statistics: mean, median, std, min, max, 25th, 75th percentiles.
Saves:
- results/paper_figures/fig_object_size_distribution.png
- results/paper_figures/fig_object_size_distribution.pdf
- results/paper_tables/object_size_statistics.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA_DIR = BASE / "data/combined_hm"
RESULTS = BASE / "results"
OUT_FIG = RESULTS / "paper_figures"
OUT_TAB = RESULTS / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

splits = {
    'Train (4,379 images)': DATA_DIR / 'train/labels',
    'Val (697 images)': DATA_DIR / 'val/labels',
    'Test (262 images)': DATA_DIR / 'test/labels',
}

# Image dimension is 640x640 in standard YOLO format
IMG_W, IMG_H = 640, 640

def parse_labels(label_dir):
    widths, heights, areas, norm_areas = [], [], [], []
    for f in label_dir.glob("*.txt"):
        with open(f, 'r') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class x_c y_c w h (normalized 0 to 1)
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    w_px = w_norm * IMG_W
                    h_px = h_norm * IMG_H
                    area_px = w_px * h_px
                    norm_area = w_norm * h_norm
                    
                    widths.append(w_px)
                    heights.append(h_px)
                    areas.append(area_px)
                    norm_areas.append(norm_area)
    return {
        'width_px': np.array(widths),
        'height_px': np.array(heights),
        'area_px': np.array(areas),
        'norm_area': np.array(norm_areas),
    }

def calc_stats(arr, name, unit):
    return {
        'Metric': name,
        'Unit': unit,
        'Count': len(arr),
        'Mean': round(float(np.mean(arr)), 4),
        'Std': round(float(np.std(arr)), 4),
        'Median': round(float(np.median(arr)), 4),
        'Min': round(float(np.min(arr)), 4),
        'Max': round(float(np.max(arr)), 4),
        'P25 (25th)': round(float(np.percentile(arr, 25)), 4),
        'P75 (75th)': round(float(np.percentile(arr, 75)), 4),
        'IQR': round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
    }

def main():
    print("=" * 60)
    print("Task E: Extracting Object-Size Distributions from Source Annotations")
    print("=" * 60)
    
    split_data = {}
    for s_name, s_dir in splits.items():
        print(f"Reading {s_name} from {s_dir}...")
        split_data[s_name] = parse_labels(s_dir)
        print(f"  Loaded {len(split_data[s_name]['width_px']):,} bounding boxes.")

    # Aggregate all source boxes
    all_w = np.concatenate([d['width_px'] for d in split_data.values()])
    all_h = np.concatenate([d['height_px'] for d in split_data.values()])
    all_area = np.concatenate([d['area_px'] for d in split_data.values()])
    all_norm_area = np.concatenate([d['norm_area'] for d in split_data.values()])

    print(f"\nTotal source bounding boxes analyzed: {len(all_w):,}")

    # Calculate statistics table
    stats_list = [
        calc_stats(all_w, "BBox Width (All Source)", "pixels (640x640)"),
        calc_stats(all_h, "BBox Height (All Source)", "pixels (640x640)"),
        calc_stats(all_area, "BBox Area (All Source)", "pixels^2"),
        calc_stats(all_norm_area, "BBox Normalized Area", "fraction of image area"),
        calc_stats(split_data['Train (4,379 images)']['area_px'], "Train Set BBox Area", "pixels^2"),
        calc_stats(split_data['Val (697 images)']['area_px'], "Validation Set BBox Area", "pixels^2"),
        calc_stats(split_data['Test (262 images)']['area_px'], "Test Set BBox Area", "pixels^2"),
    ]
    
    # Also size bins based on area percentiles:
    # Small: < P33 (~33%), Medium: P33-P66, Large: > P66
    p33, p66 = np.percentile(all_area, [33.33, 66.67])
    small_count = np.sum(all_area < p33)
    med_count = np.sum((all_area >= p33) & (all_area <= p66))
    large_count = np.sum(all_area > p66)
    
    print(f"Size bin thresholds: Small (<{p33:.1f} px^2), Medium ({p33:.1f}-{p66:.1f} px^2), Large (>{p66:.1f} px^2)")

    df_stats = pd.DataFrame(stats_list)
    csv_path = OUT_TAB / "object_size_statistics.csv"
    df_stats.to_csv(csv_path, index=False)
    print(f"[Done] Saved statistics: {csv_path}")

    # Plotting Publication-Quality Distributions
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10,
        'grid.alpha': 0.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. BBox Width Distribution
    ax0 = axes[0]
    ax0.hist(all_w, bins=60, range=(0, 150), color='#1f77b4', edgecolor='black', alpha=0.75, lw=0.6)
    w_med = np.median(all_w)
    w_mean = np.mean(all_w)
    ax0.axvline(w_med, color='red', linestyle='--', lw=1.8, label=f'Median: {w_med:.1f} px')
    ax0.axvline(w_mean, color='darkgreen', linestyle=':', lw=1.8, label=f'Mean: {w_mean:.1f} px')
    ax0.set_title("(a) Bounding Box Width", fontweight='bold', pad=10)
    ax0.set_xlabel("Width (pixels in 640×640)", fontweight='bold')
    ax0.set_ylabel("Instance Frequency", fontweight='bold')
    ax0.legend(frameon=True, facecolor='white', framealpha=0.9)

    # 2. BBox Height Distribution
    ax1 = axes[1]
    ax1.hist(all_h, bins=60, range=(0, 150), color='#ff7f0e', edgecolor='black', alpha=0.75, lw=0.6)
    h_med = np.median(all_h)
    h_mean = np.mean(all_h)
    ax1.axvline(h_med, color='red', linestyle='--', lw=1.8, label=f'Median: {h_med:.1f} px')
    ax1.axvline(h_mean, color='darkgreen', linestyle=':', lw=1.8, label=f'Mean: {h_mean:.1f} px')
    ax1.set_title("(b) Bounding Box Height", fontweight='bold', pad=10)
    ax1.set_xlabel("Height (pixels in 640×640)", fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

    # 3. BBox Area Distribution (Log Scale)
    ax2 = axes[2]
    # Filter non-zero
    log_area = np.log10(all_area[all_area > 0])
    ax2.hist(log_area, bins=60, color='#2ca02c', edgecolor='black', alpha=0.75, lw=0.6)
    med_log = np.median(log_area)
    ax2.axvline(med_log, color='red', linestyle='--', lw=1.8, 
                label=f'Median: {10**med_log:.0f} px$^2$ (10$^{{{med_log:.2f}}}$)')
    ax2.axvline(np.log10(p33), color='purple', linestyle=':', lw=1.5, label=f'P33: {p33:.0f} px$^2$')
    ax2.axvline(np.log10(p66), color='brown', linestyle=':', lw=1.5, label=f'P66: {p66:.0f} px$^2$')
    ax2.set_title("(c) Bounding Box Area (Log Scale)", fontweight='bold', pad=10)
    ax2.set_xlabel(r"log$_{10}$(Area [pixels$^2$])", fontweight='bold')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    png_path = OUT_FIG / "fig_object_size_distribution.png"
    pdf_path = OUT_FIG / "fig_object_size_distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] Saved size distribution figure: {png_path} and {pdf_path}")

if __name__ == '__main__':
    main()
