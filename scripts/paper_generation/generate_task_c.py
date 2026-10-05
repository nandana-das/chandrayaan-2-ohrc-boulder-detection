"""Task C: Same-Tile Cross-Model Comparison on Unlabeled OHRC Imagery.
Visualizes 4 representative tiles where model behavior illustrates the target-domain detection densities:
YOLO26n (conservative, 5,084 total detections),
YOLOv5s (intermediate, 24,452 total detections),
YOLOv8n (dense/hypersensitive, 353,427 total detections).
Saves:
- results/paper_figures/fig_cross_model_behavior.png
- results/paper_figures/fig_cross_model_behavior.pdf
"""
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
TILES_DIR = BASE / "data/tiles/usable"
RESULTS = BASE / "results"
OUT_FIG = RESULTS / "paper_figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Selected representative tiles showing distinct behavioral regimes
selected_tiles = [
    {
        'id': 'Tile A: Isolated Candidate / Sparse Field',
        'filename': 'ch2_ohr_ncp_20240425T1209509264_d_img_d18_x01920_y19200.png',
        'desc': 'Single prominent boulder candidate detected by YOLO26n; intermediate detection by YOLOv5s; background clutter triggered by YOLOv8n.'
    },
    {
        'id': 'Tile B: Multi-Candidate Cluster / Impact Rim',
        'filename': 'ch2_ohr_ncp_20240425T1012478407_d_img_d18_x01920_y91520.png',
        'desc': 'High-confidence boulder cluster along crater rim: YOLO26n and YOLOv5s isolate crater edge features while YOLOv8n floods surrounding ejecta.'
    },
    {
        'id': 'Tile C: Dense Boulder Field / Landslide Ejecta',
        'filename': 'ch2_ohr_ncp_20190906T2241285714_d_img_gds_x08960_y10240.png',
        'desc': 'Equatorial boulder field: YOLOv5s identifies 21 candidates with crisp localization; YOLO26n is conservative (12); YOLOv8n produces 58 detections.'
    },
    {
        'id': 'Tile D: High-Contrast Slump / Texture Boundary',
        'filename': 'ch2_ohr_ncp_20240425T1406019344_d_img_d18_x01920_y75520.png',
        'desc': 'Extreme texture boundary illustrating YOLOv8n hypersensitivity (170 detections) vs YOLOv5s moderation (4) and YOLO26n peak response (26).'
    },
]

# Load detection CSVs
df26 = pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolo26n.csv')
df26s = pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolo26s.csv')
df26m = pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolo26m.csv')
dfv8 = pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolov8n.csv')
dfv5 = pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolov5s.csv')

colors = {
    'YOLO26n': '#0055d4',  # deep blue
    'YOLO26s': '#0097a7',  # dark cyan
    'YOLO26m': '#7b1fa2',  # deep purple
    'YOLOv8n': '#e65100',  # vibrant orange
    'YOLOv5s': '#1b5e20',  # dark green
}

def draw_boxes(ax, df_tile, color, max_boxes=40):
    boxes = df_tile.sort_values(by='conf', ascending=False)
    count = len(boxes)
    # Draw boxes
    for idx, (_, row) in enumerate(boxes.iterrows()):
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        w = max(x2 - x1, 2)
        h = max(y2 - y1, 2)
        conf = row['conf']
        
        # High opacity for top boxes, lower for dense background
        alpha = 0.9 if idx < 15 else 0.45
        lw = 1.4 if idx < 15 else 0.8
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(rect)
        
        # Annotate top 4 boxes with confidence
        if idx < 4:
            ax.text(x1, max(y1 - 3, 10), f"{conf:.2f}", color='yellow', fontsize=6.5,
                    fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', facecolor='black', alpha=0.6, lw=0))

def main():
    print("=" * 60)
    print("Task C: Generating Same-Tile Cross-Model Comparison Figure")
    print("=" * 60)
    
    n_tiles = len(selected_tiles)
    fig, axes = plt.subplots(n_tiles, 6, figsize=(24, 4.0 * n_tiles))
    
    col_titles = [
        "Unlabeled OHRC Tile (Raw)",
        "YOLO26n (Nano)",
        "YOLO26s (Small)",
        "YOLO26m (Medium)",
        "YOLOv8n (Dense Baseline)",
        "YOLOv5s (Intermediate Baseline)"
    ]
    
    for row_idx, item in enumerate(selected_tiles):
        fname = item['filename']
        img_path = TILES_DIR / fname
        assert img_path.exists(), f"Missing tile: {img_path}"
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Subsets
        b26 = df26[df26['tile'] == fname]
        b26s = df26s[df26s['tile'] == fname]
        b26m = df26m[df26m['tile'] == fname]
        bv8 = dfv8[dfv8['tile'] == fname]
        bv5 = dfv5[dfv5['tile'] == fname]
        
        # 0. Raw image
        ax0 = axes[row_idx, 0]
        ax0.imshow(img, cmap='gray')
        ax0.set_title(col_titles[0] if row_idx == 0 else "", fontsize=12, fontweight='bold', pad=8)
        ax0.set_ylabel(item['id'], fontsize=11, fontweight='bold', labelpad=8)
        ax0.axis('off')
        
        # 1. YOLO26n
        ax1 = axes[row_idx, 1]
        ax1.imshow(img, cmap='gray')
        draw_boxes(ax1, b26, colors['YOLO26n'])
        ax1.set_title(col_titles[1] if row_idx == 0 else "", fontsize=12, fontweight='bold', pad=8)
        ax1.text(0.03, 0.94, f"N = {len(b26)}", transform=ax1.transAxes, color='white',
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#0055d4', alpha=0.85))
        ax1.axis('off')
        
        # 2. YOLO26s
        ax2 = axes[row_idx, 2]
        ax2.imshow(img, cmap='gray')
        draw_boxes(ax2, b26s, colors['YOLO26s'])
        ax2.set_title(col_titles[2] if row_idx == 0 else "", fontsize=12, fontweight='bold', pad=8)
        ax2.text(0.03, 0.94, f"N = {len(b26s)}", transform=ax2.transAxes, color='white',
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#0097a7', alpha=0.85))
        ax2.axis('off')
        
        # 3. YOLO26m
        ax3 = axes[row_idx, 3]
        ax3.imshow(img, cmap='gray')
        draw_boxes(ax3, b26m, colors['YOLO26m'])
        ax3.set_title(col_titles[3] if row_idx == 0 else "", fontsize=12, fontweight='bold', pad=8)
        ax3.text(0.03, 0.94, f"N = {len(b26m)}", transform=ax3.transAxes, color='white',
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#7b1fa2', alpha=0.85))
        ax3.axis('off')
        
        # 4. YOLOv8n
        ax4 = axes[row_idx, 4]
        ax4.imshow(img, cmap='gray')
        draw_boxes(ax4, bv8, colors['YOLOv8n'])
        ax4.set_title(col_titles[4] if row_idx == 0 else "", fontsize=12, fontweight='bold', pad=8)
        ax4.text(0.03, 0.94, f"N = {len(bv8)}", transform=ax4.transAxes, color='white',
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#e65100', alpha=0.85))
        ax4.axis('off')
        
        # 5. YOLOv5s
        ax5 = axes[row_idx, 5]
        ax5.imshow(img, cmap='gray')
        draw_boxes(ax5, bv5, colors['YOLOv5s'])
        ax5.set_title(col_titles[5] if row_idx == 0 else "", fontsize=12, fontweight='bold', pad=8)
        ax5.text(0.03, 0.94, f"N = {len(bv5)}", transform=ax5.transAxes, color='white',
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1b5e20', alpha=0.85))
        ax5.axis('off')

    plt.tight_layout()
    png_path = OUT_FIG / "fig_cross_model_behavior.png"
    pdf_path = OUT_FIG / "fig_cross_model_behavior.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] Saved cross-model comparison figure: {png_path} and {pdf_path}")

if __name__ == '__main__':
    main()
