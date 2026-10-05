"""Task B: Qualitative OHRC Candidate Detection Figure & Documentation.
Generates a 3x3 grid of actual target-domain candidate detections from unlabeled Chandrayaan-2 OHRC imagery.
Labels all predictions strictly as candidate detections without ground-truth claims.
Saves:
- results/paper_figures/fig_ohrc_qualitative_detections.png
- results/paper_figures/fig_ohrc_qualitative_detections.pdf
- results/paper_tables/qualitative_examples.csv
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
OUT_TAB = RESULTS / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# 9 Selected qualitative cases for a 3x3 grid
examples = [
    {
        'panel': '(a) High-Confidence Candidate',
        'tile': 'ch2_ohr_ncp_20240425T1406019344_d_img_d18_x01920_y75520.png',
        'model': 'YOLO26n',
        'category': 'High-confidence candidate detection',
        'feature_type': 'Prominent boulder cluster on high-relief slope',
        'crop_box': None, # full tile or centered
        'reason': 'Exemplifies high-confidence (conf > 0.50) localization by YOLO26n on steep terrain.'
    },
    {
        'panel': '(b) Isolated Candidate with Cast Shadow',
        'tile': 'ch2_ohr_ncp_20240425T1209509264_d_img_d18_x01920_y19200.png',
        'model': 'YOLO26n',
        'category': 'Isolated candidate with shadow',
        'feature_type': 'Single isolated boulder candidate with distinct east-facing shadow',
        'reason': 'Demonstrates single isolated candidate detection with photometric shadow alignment.'
    },
    {
        'panel': '(c) Rim-Adjacent Ejecta Candidate',
        'tile': 'ch2_ohr_ncp_20240425T1012478407_d_img_d18_x01920_y91520.png',
        'model': 'YOLOv5s',
        'category': 'Rim-adjacent candidate',
        'feature_type': 'Crater crest rim with candidate block fragments',
        'reason': 'Shows intermediate YOLOv5s candidate detection along crater rim margin.'
    },
    {
        'panel': '(d) Multiple Detections in One Tile',
        'tile': 'ch2_ohr_ncp_20190906T2241285714_d_img_gds_x08960_y10240.png',
        'model': 'YOLOv5s',
        'category': 'Multiple detections in boulder field',
        'feature_type': 'Equatorial ejecta field with distributed candidate boulders',
        'reason': 'Illustrates multiple distinct candidate bounding boxes resolved across a complex surface.'
    },
    {
        'panel': '(e) Scaled Architecture Detection',
        'tile': 'ch2_ohr_ncp_20190906T2241285714_d_img_gds_x05120_y07040.png',
        'model': 'YOLO26m',
        'category': 'Scaled YOLO26m candidate cluster',
        'feature_type': 'Prominent boulder group detected by scaled YOLO26m',
        'reason': 'Demonstrates YOLO26m high-confidence boulder localization with restrained activation rate.'
    },
    {
        'panel': '(f) Sparse Conservative Behavior',
        'tile': 'ch2_ohr_ncp_20240425T1406019344_d_img_d18_x01920_y03200.png',
        'model': 'YOLO26n',
        'category': 'Sparse YOLO26n behavior',
        'feature_type': 'Low-incidence regolith terrain with isolated candidate',
        'reason': 'Shows YOLO26n conservative thresholding detecting only the most distinct rock candidates.'
    },
    {
        'panel': '(g) Dense Texture Response',
        'tile': 'ch2_ohr_ncp_20240425T1406019344_d_img_d18_x01280_y23680.png',
        'model': 'YOLOv8n',
        'category': 'Dense YOLOv8n clustering',
        'feature_type': 'High-roughness ejecta blanket triggering dense bounding boxes',
        'reason': 'Exemplifies YOLOv8n hypersensitivity where small-scale regolith roughness triggers dense candidate boxes.'
    },
    {
        'panel': '(h) Shadow-Associated Depressed Feature',
        'tile': 'ch2_ohr_ncp_20190906T2241285714_d_img_gds_x00000_y100480.png',
        'model': 'YOLO26n',
        'category': 'Shadow-associated candidate',
        'feature_type': 'Candidate boulder at floor of small pit crater with attached shadow',
        'reason': 'Demonstrates boundary detection at interface of illuminated crest and internal shadow.'
    },
    {
        'panel': '(i) Ambiguous / Low-Confidence Candidates',
        'tile': 'ch2_ohr_ncp_20240425T1406019344_d_img_d18_x01920_y80000.png',
        'model': 'YOLOv5s',
        'category': 'Ambiguous / low-confidence candidate',
        'feature_type': 'Degraded crater floor with low contrast features (conf ~ 0.21)',
        'reason': 'Shows threshold-adjacent candidate detection near the operational cutoff (tau = 0.20).'
    },
]

# Load model detection CSVs
dfs = {
    'YOLO26n': pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolo26n.csv'),
    'YOLO26m': pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolo26m.csv'),
    'YOLOv8n': pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolov8n.csv'),
    'YOLOv5s': pd.read_csv(RESULTS / 'ohrc_inference/raw_detections_yolov5s.csv'),
}

box_colors = {
    'YOLO26n': '#00e5ff',  # bright cyan for dark imagery
    'YOLO26m': '#e040fb',  # bright neon magenta/purple
    'YOLOv8n': '#ff9100',  # amber orange
    'YOLOv5s': '#76ff03',  # lime green
}

def main():
    print("=" * 60)
    print("Task B: Generating Qualitative OHRC Detection Figure & Manifest")
    print("=" * 60)
    
    # Save CSV manifest
    csv_rows = []
    for ex in examples:
        fname = ex['tile']
        m = ex['model']
        sub = dfs[m][dfs[m]['tile'] == fname]
        det_count = len(sub)
        max_c = sub['conf'].max() if det_count > 0 else 0.0
        mean_c = sub['conf'].mean() if det_count > 0 else 0.0
        
        csv_rows.append({
            'Panel': ex['panel'],
            'Tile_Filename': fname,
            'Model_Shown': m,
            'Detection_Category': ex['category'],
            'Geomorphological_Context': ex['feature_type'],
            'Selection_Rationale': ex['reason'],
            'Total_Detections_in_Tile': det_count,
            'Max_Confidence': round(float(max_c), 3),
            'Mean_Confidence': round(float(mean_c), 3),
        })
        
    df_manifest = pd.DataFrame(csv_rows)
    csv_path = OUT_TAB / "qualitative_examples.csv"
    df_manifest.to_csv(csv_path, index=False)
    print(f"[Done] Saved manifest: {csv_path}")

    # Plot 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, ex in enumerate(examples):
        ax = axes[i]
        fname = ex['tile']
        m = ex['model']
        img_p = TILES_DIR / fname
        assert img_p.exists(), f"Tile missing: {img_p}"
        
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        
        # Overlay detections
        sub = dfs[m][dfs[m]['tile'] == fname].sort_values(by='conf', ascending=False)
        c_color = box_colors[m]
        
        # Plot boxes
        for idx, (_, r) in enumerate(sub.iterrows()):
            x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
            conf = r['conf']
            w = max(x2 - x1, 2)
            h = max(y2 - y1, 2)
            
            # Draw box
            lw = 1.6 if idx < 10 else 0.9
            alpha = 0.95 if idx < 10 else 0.5
            rect = patches.Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=c_color, facecolor='none', alpha=alpha)
            ax.add_patch(rect)
            
            # Draw label for top 3 boxes
            if idx < 3:
                ax.text(x1, max(y1 - 4, 12), f"{conf:.2f}", color='black', fontsize=7,
                        fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', facecolor=c_color, alpha=0.9, lw=0))

        # Title & Panel Annotation
        ax.set_title(ex['panel'], fontsize=11, fontweight='bold', pad=8)
        
        # Sub-badge with Model & Count
        badge_text = f"{m} (N = {len(sub)})"
        ax.text(0.03, 0.05, badge_text, transform=ax.transAxes, color='white',
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.75))
        ax.axis('off')

    plt.suptitle("Candidate Detections on Unlabeled Chandrayaan-2 OHRC Imagery (Operational $\\tau = 0.20$)\n"
                 "[Note: Features represent candidate rockfall-related boulder detections; target domain lacks ground-truth annotations]",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    png_path = OUT_FIG / "fig_ohrc_qualitative_detections.png"
    pdf_path = OUT_FIG / "fig_ohrc_qualitative_detections.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] Saved qualitative figure: {png_path} and {pdf_path}")

if __name__ == '__main__':
    main()
