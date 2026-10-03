"""Task F: Model Complexity Table for YOLO26n, YOLOv8n, and YOLOv5s.
Extracts parameter counts, GFLOPs, model weight size, layers, and input dimensions directly from models.
Saves:
- results/paper_tables/model_complexity.csv
- results/paper_tables/model_complexity.md
"""
import torch
import thop
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
RUNS = BASE / "runs"
RESULTS = BASE / "results"
OUT_TAB = RESULTS / "paper_tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

MODELS = {
    'YOLO26n': RUNS / 'stage2_yolo26n_combined_hm_cosine/weights/best.pt',
    'YOLOv8n': RUNS / 'stage2_yolov8n_combined_hm_cosine/weights/best.pt',
    'YOLOv5s': RUNS / 'stage2_yolov5s_combined_hm_cosine/weights/best.pt',
}

def main():
    print("=" * 60)
    print("Task F: Extracting Model Complexity Metrics")
    print("=" * 60)
    
    records = []
    dummy_input = torch.zeros(1, 3, 640, 640)
    
    for name, p in MODELS.items():
        assert p.exists(), f"Weights missing: {p}"
        size_mb = p.stat().st_size / (1024 * 1024)
        
        yolo_obj = YOLO(str(p))
        eval_model = yolo_obj.model.eval()
        
        # Calculate precise GFLOPs and parameter count using thop profiling
        flops, params_profiled = thop.profile(eval_model, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
        
        # Total parameters and layers
        total_params = sum(param.numel() for param in eval_model.parameters())
        layers = len(list(eval_model.modules()))
        
        arch = 'YOLOv11-Nano (Custom Head)' if name == 'YOLO26n' else ('YOLOv8-Nano' if name == 'YOLOv8n' else 'YOLOv5-Small')
        
        print(f"{name}: Params = {total_params:,}, GFLOPs = {flops_g:.2f}, Size = {size_mb:.2f} MB, Layers = {layers}")
        
        records.append({
            'Model': name,
            'Architecture_Base': arch,
            'Input_Resolution': '640 × 640',
            'Parameters': total_params,
            'Parameters_M': round(total_params / 1e6, 3),
            'GFLOPs': round(flops_g, 2),
            'Model_Size_MB': round(size_mb, 2),
            'Total_Modules': layers,
        })
        
    df = pd.DataFrame(records)
    csv_path = OUT_TAB / "model_complexity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Done] Saved CSV: {csv_path}")
    
    # Generate publication-ready markdown table
    md_lines = [
        "# Model Complexity & Architectural Properties (Authoritative Stage-2 Models)",
        "",
        "| Model | Architecture Base | Input Resolution | Parameters (M) | FLOPs (G) | File Size (MB) | Total Modules |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        md_lines.append(
            f"| **{r['Model']}** | {r['Architecture_Base']} | {r['Input_Resolution']} | {r['Parameters_M']:.3f}M ({r['Parameters']:,}) | {r['GFLOPs']:.2f} G | {r['Model_Size_MB']:.2f} MB | {r['Total_Modules']} |"
        )
    md_lines.append("")
    md_lines.append("> **Note:** Parameter counts and GFLOPs were directly measured using `thop` profiling on a single 1×3×640×640 tensor input on the PyTorch execution graph. File size represents the serialized TorchScript/PyTorch checkpoint on disk.")
    
    md_content = "\n".join(md_lines) + "\n"
    md_path = OUT_TAB / "model_complexity.md"
    md_path.write_text(md_content, encoding='utf-8')
    print(f"[Done] Saved Markdown: {md_path}")
    print("\n" + md_content)

if __name__ == '__main__':
    main()
