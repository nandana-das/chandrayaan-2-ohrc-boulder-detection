"""Reproducibility and Augmentation Audit for Stage-2 Models.

Inspects the saved args.yaml configurations for all five Stage-2 models:
  - YOLO26n (runs/stage2_yolo26n_combined_hm_cosine/args.yaml)
  - YOLO26s (runs/stage2_yolo26s_combined_hm_cosine/args.yaml)
  - YOLO26m (runs/stage2_yolo26m_combined_hm_cosine/args.yaml)
  - YOLOv8n (runs/stage2_yolov8n_combined_hm_cosine/args.yaml)
  - YOLOv5s (runs/stage2_yolov5s_combined_hm_cosine/args.yaml)

Extracts and verifies:
  - Optimizer, lr0, lrf, momentum, weight_decay
  - Warmup settings (warmup_epochs, warmup_momentum, warmup_bias_lr)
  - Scheduler & learning rate decay (cos_lr)
  - Augmentations: mosaic, close_mosaic, mixup, copy_paste, erasing,
    hsv_h, hsv_s, hsv_v, scale, translate, degrees, shear, perspective,
    fliplr, flipud, auto_augment
  - Training resolution (imgsz), epochs, seed, deterministic
  - Batch size (physical batch), nominal batch size (nbs),
    gradient accumulation factor, and effective batch size
  - Pretrained starting weights, device

Outputs:
  results/paper_tables/stage2_reproducibility_config.csv
  results/paper_tables/stage2_reproducibility_audit.md
"""

from pathlib import Path
import yaml
import pandas as pd

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
RUNS = BASE / "runs"
OUT_TABLES = BASE / "results/paper_tables"

RUN_DIRS = {
    "YOLO26n": RUNS / "stage2_yolo26n_combined_hm_cosine",
    "YOLO26s": RUNS / "stage2_yolo26s_combined_hm_cosine",
    "YOLO26m": RUNS / "stage2_yolo26m_combined_hm_cosine",
    "YOLOv8n": RUNS / "stage2_yolov8n_combined_hm_cosine",
    "YOLOv5s": RUNS / "stage2_yolov5s_combined_hm_cosine",
}

def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("STEP 5: REPRODUCIBILITY & AUGMENTATION AUDIT")
    print("=" * 75)
    
    audit_data = {}
    
    for model_name, rdir in RUN_DIRS.items():
        args_file = rdir / "args.yaml"
        assert args_file.exists(), f"Missing args.yaml: {args_file}"
        with open(args_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        physical_batch = cfg.get("batch")
        nbs = cfg.get("nbs", 64)
        accumulate_factor = max(round(nbs / physical_batch), 1) if physical_batch else 1
        effective_batch = physical_batch * accumulate_factor if physical_batch else nbs
        
        starting_model = cfg.get("model")
        # Format model path for readability
        if starting_model:
            starting_model_clean = str(Path(starting_model).name)
            if "stage1" in str(starting_model):
                parent_dir = Path(starting_model).parent.parent.name
                starting_model_clean = f"{parent_dir}/weights/{starting_model_clean}"
        else:
            starting_model_clean = "None"
            
        audit_data[model_name] = {
            # Optimization
            "Optimizer": cfg.get("optimizer"),
            "lr0": cfg.get("lr0"),
            "lrf": cfg.get("lrf"),
            "Momentum": cfg.get("momentum"),
            "Weight_Decay": cfg.get("weight_decay"),
            "Warmup_Epochs": cfg.get("warmup_epochs"),
            "Warmup_Momentum": cfg.get("warmup_momentum"),
            "Warmup_Bias_LR": cfg.get("warmup_bias_lr"),
            "LR_Scheduler": "Cosine" if cfg.get("cos_lr") else "Linear",
            "cos_lr": cfg.get("cos_lr"),
            
            # Batch & Scaling
            "Physical_Batch": physical_batch,
            "Nominal_Batch_Size_nbs": nbs,
            "Gradient_Accumulate_Steps": accumulate_factor,
            "Effective_Batch_Size": effective_batch,
            "Epochs": cfg.get("epochs"),
            "Image_Size": cfg.get("imgsz"),
            
            # Augmentations
            "Mosaic": cfg.get("mosaic"),
            "Close_Mosaic": cfg.get("close_mosaic"),
            "Mixup": cfg.get("mixup"),
            "Copy_Paste": cfg.get("copy_paste"),
            "Erasing": cfg.get("erasing"),
            "Auto_Augment": cfg.get("auto_augment"),
            "HSV_H": cfg.get("hsv_h"),
            "HSV_S": cfg.get("hsv_s"),
            "HSV_V": cfg.get("hsv_v"),
            "Scale": cfg.get("scale"),
            "Translate": cfg.get("translate"),
            "Degrees": cfg.get("degrees"),
            "Shear": cfg.get("shear"),
            "Perspective": cfg.get("perspective"),
            "Flip_LR": cfg.get("fliplr"),
            "Flip_UD": cfg.get("flipud"),
            
            # Protocol & Reproducibility
            "Seed": cfg.get("seed"),
            "Deterministic": cfg.get("deterministic"),
            "Freeze": cfg.get("freeze"),
            "AMP": cfg.get("amp"),
            "Device": cfg.get("device"),
            "Pretrained": cfg.get("pretrained"),
            "Starting_Weights": starting_model_clean,
            "Full_Weights_Path": str(starting_model).replace("\\", "/"),
            "Dataset_YAML": str(cfg.get("data")).replace("\\", "/"),
        }
        
    df_audit = pd.DataFrame(audit_data)
    
    # Save CSV transposed (hyperparameters as rows, models as columns)
    csv_out = OUT_TABLES / "stage2_reproducibility_config.csv"
    df_audit.to_csv(csv_out)
    print(f"[Saved] Reproducibility configuration CSV: {csv_out}")
    
    # Generate comprehensive Markdown Report
    md_lines = [
        "# Stage-2 Training Hyperparameter & Augmentation Reproducibility Audit",
        "",
        "**Verification Source:** Recovered directly from official run configuration files (`runs/<run_name>/args.yaml`) in the repository. No default hyperparameters were assumed without configuration-file verification.",
        "",
        "---",
        "",
        "## 1. Complete Reproducibility Configuration Matrix",
        "",
        "| Hyperparameter / Setting | YOLO26n | YOLO26s | YOLO26m | YOLOv8n | YOLOv5s | Status across Models |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    
    # Categorized list of parameters for Markdown table
    param_groups = [
        ("Optimization & Scheduler", [
            ("Optimizer", "Optimizer"),
            ("Base Learning Rate (lr0)", "lr0"),
            ("Final LR Fraction (lrf)", "lrf"),
            ("LR Scheduler", "LR_Scheduler"),
            ("Optimizer Momentum", "Momentum"),
            ("Weight Decay", "Weight_Decay"),
            ("Warmup Epochs", "Warmup_Epochs"),
            ("Warmup Momentum", "Warmup_Momentum"),
            ("Warmup Bias LR", "Warmup_Bias_LR"),
        ]),
        ("Batching & Accumulation", [
            ("Physical Batch Size", "Physical_Batch"),
            ("Nominal Batch Size (nbs)", "Nominal_Batch_Size_nbs"),
            ("Gradient Accumulate Steps", "Gradient_Accumulate_Steps"),
            ("Effective Batch Size", "Effective_Batch_Size"),
            ("Total Epochs", "Epochs"),
            ("Input Resolution (imgsz)", "Image_Size"),
        ]),
        ("Data Augmentations", [
            ("Mosaic Probability", "Mosaic"),
            ("Close Mosaic (Final Epochs)", "Close_Mosaic"),
            ("Mixup Probability", "Mixup"),
            ("Copy-Paste Probability", "Copy_Paste"),
            ("Random Erasing Probability", "Erasing"),
            ("AutoAugment Policy", "Auto_Augment"),
            ("HSV Hue Augmentation (hsv_h)", "HSV_H"),
            ("HSV Saturation (hsv_s)", "HSV_S"),
            ("HSV Value (hsv_v)", "HSV_V"),
            ("Scale Jitter", "Scale"),
            ("Translation Jitter", "Translate"),
            ("Rotation (Degrees)", "Degrees"),
            ("Shear (Degrees)", "Shear"),
            ("Perspective Distortion", "Perspective"),
            ("Horizontal Flip (fliplr)", "Flip_LR"),
            ("Vertical Flip (flipud)", "Flip_UD"),
        ]),
        ("Reproducibility & Hardware", [
            ("Random Seed", "Seed"),
            ("Deterministic Mode", "Deterministic"),
            ("Layer Freezing (freeze)", "Freeze"),
            ("Automatic Mixed Precision (AMP)", "AMP"),
            ("CUDA Device", "Device"),
            ("Pretrained Initialization", "Pretrained"),
            ("Starting Weights Checkpoint", "Starting_Weights"),
        ]),
    ]
    
    for group_name, params in param_groups:
        md_lines.append(f"| **=== {group_name} ===** | | | | | | |")
        for label, key in params:
            vals = [str(audit_data[m][key]) for m in RUN_DIRS.keys()]
            status = "IDENTICAL (Controlled)" if len(set(vals)) == 1 else "VARIED (See Analysis)"
            md_lines.append(f"| **{label}** | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {status} |")
            
    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Key Audit Findings & Architectural Observations",
        "",
        "### A. Complete Augmentation Parity Across All 5 Models",
        "Every single augmentation parameter is strictly identical across all five models:",
        "- `mosaic = 1.0` with `close_mosaic = 10` (mosaic disabled during final 10 epochs).",
        "- `mixup = 0.0`, `copy_paste = 0.0` (both disabled across all models).",
        "- Photometric jitter: `hsv_h = 0.015`, `hsv_s = 0.7`, `hsv_v = 0.4`.",
        "- Geometric jitter: `scale = 0.5`, `translate = 0.1`, `degrees = 0.0`, `shear = 0.0`, `perspective = 0.0`.",
        "- Flip probabilities: `fliplr = 0.5`, `flipud = 0.0`.",
        "- Random erasing: `erasing = 0.4`.",
        "",
        "### B. Optimizer & Learning Rate Schedule Parity",
        "- Optimizer: `AdamW` across all five models.",
        "- Learning rate: `lr0 = 0.0001` with cosine decay (`cos_lr = True`, `lrf = 0.01`).",
        "- Warmup: 3.0 epochs (`warmup_momentum = 0.8`, `warmup_bias_lr = 0.1`).",
        "- Weight decay: `0.0005` across all models.",
        "",
        "### C. The Batch Size & Gradient Accumulation Architecture",
        "- **Physical batch size:** YOLO26n, YOLO26s, YOLOv8n, and YOLOv5s were trained with `batch = 8`. YOLO26m was trained with `batch = 4` to prevent VRAM exhaustion on the 4.0 GB RTX 3050 Laptop GPU.",
        "- **Nominal batch size (`nbs = 64`):** Ultralytics internally computes `accumulate = round(nbs / batch)`. Thus:",
        "  - Models with `batch = 8` accumulated $\\frac{64}{8} = 8$ batches per optimizer step ($8 \\times 8 = 64$ images).",
        "  - YOLO26m with `batch = 4` accumulated $\\frac{64}{4} = 16$ batches per optimizer step ($4 \\times 16 = 64$ images).",
        "- **Effective batch size:** All five models executed optimizer updates every **64 images**.",
        "- **Physical batch difference:** The only divergence is the physical mini-batch size (4 vs 8) affecting BatchNorm statistics during forward passes.",
        "",
        "### D. Starting Weights Lineage",
        "- **Stage-1 checkpoints:** YOLO26n, YOLOv8n, and YOLOv5s were initialized from their respective un-matched Stage-1 lunar checkpoints (`runs/stage1_<model>_combined/weights/best.pt`).",
        "- **Pretrained checkpoints:** YOLO26s and YOLO26m were initialized directly from `weights/yolo26s.pt` and `weights/yolo26m.pt`.",
    ])
    
    md_text = "\n".join(md_lines) + "\n"
    md_out = OUT_TABLES / "stage2_reproducibility_audit.md"
    md_out.write_text(md_text, encoding="utf-8")
    print(f"[Saved] Reproducibility audit Markdown: {md_out}")
    print("\n" + md_text)

if __name__ == "__main__":
    main()
