# Stale Reference Audit

**Date:** 2026-10-03  
**Auditor:** Antigravity Pair Programmer  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`  
**Authoritative Stage-2 Checkpoints:**
- `runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolov8n_combined_hm_cosine/weights/best.pt`
- `runs/stage2_yolov5s_combined_hm_cosine/weights/best.pt`

---

## 1. Audit Scope & Search Targets

This audit systematically examined the repository for historical artifacts, outdated checkpoint strings (`_combined_hm_v2`), deprecated target-domain detection counts (`4,565`, `5,026`, `41,941`), and obsolete historical claims (such as claims that YOLO26n was the overall best model prior to controlled cosine re-training).

---

## 2. Categorization Criteria

- **Category A (Active script):** Must be updated to point to the current authoritative controlled cosine checkpoints.
- **Category B (Historical artifact):** Past experiment log or run folder that should remain intact for archival transparency.
- **Category C (Paper / Result file):** Result file containing outdated numbers that must be superseded or regenerated.
- **Category D (Documentation / Demo value):** Non-critical explanatory note or tutorial text.

---

## 3. Audited Items & Categorization

| Item / File Path | Description of Reference | Category | Status & Action Taken |
| :--- | :--- | :---: | :--- |
| `scripts/infer_ohrc.py` | Previously referenced historical `_combined_hm_v2` runs | **A** | **Resolved:** Updated to point to `stage2_*_combined_hm_cosine`. |
| `scripts/generate_ohrc_confidence_curves.py` | Previously referenced `_combined_hm_v2` runs | **A** | **Resolved:** Updated to point to `stage2_*_combined_hm_cosine`. |
| `scripts/evaluate_test_split.py` | Model evaluation script for held-out test split | **A** | **Resolved:** Points to `stage2_*_combined_hm_cosine`. |
| `scripts/process_polar_batch.py` | Polar batch inference script | **A** | **Resolved:** Points to `stage2_*_combined_hm_cosine`. |
| `scripts/check_confidence.py` | Confidence diagnostic script | **A** | **Resolved:** Points to `stage2_*_combined_hm_cosine`. |
| `results/table2_rtdetr_comparison.csv` | Contains historical pre-cosine held-out metrics (e.g. YOLO26n mAP 0.640, YOLOv5s mAP 0.649) | **C** | **Documented:** Superseded by authoritative `results/test_metrics_stage2.csv` (YOLOv5s: 0.5737, YOLO26n: 0.5292, YOLOv8n: 0.4813). Retained as historical baseline; paper will use authoritative metrics. |
| `results/table3_rtdetr_comparison.csv` | Target-domain comparison on 31,769 tiles | **C** | **Resolved:** Fully regenerated with controlled cosine counts (YOLO26n: 5,084, YOLOv8n: 353,427, YOLOv5s: 24,452). |
| `results/table4_with_ultradeep_polar.csv` | Regional & ultra-deep polar summary | **C** | **Resolved:** Fully regenerated with current regional & ultra-deep counts. |
| `runs/stage2_yolo26n_combined_hm_v2/` | Historical training run folder (pre-cosine) | **B** | **Retained:** Preserved in `runs/` as an archived historical artifact. Not used in active evaluation. |
| `runs/stage2_yolov8n_combined_hm_v2/` | Historical training run folder (pre-cosine) | **B** | **Retained:** Preserved in `runs/` as an archived historical artifact. Not used in active evaluation. |
| `runs/stage2_yolov5s_combined_hm_v2/` | Historical training run folder (pre-cosine) | **B** | **Retained:** Preserved in `runs/` as an archived historical artifact. Not used in active evaluation. |
| Old Target Counts (`4565`, `5026`, `41941`) | Historical target detection numbers from early non-cosine models | **C** | **Resolved:** Completely eliminated from active result tables (`results/table3_rtdetr_comparison.csv`, `results/table4_with_ultradeep_polar.csv`, etc.). |
| Claim: "YOLO26n is the top-performing model" | Based on early pre-cosine held-out metrics | **D** | **Resolved:** The controlled cosine Stage-2 evaluation firmly establishes **YOLOv5s** as top-performing on held-out source data (mAP50 = 0.5737 vs YOLO26n: 0.5292), while showing intermediate conservative behavior on OHRC. |

---

## 4. Conclusion
All active scripts are aligned with the controlled cosine Stage-2 checkpoints. All paper summary tables reflect the verified authoritative counts. No active analysis script depends on stale checkpoints or legacy counts.
