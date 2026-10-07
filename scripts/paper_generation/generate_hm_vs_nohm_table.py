"""Generate Controlled HM vs. No-HM Full Comparison Table.
Combines source held-out test metrics and target OHRC candidate behavior across all 5 models.
For models without a No-HM condition (YOLO26s and YOLO26m), values are reported as 'NA'.
Outputs:
- results/paper_tables/table_hm_vs_nohm_full.csv
- results/paper_tables/table_hm_vs_nohm_full.md
"""
import pandas as pd
from pathlib import Path

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
OUT_TAB = BASE / "results/paper_tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

# Full 5-model data dictionary
data = [
    {
        "Model": "YOLO26n",
        "Condition": "HM",
        "Test mAP50": 0.5292,
        "Test mAP50-95": 0.1896,
        "Test Precision": 0.5873,
        "Test Recall": 0.5069,
        "Test F1": 0.5442,
        "OHRC detections": 5084,
        "OHRC positive tiles": 2134,
        "OHRC positive-tile rate": "6.72%",
        "Mean confidence": 0.254,
    },
    {
        "Model": "YOLO26n",
        "Condition": "No-HM",
        "Test mAP50": 0.6736,
        "Test mAP50-95": 0.2664,
        "Test Precision": 0.6898,
        "Test Recall": 0.6172,
        "Test F1": 0.6515,
        "OHRC detections": 2429131,
        "OHRC positive tiles": 26956,
        "OHRC positive-tile rate": "84.85%",
        "Mean confidence": 0.347,
    },
    {
        "Model": "YOLO26s",
        "Condition": "HM",
        "Test mAP50": 0.6023,
        "Test mAP50-95": 0.2308,
        "Test Precision": 0.6260,
        "Test Recall": 0.5691,
        "Test F1": 0.5962,
        "OHRC detections": 14102,
        "OHRC positive tiles": 2050,
        "OHRC positive-tile rate": "6.45%",
        "Mean confidence": 0.295,
    },
    {
        "Model": "YOLO26s",
        "Condition": "No-HM",
        "Test mAP50": "NA",
        "Test mAP50-95": "NA",
        "Test Precision": "NA",
        "Test Recall": "NA",
        "Test F1": "NA",
        "OHRC detections": "NA",
        "OHRC positive tiles": "NA",
        "OHRC positive-tile rate": "NA",
        "Mean confidence": "NA",
    },
    {
        "Model": "YOLO26m",
        "Condition": "HM",
        "Test mAP50": 0.6463,
        "Test mAP50-95": 0.2622,
        "Test Precision": 0.6380,
        "Test Recall": 0.6172,
        "Test F1": 0.6275,
        "OHRC detections": 1484,
        "OHRC positive tiles": 443,
        "OHRC positive-tile rate": "1.39%",
        "Mean confidence": 0.293,
    },
    {
        "Model": "YOLO26m",
        "Condition": "No-HM",
        "Test mAP50": "NA",
        "Test mAP50-95": "NA",
        "Test Precision": "NA",
        "Test Recall": "NA",
        "Test F1": "NA",
        "OHRC detections": "NA",
        "OHRC positive tiles": "NA",
        "OHRC positive-tile rate": "NA",
        "Mean confidence": "NA",
    },
    {
        "Model": "YOLOv8n",
        "Condition": "HM",
        "Test mAP50": 0.4813,
        "Test mAP50-95": 0.1671,
        "Test Precision": 0.5469,
        "Test Recall": 0.4798,
        "Test F1": 0.5111,
        "OHRC detections": 353427,
        "OHRC positive tiles": 14129,
        "OHRC positive-tile rate": "44.47%",
        "Mean confidence": 0.304,
    },
    {
        "Model": "YOLOv8n",
        "Condition": "No-HM",
        "Test mAP50": 0.6455,
        "Test mAP50-95": 0.2488,
        "Test Precision": 0.6756,
        "Test Recall": 0.5930,
        "Test F1": 0.6316,
        "OHRC detections": 2448714,
        "OHRC positive tiles": 26755,
        "OHRC positive-tile rate": "84.22%",
        "Mean confidence": 0.331,
    },
    {
        "Model": "YOLOv5s",
        "Condition": "HM",
        "Test mAP50": 0.5737,
        "Test mAP50-95": 0.2121,
        "Test Precision": 0.6108,
        "Test Recall": 0.5499,
        "Test F1": 0.5788,
        "OHRC detections": 24452,
        "OHRC positive tiles": 3896,
        "OHRC positive-tile rate": "12.26%",
        "Mean confidence": 0.330,
    },
    {
        "Model": "YOLOv5s",
        "Condition": "No-HM",
        "Test mAP50": 0.7122,
        "Test mAP50-95": 0.2908,
        "Test Precision": 0.7207,
        "Test Recall": 0.6475,
        "Test F1": 0.6821,
        "OHRC detections": 3356769,
        "OHRC positive tiles": 30152,
        "OHRC positive-tile rate": "94.91%",
        "Mean confidence": 0.402,
    },
]

df = pd.DataFrame(data)
csv_path = OUT_TAB / "table_hm_vs_nohm_full.csv"
df.to_csv(csv_path, index=False)
print(f"[Artifact Saved] Full HM vs No-HM CSV: {csv_path}")

md_lines = [
    "# Controlled Histogram Matching (HM vs. No-HM) Full Comparison Table",
    "",
    "**Source Test Set:** Untouched 262-image held-out source split (7,268 ground truth instances).  ",
    "**Target OHRC Set:** 31,769 primary usable tiles (unlabeled; operational detection threshold $\\tau = 0.20$).  ",
    "**Note:** Where No-HM experiments were not conducted (YOLO26s, YOLO26m), values are explicitly recorded as NA.",
    "",
    "| Model | Condition | Test mAP50 | Test mAP50-95 | Test Precision | Test Recall | Test F1 | OHRC detections | OHRC positive tiles | OHRC positive-tile rate | Mean confidence |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
]

for _, r in df.iterrows():
    det_str = f"{r['OHRC detections']:,}" if isinstance(r['OHRC detections'], int) else str(r['OHRC detections'])
    pos_str = f"{r['OHRC positive tiles']:,}" if isinstance(r['OHRC positive tiles'], int) else str(r['OHRC positive tiles'])
    md_lines.append(
        f"| **{r['Model']}** | {r['Condition']} | "
        f"{r['Test mAP50']} | {r['Test mAP50-95']} | {r['Test Precision']} | {r['Test Recall']} | {r['Test F1']} | "
        f"{det_str} | {pos_str} | "
        f"{r['OHRC positive-tile rate']} | {r['Mean confidence']} |"
    )

md_lines.extend([
    "",
    "### Empirical Takeaways:",
    "1. **Source Representation Trade-Off:** Across all architectures evaluated with No-HM (YOLO26n, YOLOv8n, YOLOv5s), the No-HM condition achieves higher in-distribution test metrics on the native source distribution ($\\Delta \\text{mAP@0.5} \\approx -0.14$ to $-0.16$ under HM). This is because HM deliberately warps source histograms to match the target OHRC CDF, creating a synthetic domain divergence from the native source test distribution.",
    "2. **Target Activation Regularization:** In the unlabeled target OHRC domain, No-HM models suffer from massive over-activation (positive-tile rates of 84.22%–94.91%, with 2.43M to 3.36M detections). HM stabilizes the detectors, reducing YOLO26n positive tiles from 84.85% to 6.72% (5,084 detections) and YOLOv5s from 94.91% to 12.26% (24,452 detections).",
    "3. **Absence of Ground Truth:** Because target OHRC imagery is completely unlabeled, target candidate reductions must NOT be described as an accuracy improvement, but rather as evidence of photometric domain stabilization preventing widespread background false-triggering.",
])

md_path = OUT_TAB / "table_hm_vs_nohm_full.md"
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"[Artifact Saved] Full HM vs No-HM Markdown: {md_path}")
