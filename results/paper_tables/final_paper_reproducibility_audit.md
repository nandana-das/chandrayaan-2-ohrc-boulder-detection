# Final Reproducibility and Paper-Audit Pass Report

**Repository:** `https://github.com/nandana-das/chandrayaan-2-ohrc-boulder-detection`  
**Local Workspace:** `D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection`  
**Branch:** `main`  
**Base Commit (HEAD):** `52cd2eb534dca27bedcfe2f40e797f55103d7027`  
**Audit Date:** 2026-10-07  
**Scope:** Stage-2 Five-Model Benchmark, Reproducibility Audit, Numerical Consistency, Scientific Language Neutralization, and Manuscript Compliance.

---

## 1. Executive Summary & Audit Outcome

This audit represents the final comprehensive reproducibility, consistency, and scientific-rigor verification pass for the research manuscript:
> *"Domain-Adaptive Transfer Learning for Rockfall-Related Feature Detection in Chandrayaan-2 OHRC Imagery"*

### Summary Assessment:
- **Retraining Required:** **0 (Zero).** All five Stage-2 models, ablation checkpoints, inference caches, and evaluation tables are intact, valid, and internally consistent. Retraining is neither necessary nor justified.
- **Experimental Integrity:** 100% preserved. No experimental results were fabricated, altered, or artificially smoothed.
- **Reproducibility Status:** `requirements.txt` has been pinned to `ultralytics==8.4.173` with explicit documentation of the historical `8.4.155` training environment recovered from experiment logs.
- **Language Neutralization:** Complete removal of overclaiming terminology ("false positives" on unlabeled data, "catastrophic false-triggering", "hyper-activation", "genuine geomorphic structures", "statistically significant"). Replaced with technically precise, objective terminology.
- **Batch Size Parity:** Verified from `args.yaml` configurations and framework accumulation logic (`nbs = 64`); phrasing updated to accurately convey physical mini-batch differences versus optimizer accumulation parity.

---

## 2. Objective 1 — Batch Size & Effective Optimizer Batch Verification

### A. Saved Run Configurations (`args.yaml`)
Inspected from `runs/stage2_<model>_combined_hm_cosine/args.yaml`:

| Model | Physical Mini-Batch (`batch`) | Nominal Batch Size (`nbs`) | Framework Accumulate Formula | Accumulate Steps | Effective Optimizer Batch Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 8 | 64 | `max(round(64 / 8), 1)` | 8 | **64 images** |
| **YOLO26s** | 8 | 64 | `max(round(64 / 8), 1)` | 8 | **64 images** |
| **YOLO26m** | 4 | 64 | `max(round(64 / 4), 1)` | 16 | **64 images** |
| **YOLOv8n** | 8 | 64 | `max(round(64 / 8), 1)` | 8 | **64 images** |
| **YOLOv5s** | 8 | 64 | `max(round(64 / 8), 1)` | 8 | **64 images** |

### B. Implementation-Derived vs. Runtime Evidence Note
- **No Fabricated Logs:** The raw training terminal output did not explicitly emit a text line reading `accumulate=16`. Therefore, this property is described as **framework- and configuration-derived**, based on the saved `batch: 4` and `nbs: 64` hyperparameters in `args.yaml` and the Ultralytics training engine accumulation formula (`ultralytics/engine/trainer.py`).
- **Memory Constraint Justification:** Physical batch 8 on YOLO26m required 4.70 GB VRAM, exceeding the physical 4.0 GB limit of the NVIDIA RTX 3050 Laptop GPU and causing memory exhaustion / CUDA OOM.
- **Refined Neutral Phrasing:** Replaced the previous phrase *"The only divergence is batch normalization statistics"* with the approved technically accurate wording:
  > *"The YOLO26m run used a smaller physical mini-batch because of the 4-GB GPU memory constraint; the framework's accumulation schedule maintained an effective optimizer batch size of 64. The physical mini-batch difference may still affect batch-dependent behavior."*

---

## 3. Objective 2 — Reproducibility & Environment Version Pinning

### A. Ultralytics Version Provenance
- **Historical Training Logs:** Record `Ultralytics 8.4.155` under Python 3.11.9, PyTorch 2.5.1+cu121 on Windows (documented in `results/paper_tables/hm_ablation_report.md`, Line 52).
- **Audit & Evaluation Environment:** Executed under `Ultralytics 8.4.173` on Python 3.11 / PyTorch 2.6.0+cu124.
- **Resolution in `requirements.txt`:** Updated from loose `ultralytics>=8.0` to pinned:
  ```txt
  ultralytics==8.4.173  # Historical training executed under Ultralytics 8.4.155; evaluation/audit verified on 8.4.173
  ```
  This preserves exact audit repeatability without inventing unverified dependencies.

---

## 4. Objective 3 — Numerical Consistency Audit

Every reported quantity has been audited against the authoritative raw CSV and JSON artifacts:

| Item | Stale / Discrepant Value | Authoritative Verified Value | Authoritative Source Artifact | Status |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO26m positive tile rate ($\tau=0.20$)** | ~1.83% (draft prose) | **1.39%** (1,484 candidate detections / 443 positive tiles) | `results/confidence_analysis/target_domain_threshold_sensitivity.csv` | **FIXED** in all summaries and scripts |
| **YOLO26n ultra-deep polar positive tiles** | 1,174 (early draft typo) | **1,173** (10,085 detections, 8.44% positive tile rate) | `results/paper_tables/ultradeep_polar_summary.csv` | **VERIFIED** |
| **YOLO26s parameter count** | 10.01M (un-consolidated) | **9.949M** (9,948,638 parameters) | `results/paper_tables/model_complexity.csv` | **VERIFIED & ANNOTATED** as authoritative |
| **YOLO26m parameter count** | 21.90M (un-consolidated) | **21.774M** (21,774,430 parameters) | `results/paper_tables/model_complexity.csv` | **VERIFIED & ANNOTATED** as authoritative |
| **YOLO26m inference latency** | 42.69 ms (older draft) | **59.96 ms** mean / **61.02 ms** median / **16.7 FPS** | `results/paper_tables/inference_latency.csv` | **VERIFIED** |
| **Held-out source test split** | Varies in older notes | **262 images, 7,268 GT instances** | `data/combined_hm/dataset_combined_hm.yaml` | **VERIFIED** |
| **Primary OHRC mosaic tile count** | 31,769 | **31,769 usable tiles** (640×640 px) | `results/dataset_manifest.csv` | **VERIFIED** |
| **Ultra-deep polar tile count** | 13,906 | **13,906 usable tiles** across 20 PDS4 products | `results/paper_tables/ultradeep_polar_summary.csv` | **VERIFIED** |

---

## 5. Objective 4 — Scientific Language Neutralization

All instances of speculative, evaluative, or overclaiming prose across reports and scripts were replaced:

| Old / Prohibited Wording | New Scientifically Objective Wording | Rationale |
| :--- | :--- | :--- |
| "false positives" / "false negatives" on OHRC | **"candidate detections" / "unconfirmed candidate features"** | The target Chandrayaan-2 OHRC mosaic is completely unlabeled. |
| "catastrophic target-domain false-triggering" | **"extensive target-domain candidate over-activation"** | Removes emotionally charged hyperbole; objectively describes tile activation density. |
| "catastrophic over-activation" | **"extensive candidate over-activation"** | Retains quantitative observation without hyperbole. |
| "hyper-activating architectures" | **"higher-activation architectures"** | Neutral behavioral description. |
| "genuine geomorphic structures" | **"prominent local contrast features"** | Unlabeled optical features cannot be asserted as verified geomorphic structures without field or altimetric ground truth. |
| "statistically significant" | **"the bootstrap intervals indicate clear separation in this evaluation"** | Non-overlapping empirical bootstrap percentiles do not constitute a formal null-hypothesis significance test. |
| "state-of-the-art" | **"best among the evaluated models"** | Scopes claim strictly to the controlled benchmark models. |
| "target-domain accuracy / precision / recall" | **"candidate activation density / positive-tile rate / confidence distribution"** | Preserves mathematical integrity under absence of target ground truth. |
| Single physical cause for polar disparity | **"The regional variation may be influenced by differences in solar incidence angle, illumination, terrain morphology, and product distribution."** | Acknowledges multiple confounding observational and physical variables. |

---

## 6. Objective 5 — Histogram Matching Description Parity

- **Implementation Fact:** The preparation pipeline (`scripts/combined_hm_prep.py`) samples 20 representative Chandrayaan-2 OHRC tiles, computes their average cumulative distribution function (CDF) to construct **one single reference image/profile**, and applies histogram specification from that averaged reference across the source dataset.
- **Terminology Rule:**
  - **APPROVED:** *"20-tile averaged-reference histogram matching"*
  - **PROHIBITED:** *"multi-reference histogram matching"* (which erroneously implies multiple distinct reference profiles or multi-modal matching).
- **Verification:** Audited across `scripts/train_rtdetr_pipeline.py`, `demo/index.html`, and paper tables. All occurrences of "multi-reference" have been corrected.

---

## 7. Objective 6 & 7 — Source-Domain Benchmark & Bootstrap Confidence Intervals

### Controlled Held-Out Source Test Set
- **Dataset Split:** 262 images, 7,268 ground-truth boulder instances (LROC NAC / Prieur et al. test split).
- **Bootstrap Protocol:** $N = 1,000$ image-level resamples with replacement, fixed random seed 42, evaluated under official Ultralytics DetectionValidator `ap_per_class` protocol.

### Consolidated Authoritative Table (`results/paper_tables/test_metrics_bootstrap_summary.csv`):

| Model | Precision (95% CI) | Recall (95% CI) | F1-Score (95% CI) | mAP@0.5 (95% CI) | mAP@0.5:0.95 (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26m** | 0.6380 [0.6128, 0.6655] | 0.6172 [0.5865, 0.6403] | 0.6275 [0.6023, 0.6492] | **0.6463 [0.6237, 0.6709]** | **0.2622 [0.2500, 0.2755]** |
| **YOLO26s** | 0.6260 [0.5982, 0.6526] | 0.5691 [0.5468, 0.5901] | 0.5962 [0.5747, 0.6168] | **0.6023 [0.5774, 0.6286]** | **0.2308 [0.2187, 0.2439]** |
| **YOLOv5s** | 0.6108 [0.5898, 0.6335] | 0.5499 [0.5293, 0.5702] | 0.5788 [0.5620, 0.5957] | **0.5737 [0.5510, 0.5979]** | **0.2121 [0.2006, 0.2237]** |
| **YOLO26n** | 0.5873 [0.5660, 0.6118] | 0.5069 [0.4855, 0.5274] | 0.5442 [0.5260, 0.5624] | **0.5292 [0.5030, 0.5551]** | **0.1896 [0.1788, 0.2015]** |
| **YOLOv8n** | 0.5469 [0.5205, 0.5706] | 0.4798 [0.4606, 0.5036] | 0.5111 [0.4936, 0.5295] | **0.4813 [0.4552, 0.5068]** | **0.1671 [0.1560, 0.1784]** |

*Interpretation:* The 95% bootstrap intervals for mAP@0.5:0.95 exhibit non-overlapping separation between YOLO26n, YOLO26s, and YOLO26m, indicating that capacity scaling provides consistent localization improvements in this evaluation.

---

## 8. Objective 8 — Target-Domain Confidence Threshold Sensitivity ($\tau$)

### Authoritative Target Table across 31,769 Primary OHRC Tiles (`results/confidence_analysis/target_domain_threshold_sensitivity.csv`):

| Threshold $\tau$ | YOLO26n (Dets / Rate) | YOLO26s (Dets / Rate) | YOLO26m (Dets / Rate) | YOLOv8n (Dets / Rate) | YOLOv5s (Dets / Rate) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 5,084 / 6.72% | 14,102 / 6.45% | 1,484 / 1.39% | 353,427 / 44.47% | 24,452 / 12.26% |
| **0.25** | 2,023 / 3.01% | 8,498 / 3.87% | 871 / 0.88% | 221,691 / 39.91% | 17,049 / 8.14% |
| **0.30** | 805 / 1.35% | 5,218 / 2.50% | 543 / 0.55% | 140,902 / 35.31% | 12,047 / 5.67% |
| **0.35** | 322 / 0.58% | 3,206 / 1.71% | 321 / 0.34% | 88,721 / 30.38% | 8,558 / 4.30% |
| **0.40** | 136 / 0.26% | 1,879 / 1.16% | 189 / 0.22% | 54,731 / 25.42% | 5,937 / 3.37% |
| **0.45** | 50 / 0.09% | 1,025 / 0.77% | 95 / 0.13% | 32,175 / 20.23% | 3,950 / 2.74% |
| **0.50** | 18 / 0.03% | 484 / 0.45% | 45 / 0.08% | 17,657 / 15.11% | 2,472 / 2.23% |
| **0.60** | 2 / 0.01% | 63 / 0.11% | 10 / 0.02% | 3,802 / 6.07% | 599 / 1.08% |
| **0.70** | 0 / 0.00% | 2 / 0.01% | 0 / 0.00% | 369 / 0.98% | 31 / 0.09% |
| **0.80** | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 13 / 0.04% | 0 / 0.00% |
| **0.90** | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |

---

## 9. Objective 9 — Target-Domain Performance Summary

### A. Primary OHRC Mosaic (31,769 Usable Tiles, $\tau = 0.20$)
- **YOLO26m:** 1,484 candidate detections across 443 positive tiles (**1.39%** positive-tile rate); mean confidence $0.293 \pm 0.088$.
- **YOLO26s:** 14,102 candidate detections across 2,050 positive tiles (**6.45%**); mean confidence $0.2954 \pm 0.087$.
- **YOLO26n:** 5,084 candidate detections across 2,134 positive tiles (**6.72%**); mean confidence $\approx 0.2543$.
- **YOLOv5s:** 24,452 candidate detections across 3,896 positive tiles (**12.26%**); mean confidence $0.3302 \pm 0.106$.
- **YOLOv8n:** 353,427 candidate detections across 14,129 positive tiles (**44.47%**); mean confidence $0.3036 \pm 0.093$.

### B. Ultra-Deep Polar OHRC Mosaic (13,906 Usable Tiles across 20 Products, $\tau = 0.20$)
- **YOLO26m:** 1,447 detections across 309 positive tiles (**2.22%** positive-tile rate); mean confidence $0.3057 \pm 0.0924$.
- **YOLO26s:** 8,128 detections across 2,124 positive tiles (**15.27%**); mean confidence $0.2699 \pm 0.0708$.
- **YOLO26n:** 10,085 detections across **1,173** positive tiles (**8.44%**); mean confidence $0.2907$.
- **YOLOv5s:** 7,202 detections across 1,779 positive tiles (**12.79%**); mean confidence $0.3214$.
- **YOLOv8n:** 71,001 detections across 6,047 positive tiles (**43.48%**); mean confidence $0.2959$.

---

## 10. Objective 10 — Controlled Histogram Matching (HM vs. No-HM) Ablation

### Authoritative Controlled Split
- **Training Set:** 4,379 images / 122,537 boulder instances
- **Validation Set:** 697 images / 24,303 instances
- **Test Set:** 262 images / 7,268 instances
- **Total Combined Dataset:** 5,338 images / 154,108 ground truth instances

### Full Comparison Table (`results/paper_tables/table_hm_vs_nohm_full.csv`):

| Model | Condition | Source Test mAP@0.5 | Source Test mAP@0.5:0.95 | Source Test F1 | OHRC Candidate Detections ($\tau=0.20$) | OHRC Positive Tiles | OHRC Positive-Tile Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | HM | 0.5292 | 0.1896 | 0.5442 | 5,084 | 2,134 | 6.72% |
| **YOLO26n** | No-HM | 0.6736 | 0.2664 | 0.6515 | 2,429,131 | 26,956 | 84.85% |
| **YOLO26s** | HM | 0.6023 | 0.2308 | 0.5962 | 14,102 | 2,050 | 6.45% |
| **YOLO26s** | No-HM | NA | NA | NA | NA | NA | NA |
| **YOLO26m** | HM | 0.6463 | 0.2622 | 0.6275 | 1,484 | 443 | 1.39% |
| **YOLO26m** | No-HM | NA | NA | NA | NA | NA | NA |
| **YOLOv8n** | HM | 0.4813 | 0.1671 | 0.5111 | 353,427 | 14,129 | 44.47% |
| **YOLOv8n** | No-HM | 0.6455 | 0.2488 | 0.6316 | 2,448,714 | 26,755 | 84.22% |
| **YOLOv5s** | HM | 0.5737 | 0.2121 | 0.5788 | 24,452 | 3,896 | 12.26% |
| **YOLOv5s** | No-HM | 0.7122 | 0.2908 | 0.6821 | 3,356,769 | 30,152 | 94.91% |

### Scientifically Rigorous Interpretation:
> *Histogram matching decreases source-domain performance in this controlled comparison, while substantially changing target-domain candidate activation density. Because the target images are unlabeled, this does not demonstrate improved target-domain detection accuracy. Rather, histogram matching functions as a zero-parameter photometric alignment baseline that mitigates extensive candidate over-activation across background terrain textures.*

---

## 11. Objective 11, 12, 13 & 14 — Manuscript Auditing & IEEE Compliance Text

### A. Manuscript Location Status
- The manuscript LaTeX files (`conference_101719.tex` and compiled PDF) reside on the author's separate writing environment (`/mnt/data/paper_work/`) and are not checked into this git repository.
- Below are the exact, verified IEEE-compatible LaTeX sections prepared for immediate insertion into the manuscript.

### B. Ready-to-Copy IEEE Compliance Text Blocks

#### 1. Data Availability Statement
```latex
\section*{Data Availability Statement}
The source-domain training datasets (LROC NAC boulder annotations from RMaM-2020 and Prieur et al.) are publicly available through the Lunar Reconnaissance Orbiter Camera data archive and respective publisher repositories. Unlabeled Chandrayaan-2 Orbiter High Resolution Camera (OHRC) calibrated PDS4 data products were retrieved from the Indian Space Science Data Centre (ISSDC) PRADAN portal (\url{https://pradan.issdc.gov.in}). The tile coordinate manifest, bounding-box caches, and evaluation summary tables are preserved in the project repository.
```

#### 2. Code Availability Statement
```latex
\section*{Code Availability Statement}
All preprocessing scripts, training configurations, bootstrap resampling algorithms, and evaluation pipelines supporting the findings of this study are openly available on GitHub at \url{https://github.com/nandana-das/chandrayaan-2-ohrc-boulder-detection}.
```

#### 3. Author Contributions
```latex
\section*{Author Contributions}
\textbf{Nandana Narayan Das}: Conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing---original draft, visualization.
\textbf{R C Karpagalakshmi}: Conceptualization, supervision, resources, writing---review and editing.
```

#### 4. Conflicts of Interest
```latex
\section*{Conflicts of Interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
```

#### 5. Funding Statement
```latex
\section*{Funding Statement}
No external funding was received for this work.
```

#### 6. Cross-Paper Comparison Caveat (Addressing References [14] Chainani and [15] Shah)
```latex
\section*{Discussion on Cross-Paper Comparability}
A direct quantitative comparison between the detection metrics reported here and those published by Chainani et al. [14] or Shah et al. [15] is methodologically invalid. Those investigations utilized distinct datasets, non-overlapping geographic regions, differing spatial resolutions, different bounding-box annotation protocols, and task definitions that preclude mathematically rigorous cross-paper benchmark alignment. Consequently, our performance evaluations are strictly confined to controlled, identically trained architectural baselines on the unified benchmark splits established in this study.
```

#### 7. Methodological Batch Hardware Allocation Note
```latex
\section*{Training Configuration and Batch Normalization Allocation}
All Stage-2 models were trained end-to-end for 50 epochs on the combined source dataset ($640 \times 640$ pixels, freeze = 0) using the AdamW optimizer ($\text{lr}_0 = 10^{-4}$, cosine decay to $\text{lrf} = 0.01$, 3-epoch warmup). To comply with physical VRAM limits on the host NVIDIA RTX 3050 Laptop GPU (4.0 GB physical VRAM), physical mini-batch size 8 was assigned to YOLO26n, YOLO26s, YOLOv8n, and YOLOv5s, whereas YOLO26m utilized physical batch size 4 (as physical batch 8 required 4.70 GB VRAM, causing out-of-memory errors). In the Ultralytics engine, gradient accumulation is coupled to nominal batch size ($\text{nbs} = 64$) via $\text{accumulate} = \max(\text{round}(\text{nbs} / \text{batch}), 1)$. Thus, models with physical batch 8 accumulated over 8 steps ($8 \times 8 = 64$), while YOLO26m accumulated over 16 steps ($4 \times 16 = 64$). All five models executed optimizer updates with an identical effective batch size of 64 images. The physical mini-batch difference may still affect batch-dependent behavior, reflecting practical edge-workstation deployment constraints.
```

#### 8. Exclusion of External Roboflow Benchmark
- Confirmed: The external Roboflow benchmark dataset is **deliberately excluded** from the main manuscript text and tables because its collection conditions, labelling criteria, and task formulation differ substantially from the lunar boulder benchmark.

---

## 12. Exact List of Files Modified in this Audit Pass

1. **`requirements.txt`**
   - **Old:** `ultralytics>=8.0`
   - **New:** `ultralytics==8.4.173  # Historical training executed under Ultralytics 8.4.155; evaluation/audit verified on 8.4.173`
   - **Rationale:** Pins exact evaluated Ultralytics version while documenting the historical training version recovered from logs.

2. **`results/confidence_analysis/threshold_sensitivity_summary.md`**
   - **Old:** Mentioned ~1.83% positive tile rate for YOLO26m; used "catastrophic spatial saturation" and "genuine geomorphic structures".
   - **New:** Corrected to 1.39% positive tile rate; replaced overclaiming language with neutral descriptors ("excessive spatial candidate density", "prominent local contrast features").

3. **`scripts/experiments/run_target_threshold_sensitivity.py`**
   - **Old:** Generated prose with 1.83% and overclaiming phrases ("catastrophic spatial saturation", "genuine geomorphic structures").
   - **New:** Corrected prose generation to 1.39% and neutral wording.

4. **`results/paper_tables/test_metrics_bootstrap_summary.md`**
   - **Old:** Prose contained stale CI numbers (`[0.4912, 0.5647]`, `[0.5701, 0.6340]`) and claimed "confirming statistically significant separation in detection performance".
   - **New:** Updated prose to match authoritative table values (`[0.5030, 0.5551]`, `[0.5774, 0.6286]`); replaced with "the bootstrap intervals indicate clear separation in this evaluation".

5. **`scripts/experiments/compute_bootstrap_ci.py`**
   - **Old:** Script template generated stale numbers and "statistically significant separation".
   - **New:** Aligned script template with authoritative table values and approved non-overclaiming wording.

6. **`results/paper_tables/stage2_reproducibility_audit.md`**
   - **Old:** `- **Physical batch difference:** The only divergence is the physical mini-batch size (4 vs 8) affecting BatchNorm statistics during forward passes.`
   - **New:** `- **Physical batch difference:** The YOLO26m run used a smaller physical mini-batch because of the 4-GB GPU memory constraint; the framework's accumulation schedule maintained an effective optimizer batch size of 64. The physical mini-batch difference may still affect batch-dependent behavior.`

7. **`scripts/paper_generation/audit_reproducibility.py`**
   - **Old:** Generated the divergence wording above.
   - **New:** Updated script generator to output the approved technically accurate batch wording.

8. **`results/paper_tables/yolo26_scale_report.md`**
   - **Old:** Stale parameter references ("10.01M", "21.90M") in executive summary; divergence batch statement.
   - **New:** Updated to authoritative consolidated parameters ("9.949M", "21.774M") noting stale un-consolidated figures; replaced batch statement with approved wording.

9. **`results/paper_tables/paper_results_generation_report.md`**
   - **Old:** Divergence batch statement.
   - **New:** Replaced with approved technically accurate batch wording.

10. **`results/paper_tables/phase2b_rigor_integration_report.md`**
    - **Old:** Contained "catastrophic target-domain false-triggering", "catastrophic over-activation", "hyper-activates", "genuine geomorphic structures", and divergence batch wording.
    - **New:** Replaced all terms with neutral descriptors ("extensive target-domain candidate over-activation", "extensive candidate activation", "prominent local contrast features") and approved batch wording.

11. **`results/paper_tables/table_hm_vs_nohm_full.md` & `scripts/paper_generation/generate_hm_vs_nohm_table.py`**
    - **Old:** "preventing widespread background false-triggering".
    - **New:** "mitigating extensive candidate over-activation across background terrain".

12. **`scripts/train_rtdetr_pipeline.py` & `demo/index.html`**
    - **Old:** "multi-reference histogram matching".
    - **New:** "20-tile averaged-reference histogram matching".

13. **`results/paper_tables/final_paper_reproducibility_audit.md`**
    - **Created:** Consolidated final audit document fulfilling all 15 Main Objectives.

---

## 13. Retraining Decision

**NO RETRAINING REQUIRED.**  
All 5 model checkpoints (`weights/` and `runs/`), evaluation caches, bootstrap resamples, threshold sensitivity logs, and mosaic inference catalogs are complete, verified, and internally consistent. Retraining would consume significant compute without altering or improving the scientific conclusions.
