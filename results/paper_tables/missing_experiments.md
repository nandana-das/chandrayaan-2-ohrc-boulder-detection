# Missing Experiments & Ablation Audit

**Date:** 2026-10-03  
**Project:** Lunar OHRC Boulder & Rockfall Detection  
**Repository:** `nandana-das/chandrayaan-2-ohrc-boulder-detection`

---

## 1. Primary Ablation: Controlled No-HM vs. HM Ablation
- **Status:** **COMPLETED** (2026-10-03)
- **Protocol:** Exactly identical train (4,379), val (697), and test (262) image splits; identical AdamW optimizer, $lr_0 = 10^{-4}$, cosine schedule, 50 epochs, batch 8, resolution 640×640, device CUDA:0.
- **Artifacts Generated:**
  - Training runs: `runs/stage2_*_combined_nohm_cosine`
  - Test metrics: `results/nohm_ablation/test_metrics_nohm.csv`, `results/nohm_ablation/hm_vs_nohm_test_metrics.csv`
  - Publication tables: `results/paper_tables/table_hm_ablation.csv`, `results/paper_tables/table_hm_ablation.md`
  - Target behavior table: `results/paper_tables/table_hm_target_behavior.csv`
  - Publication figures: `results/paper_figures/fig_hm_vs_nohm_training_curves.png` & `.pdf`, `results/paper_figures/fig_hm_vs_nohm_target_detection_density.png` & `.pdf`
  - Comprehensive report: `results/paper_tables/hm_ablation_report.md`
- **Key Finding:** Isolating histogram matching demonstrates a classic domain-adaptation trade-off: No-HM produces higher held-out source test metrics (mAP50 0.645–0.712 vs 0.481–0.574 for HM), but triggers extreme target candidate over-activation across 31,769 unlabeled OHRC tiles (84%–95% positive tile rates; 2.4M–3.4M candidates). Histogram matching regularizes target candidate density by orders of magnitude (to 6.7%–44.5% positive tile rates; 5K–353K candidates).

---

## 2. Uncompleted / Future Potential Investigations
While the controlled No-HM vs. HM ablation is complete, this does **NOT** claim that the entire domain-adaptation problem is solved. The following potential future directions remain open:

1. **Target-Domain Ground Truth Benchmark:**
   - *Status:* Deferred / Not yet available.
   - *Description:* Because all 31,769 usable OHRC tiles and 13,906 ultra-deep polar tiles lack human ground-truth annotations, computing target accuracy, precision, recall, or F1 remains impossible. Establishing an expert-verified OHRC test set (e.g., 200–500 tiles) would provide direct target-domain mAP benchmarking.

2. **Feature-Level Domain Adaptation (DANN / Adversarial Alignment):**
   - *Status:* Deferred.
   - *Description:* The current methodology addresses domain shift at the pixel/input level via histogram matching. Feature-level domain adaptation (e.g., gradient reversal layers or adversarial domain discriminators) could be explored in future work.

3. **Generative Style Transfer (CycleGAN):**
   - *Status:* Deferred.
   - *Description:* Comparing 20-tile averaged histogram matching against generative image-to-image translation models represents a possible avenue for future exploration, though computationally more intensive.
