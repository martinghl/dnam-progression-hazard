# Methods

## Cohort

We used the Alzheimer's Disease Neuroimaging Initiative (ADNI) DNA methylation cohort with paired blood draws. The cohort comprises 190 unique subjects with mild cognitive impairment (MCI) at baseline, of whom 48 progressed to Alzheimer's disease (pMCI) and 77 remained stable (sMCI) within the 5-year horizon (the remainder were censored before 5y or had only earlier-horizon labels). Two whole-blood DNA methylation profiles were measured per subject: a baseline draw (`t0`) and a follow-up draw (`t1`) approximately 12 months later (mean 12.65 ± 0.90 months, range 9.6–14.7). Per-horizon labels and event counts are given in Table 1. Both methylation profiles for each subject were measured on the Illumina EPIC array; the analysis was restricted to the same pre-selected 50,000-CpG universe used in prior model-development work on this cohort. Pre-selection was performed at the cohort level and is therefore inherited identically by every method compared here.

## Outcome and label structure

For each subject and horizon h ∈ {2y, 3y, 5y} we have a binary label `y_h` indicating MCI→AD conversion within h years from baseline, or NaN if censored before h. We additionally use the per-head training masks defined in Banerjee et al. (2024; v8 model): the 2-year head trains on all subjects with `y_2y` observed; the 3-year head trains on subjects with `y_2y == 0` and `y_3y` observed; the 5-year head trains on subjects with `y_3y == 0` and `y_5y` observed. This progressive risk-set masking aligns with the conditional discrete-time hazard formulation.

For methods that require a single survival representation (Random Survival Forest, DeepSurv, LogisticHazard), we encoded each subject's labels as a single (event, time) pair at horizon resolution: time = the smallest horizon at which an event was observed (24, 36, or 60 months), or the latest observed-and-censored horizon otherwise. Subjects with all NaN labels were dropped from these methods only.

## Proposed model (`model_v8`)

The proposed paired interval-hazard model takes (X0, X1, dt) as input and produces calibrated cumulative risks (p2, p3, p5). It comprises four components:

1. **Per-CpG context-aware projector.** A per-CpG annotation matrix E ∈ ℝ^{50000 × ~88} (genomic, regulatory, and assay-design features) is mapped through a shared MLP `g(·)` to produce a 64-dim projector V = g(E) + α·W_free, where α=0.05 is a learned scale and W_free is a 50,000 × 64 free residual initialized with N(0, 0.01). The same V projects both timepoints: z_t = X_t @ V (M-values input).
2. **Siamese hazard tower.** A two-layer MLP (256→128→128) maps z_t to an embedding h_t. The hazard input is [h_0, h_1, |h_1 − h_0|, dt] which is fed to three independent two-layer MLP heads producing per-horizon logits l2, l3, l5.
3. **Loss.** Per-head masked binary cross-entropy with class-balancing pos_weights, plus a 0.1-weighted semi-hard triplet loss on dt-conditioned anchors over normalized projection embeddings.
4. **Post-hoc Platt calibration.** A two-parameter affine map (a_h, b_h) per head is fit by L-BFGS on the validation fold's logits with L2 regularization toward the identity. Calibrated logits are sigmoid-transformed and composed into cumulative risks (p2, p3, p5) following 1 − ∏(1 − r_h), with monotonicity enforced.

Hyperparameters (locked across all runs): proj_dim=64, ctx_hidden=32, emb_dim=128, dropout=0.1, lr=3e-4, batch_size=32, max_epochs=50, model selection by mean validation AUC over the three horizons.

## Comparator methods (paired-input adapter)

Five comparator methods were trained and evaluated on identical fold splits with identical labels. To give each comparator the same paired information as the proposed model, every tabular method consumed the **paired-flat-feature view**:

```
F = [PCA50(X0), PCA50(X1), PCA50(X1 − X0), dt]   ∈ ℝ^151
```

Standardization (z-score) and PCA were fit on the training fold only (stacked X0 and X1 rows), then applied to validation. We chose 50 components empirically; methylation array signal is heavily redundant and PCA-50 retains the bulk of training-fold variance while keeping every method tractable at N≈150 train rows per fold.

The comparators were:

- **Penalized discrete-time logistic regression** — three independent elastic-net logistic regressions (l1_ratio=0.5), one per horizon, with 3-fold inner-CV selection of `C` ∈ {0.1, 1.0, 10.0}, class-weight balancing, masked to the same per-head risk sets used by `model_v8`.
- **Random Survival Forest** (`scikit-survival 0.27`) — 200 trees, min_samples_leaf=5, default split. Cumulative hazard at t ∈ {24, 36, 60} months yields per-horizon risks via 1 − exp(−H(t)).
- **DeepSurv** (`pycox 0.3`) — Cox proportional-hazards MLP (64-64), dropout 0.1, Adam lr=1e-3, 50 epochs. Per-horizon risks via 1 − S(t) at t = 24/36/60.
- **LogisticHazard** (`pycox 0.3`) — discrete-time MLP (64-64), dropout 0.1, with bin cuts at [0, 24, 36, 60]. Native discrete hazards at the three cuts are composed identically to the proposed model.

In addition, we included a **methylation-aging-clock** baseline (Group B) that does not consume the projection but instead computes published epigenetic clocks per subject:

- **Aging-clock logistic regression** (`pyaging 0.1.30`) — a panel of six clocks computed at both timepoints from the original beta-values: pcgrimage, pchorvath2013, pcphenoage, grimage, dnamphenoage, dunedinpace (the original Horvath-2013, Hannum, and SkinAndBlood clocks were excluded after a coverage check on the 50,000-CpG subset gave clearly implausible mean ages; pyaging applies cohort-mean imputation for missing CpGs internally). Features [clock_t0, clock_t1, Δclock, dt] are passed to one logistic regression per horizon (l2 penalty, C=1.0, class-balanced, standardized).

All comparators (Group A and B) received the same per-fold post-hoc Platt calibration as the proposed model: a two-parameter affine map fit by L-BFGS on validation-fold logits-of-probabilities with L2 regularization toward the identity.

## Cross-validation and statistical analysis

We used 5-fold StratifiedKFold cross-validation stratified on `y_3y` (NaN→0), repeated across 5 random seeds {42, 43, 44, 45, 46}, yielding 25 fold instances per method. The fold dictionary was materialized once and shared across all methods (saved at `fold_index/folds_manifest.csv`). All within-fold preprocessing — PCA, scalers, regularization-strength selection, calibration — was fit on training data only.

For each (method, seed, fold), we recorded out-of-fold (OOF) per-subject probabilities (raw and Platt-calibrated). Per-horizon metrics — ROC-AUC, PR-AUC, Brier score, ECE-15 — were computed on the concatenated OOF predictions for each (method, seed). The headline aggregation reports the median across the 5 seeds.

Pairwise comparisons against the proposed model used a subject-level paired bootstrap (1,000 resamples) on the calibrated OOF predictions, pooled across seeds and folds by averaging per-(RID, horizon) probabilities. The bootstrap diff distribution yielded median Δ AUC, 95% CI, and a two-sided p-value. We applied Holm correction across the five comparator-method comparisons within each horizon.

We additionally ran a robustness analysis on two alternate context-feature definitions (`cpg_context_aligned.csv`, `context_full_v2.csv`) by re-training the proposed model alone on each (with the same 25 fold instances). Comparator methods are unaffected by the context choice (their feature path does not depend on E), so we re-used their headline results for cross-context comparison.

## Power statement

With N=190 paired subjects and 18, 36, 48 events at 2y, 3y, 5y respectively, the smallest reliably-detectable Δ AUC at α=0.05 with paired bootstrap is approximately 0.05–0.07 at 5y and 0.08–0.12 at 2y. We treat AUC differences below these thresholds as exploratory.

## Computational environment

All experiments ran on a single machine with Python 3.11, PyTorch 2.5.1+cu121, scikit-learn, scikit-survival 0.27, lifelines 0.30.3, pycox 0.3, pyaging 0.1.30, and pandas 2.3.3, on an NVIDIA GPU (CUDA_VISIBLE_DEVICES=1). The full benchmark of six methods × five seeds × five folds = 150 cells completed in approximately 7 minutes wall-clock time; the alternate-context robustness analysis added approximately 9.5 minutes.

## Reproducibility

All fold assignments, run statuses, per-subject predictions (long-format `predictions.parquet`), per-fold metrics, and pairwise-bootstrap CIs are persisted to `experiments/external_method_benchmark_v2/ebv2_20260425/` and the alternate-context jobs `ebv2_robust_v7orig_20260425` and `ebv2_robust_fullv2_20260425`. Every figure and table number reported in the manuscript can be back-traced to a single row of those persisted files.

## Limitations

Limitations of this benchmark, as a basis for clinical translation claims, are:

1. **Single-cohort**, ADNI only; no external-cohort replication has been performed.
2. **Pre-selected 50,000-CpG universe** is cohort-level; methods do not re-derive the feature space per fold.
3. **Per-fold Platt calibration is fit on validation and evaluated on validation**, an established source of optimism in calibration metrics (Brier, ECE) and in joint-OOF AUC after concatenation across folds with different per-fold rescalings. Both raw and calibrated probabilities are reported.
4. **No mechanism-level ablations** were performed in this round; the proposed model is reported as a single bundle (paired input + context projector + hazard composition + calibration). Round-2 work is needed to attribute the gain to specific architectural ingredients.
5. **Sample size** (N=190; 18-48 events per horizon) limits statistical power for pairwise comparisons. The benchmark is best read as a relative-ranking exercise, not as a definitive significance claim.
