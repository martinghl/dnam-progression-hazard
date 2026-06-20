# Results

Curated, committed results so the headline findings are visible without
re-running anything. Full experiment trees (hundreds of GB of sweeps,
checkpoints and OOF tables) are kept outside the repository.

## External-method benchmark (headline)

Round-1 benchmark on the ADNI MCI→AD cohort (N=190; 18/36/48 events at 2/3/5y),
6 methods × 5 seeds × 5 folds on locked StratifiedKFold splits. The proposed
model is compared against four established survival learners adapted to the same
paired flat-feature input and a panel of methylation aging clocks.

- `external_benchmark_summary.md` — full paper-ready write-up (tables, tests,
  robustness across context files, limitations).
- `external_benchmark_headline.csv` — per-method per-horizon AUC/Brier table.
- `external_benchmark_master_summary.csv` — one row per method.
- `external_benchmark_per_seed.csv` — per-seed `mean_AUC_2y3y5y` (figure source).

**Top line:** the proposed model has the highest median `mean_AUC_2y3y5y`
(0.687) versus the strongest competitor (aging clocks, 0.665) and four survival
learners (best 0.655); point estimates favour it in 14/15 pairwise comparisons.
At N=190 no comparison reaches Holm-corrected significance — the honest framing
is "competitive with or better than every benchmark, with sample size limiting
power for definitive claims." The conclusion is stable across three context
files.

## Figures

- `figures/fig_sweep_overview_poster.png` — AUC/PR-AUC across the development sweeps.
- `figures/fig_forest_pairwise.png` — Δ AUC vs the proposed model, 95% paired-bootstrap CI.
- `figures/fig_seed_scatter.png` — per-seed `mean_AUC_2y3y5y` by method.
- `figures/fig_reliability_grid.png` — reliability diagrams, 6 methods × 3 horizons.
- `figures/fig_dca_5y.png` — decision-curve analysis at 5y.

## Revision analyses

`revision/` holds the reconciled headline tables used in manuscript revision:
representative-config bootstrap CIs, paired-bootstrap comparisons, context-feature
sensitivity, risk-tertile stratification, and the sweep summary.
