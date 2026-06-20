# The model contract

The scientific unit of this project is a single pipeline:

```
(X0, X1, dt, E)  ->  (p2, p3, p5)
```

calibrated, monotone cumulative risks of MCI→AD conversion at 2, 3 and 5 years.

## Inputs

- **`X0`, `X1`** — per-subject CpG methylation at two timepoints, β converted to
  M-values (`data.beta_to_m`). The CpG universe is the top-50,000 panel defined
  by the design files.
- **`dt`** — scalar elapsed time `t1_dnam_m` between the two draws (~11–13 mo).
- **`E`** — a fixed per-CpG context matrix (~50,000 rows × ~88 features). `E` is
  a **hyperparameter** of the pipeline, not a learned quantity; freeze it for
  headline comparisons.

## Computation

1. **Context projector.** A small network `g` turns the per-CpG context `E` into
   a projection matrix `V = g(E) + α · W_free` (the *residual* projector). Then
   `z = X @ V` gives a low-dimensional latent per sample. Setting `projector_type
   = context_only` drops `W_free` and recovers a pure context projection.
2. **Siamese hazard core.** `SiameseHazardNet` encodes `(z0, z1)` to `(h0, h1)`,
   forms `[h0, h1, |h1−h0|, dt]`, and emits three logits `l2, l3, l5` — the
   **conditional** interval hazards for `1→2y`, `2→3y`, `3→5y`. These are *not*
   final risks.
3. **Per-fold Platt calibration.** `MultiHeadPlattCalibrator` (LBFGS) rescales
   each logit to a calibrated probability on the validation fold only.
4. **Composition.** Cumulative risks are composed from calibrated hazards:
   `p2 = r12`, `p3 = 1−(1−r12)(1−r23)`, `p5 = 1−(1−r12)(1−r23)(1−r35)`, with
   monotonicity enforced (`p3 ≥ p2`, `p5 ≥ p3`).

## Labels and masking

Labels are masked so each head trains only on its valid risk set: `h3` only on
subjects event-free at 2y (`y_2y == 0`), `h5` only on those event-free at 3y.
The loss is masked BCE per head (with `pos_weight` class balancing) plus an
optional semi-hard triplet loss on the projected embeddings.

## Cross-validation

5-fold `StratifiedKFold` on `y_3y` (NaN→0). Standardization is fit on stacked
`X0/X1` **within each training fold** — no fold leakage. Platt calibration is fit
on the validation fold only. OOF predictions are concatenated across folds.

## Metrics

Per horizon (2y/3y/5y): ROC-AUC, PR-AUC, Brier, ECE (uniform-bin), and Decision
Curve Analysis. `mean_AUC_2y3y5y` is the project's standard aggregate selection
metric and is also used for checkpoint selection (`--stop_metric mean_auc_235`).
