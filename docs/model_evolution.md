# Model evolution — from a per-window Siamese baseline to the final model

The headline model in this repository (`siamese_hazard/`) is the end point of a
long line of iterations. Each iteration kept the **Siamese paired-methylation
core** and changed one thing: how risk is defined, how it is calibrated, or how
CpGs are projected into the latent space. The internal development snapshots were
numbered `v1 … v9`; below we describe what each contributed and give the final
model a descriptive name rather than a version number.

| Stage | Internal tag | One-line contribution |
|------:|:-------------|:----------------------|
| 1 | `v1` | Per-window Siamese baseline (PCA features, BCE + ranking) |
| 2 | `v2` | Multi-horizon **monotone discrete hazards** + calibration |
| 3 | `v3`–`v5` | Robust **per-head calibration** (direction-aware isotonic) |
| 4 | `v6` | **Learnable Platt** calibration layer (replaces isotonic) |
| 5 | `v7` | **Context-aware projection** `V = g(E)` of CpGs |
| 6 | **`v8` → final** | **Residual context projector** `V = g(E)+α·W_free` + context preprocessing + mean-AUC selection |
| — | `v9` (explored) | Dual-path late-fusion projector — evaluated, **not** selected |

## Stage 1 — Per-window Siamese baseline (`v1`)

The first model trained a Siamese MLP on paired `(t0, t1)` methylation to predict
AD conversion within a **single** window (2y / 3y / 5y trained separately). Per
fold it applied PCA-128 to stacked train samples (no leakage), built embeddings
`h0/h1`, used the distance `|h1−h0|` plus `Δt`, and optimised BCE with a pairwise
hinge ranking term. This established the paired-sample idea but treated the three
horizons as independent problems and had no guarantee of risk monotonicity.

## Stage 2 — Multi-horizon monotone hazards (`v2`)

Risk was redefined as a **discrete-time hazard** problem with three conditional
heads (`1→2y`, `2→3y`, `3→5y`) composed into cumulative risks with **monotonicity
by design** (`p3 ≥ p2`, `p5 ≥ p3`). This is the formulation the project keeps to
this day. It also introduced calibrated hazards and decision-curve reporting, and
framed the central scientific question: *does within-person change add value over
a single baseline draw?*

## Stage 3 — Robust per-head calibration (`v3`–`v5`)

Short-horizon heads could collapse to a constant when a hazard head was
direction-reversed on a small fold. Stage 3 added **per-head isotonic
calibration** that automatically flips a head (`1 − p`) when its validation AUC
falls below 0.5, and persisted the fitted calibrators for reproducibility. This
made the 2y head stable across folds.

## Stage 4 — Learnable Platt calibration (`v6`)

Post-hoc isotonic calibration was replaced with a tiny **learnable Platt layer**
fit per fold on the validation set only: `logit' = a·logit + b` per head, by
LBFGS, lightly regularised toward `a≈1, b≈0`. This is the calibration mechanism
shipped in the final model (`calibration_layer.py`).

## Stage 5 — Context-aware projection (`v7`)

Instead of a generic PCA / free linear map, Stage 5 generated the CpG→latent
projection **from per-CpG biology**: each CpG has a fixed context vector `e_i`
(sequence, gene/regulatory, SNP annotations), a small network `g` produces a
projection row `V_i = g(e_i)`, and `z = X @ V`. This is the "context" idea —
methylation is read through an annotation-aware lens, which both helps
performance and makes the latent space interpretable.

## Stage 6 — The final model (`v8`)

The final model is a **residual context projector** plus disciplined context
preprocessing and aggregate model selection:

- **Residual projector:** `V = g(E) + α · W_free`. The context map `g(E)` is the
  inductive prior; a learned residual `W_free` (scaled by a learned `α`) lets the
  model correct the projection where biology alone is insufficient. Pure context
  projection is still available via `--projector_type context_only`.
- **Context preprocessing:** continuous annotation features are z-scored and each
  annotation **block** is scaled by `1/√(block size)` so no single feature family
  dominates the projection (`context.load_context_csv`).
- **Selection by `mean_AUC_2y3y5y`:** checkpoints are chosen on the mean AUC over
  all three horizons (with patience-based early stopping), rather than the 2y head
  alone.

This is the model packaged in `siamese_hazard/` and benchmarked in `results/`.

## Explored but not selected — dual-path fusion (`v9`)

A dual-path late-fusion projector (separate free and context paths combined by
sum / concat / gate) was implemented and swept. It did not beat the residual
projector on the locked cross-validation protocol, so the residual model remains
the headline. The negative result is part of the record, not a regression.
