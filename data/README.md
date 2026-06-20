# Data contract

**The data is not distributed with this repository.** The DNA-methylation
matrices used here are derived from the ADNI cohort and are governed by the
[ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/access-data/).
To run the model you must obtain the source data yourself and materialise the
files described below into this `data/` directory.

## Cohort

Paired-sample MCI subjects with two DNA-methylation draws per person
(`t0`, `t1`, separated by `dt ≈ 11–13` months). Labels indicate first
conversion to AD within 2-, 3-, and 5-year windows; censored subjects without a
usable conversion status are removed.

## Required files

Six gzipped design tables (top-50,000 CpG universe), aligned row-for-row
between `t0` and `t1` within each horizon window:

```
data/design_2y_t0_top50000.csv.gz   data/design_2y_t1_top50000.csv.gz
data/design_3y_t0_top50000.csv.gz   data/design_3y_t1_top50000.csv.gz
data/design_5y_t0_top50000.csv.gz   data/design_5y_t1_top50000.csv.gz
```

Each design table has:

| column        | meaning                                                        |
|---------------|----------------------------------------------------------------|
| `RID`         | subject id (identical, same order, in the matching t0/t1 file) |
| `y`           | binary conversion label for that window                        |
| `t1_dnam_m`   | elapsed time to the second draw (months) — the model's `dt`    |
| `window`      | which horizon table the row came from                          |
| `status`      | source disease status                                          |
| `cpg_<cgID>`  | 50,000 columns of β-valued methylation (`cpg_cg00000029`, …)   |

β-values are converted to M-values internally (`data.beta_to_m`).

## Context matrix (the "E" file)

One CSV with **one row per CpG** and an id column named `cpg` (bare `cgID`, or
`cpg_`-prefixed — both are accepted), followed by per-CpG annotation features
(sequence composition, gene/regulatory overlap, SNP metadata, …). It must cover
every CpG in the design matrix. This matrix is a *hyperparameter* of the
pipeline (see `docs/model_contract.md`) — freeze one file for headline runs.

```
data/cpg_context.csv     # cpg,seq_gc_20bp,ucsc_n_genes,dnase_log1p_count,...
```

The loader (`context.load_context_csv`) aligns rows to the design CpG order,
z-scores the continuous features, and applies inverse-sqrt block scaling.
