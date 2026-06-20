import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

def _normalize_cpg_id(x: str) -> str:
    x = str(x)
    x = x.strip()
    x = x.replace("cpg_", "")
    return x

def load_context_matrix(
    context_csv: str,
    cpg_cols: List[str],
    id_col: Optional[str] = None,
    drop_non_numeric: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Load per-CpG context features and align them to the CpG column order.

    Parameters
    ----------
    context_csv : str
        CSV/TSV with one row per CpG. Must include an ID column (cg ID).
    cpg_cols : list[str]
        CpG feature columns from your design matrix (e.g., 'cpg_cg0123...').
    id_col : str, optional
        Column name containing the CpG ID. If None, tries common names.
    drop_non_numeric : bool
        If True, drops non-numeric columns. If False, one-hot encodes them.

    Returns
    -------
    E : np.ndarray
        Shape (P, C) float32 matrix aligned to cpg_cols order.
    feature_names : list[str]
        Names of context features.
    """
    # auto-detect delimiter
    sep = "," if context_csv.lower().endswith(".csv") else "\t"
    df = pd.read_csv(context_csv, sep=sep, low_memory=False)

    if id_col is None:
        for cand in ["cg", "probeID", "cpg", "Composite Element REF"]:
            if cand in df.columns:
                id_col = cand
                break
    if id_col is None or id_col not in df.columns:
        raise ValueError(f"Could not find CpG ID column in context file. Provide --context_id_col.")

    df = df.copy()
    df[id_col] = df[id_col].map(_normalize_cpg_id)

    # Align IDs
    cgs = [_normalize_cpg_id(c) for c in cpg_cols]
    df = df.set_index(id_col)

    # Select features
    feat_df = df.copy()
    # Drop obvious non-features if present
    for drop in ["chr","chrom","Chromosome","CpG_chrm","CpG_beg","CpG_end","pos","position","Genomic_Coordinate"]:
        if drop in feat_df.columns:
            feat_df = feat_df.drop(columns=[drop])

    if drop_non_numeric:
        feat_df = feat_df.select_dtypes(include=[np.number])
    else:
        # One-hot encode object/category cols; keep numeric as-is.
        obj_cols = [c for c in feat_df.columns if feat_df[c].dtype == "object"]
        if obj_cols:
            feat_df[obj_cols] = feat_df[obj_cols].fillna("NA")
            feat_df = pd.get_dummies(feat_df, columns=obj_cols, drop_first=False)
        # Coerce remaining to numeric where possible
        for c in feat_df.columns:
            if feat_df[c].dtype != np.number and feat_df[c].dtype != "float64" and feat_df[c].dtype != "int64":
                # leave
                pass

    # Reindex to CpG order, fill missing with 0
    feat_df = feat_df.reindex(cgs)
    feat_df = feat_df.fillna(0.0)

    E = feat_df.to_numpy(dtype=np.float32)
    return E, feat_df.columns.tolist()

def default_context(cpg_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Fallback context: a single constant feature (all ones).
    Not recommended for real use, but keeps the pipeline runnable."""
    E = np.ones((len(cpg_cols), 1), dtype=np.float32)
    return E, ["bias"]
