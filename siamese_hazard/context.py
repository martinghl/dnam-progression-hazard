import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CONTINUOUS_FEATURES = [
    "seq_gc_20bp",
    "seq_cpg_count_20bp",
    "ucsc_n_genes",
    "gencode_basic_n_genes",
    "gencode_comp_n_genes",
    "dnase_log1p_count",
    "openchrom_log1p_count",
    "tfbs_log1p_count",
    "snp_n",
    "snp_min_abs_distance",
    "snp_max_maf",
    "snp_mean_maf",
]

BLOCK_ORDER = ["Assay", "Sequence", "GenomicGene", "Regulatory", "SNPMeta", "Other"]


def _normalize_cpg_name(x: str) -> str:
    x = str(x).strip()
    if x.startswith("cpg_"):
        x = x[4:]
    return x


def _assign_block(feature_name: str) -> str:
    if (
        feature_name.startswith("design_")
        or feature_name.startswith("next_")
        or feature_name.startswith("color_")
        or feature_name == "mfg_change_flagged"
    ):
        return "Assay"
    if feature_name.startswith("seq_"):
        return "Sequence"
    if (
        feature_name.startswith("ucsc_")
        or feature_name.startswith("island_")
        or feature_name.startswith("gencode_basic_")
        or feature_name.startswith("gencode_comp_")
    ):
        return "GenomicGene"
    if (
        feature_name.startswith("reg_")
        or feature_name.startswith("dnase_")
        or feature_name.startswith("openchrom_")
        or feature_name.startswith("tfbs_")
        or feature_name in {
            "has_phantom4_enhancer",
            "has_phantom5_enhancer",
            "has_dmr",
            "has_450k_enhancer",
            "has_hmm_island",
        }
    ):
        return "Regulatory"
    if feature_name.startswith("snp_"):
        return "SNPMeta"
    return "Other"


def _validate_block_scaling(block_scaling: str) -> None:
    if block_scaling not in {"inv_sqrt_dim", "none"}:
        raise ValueError("block_scaling must be one of: inv_sqrt_dim, none")


def load_context_csv(
    context_csv: str,
    cpg_cols: List[str],
    cpg_col_name: str = "cpg",
    zscore_continuous: bool = True,
    block_scaling: str = "inv_sqrt_dim",
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """Load, align, and preprocess CpG context features."""
    _validate_block_scaling(block_scaling)

    df = pd.read_csv(context_csv)
    if cpg_col_name not in df.columns:
        raise ValueError(f"context_csv must contain column '{cpg_col_name}'")

    df[cpg_col_name] = df[cpg_col_name].map(_normalize_cpg_name)
    df = df.drop_duplicates(subset=[cpg_col_name]).set_index(cpg_col_name)

    target = [_normalize_cpg_name(c) for c in cpg_cols]
    feat_cols = [c for c in df.columns if c != cpg_col_name]
    if not feat_cols:
        raise ValueError("No context feature columns found in context_csv.")

    missing = [cg for cg in target if cg not in df.index]
    if missing:
        show = ", ".join(missing[:10])
        raise ValueError(
            f"context_csv missing {len(missing)} CpGs from design matrix (first 10: {show})."
        )

    E = (
        df.loc[target, feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    if np.isnan(E).any():
        E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        E = np.asarray(E, dtype=np.float32)

    feature_to_idx = {name: idx for idx, name in enumerate(feat_cols)}
    continuous_present = [name for name in CONTINUOUS_FEATURES if name in feature_to_idx]
    continuous_missing = [name for name in CONTINUOUS_FEATURES if name not in feature_to_idx]
    continuous_means: Dict[str, float] = {}
    continuous_stds: Dict[str, float] = {}

    for name in continuous_present:
        idx = feature_to_idx[name]
        col = E[:, idx].astype(np.float32, copy=False)
        mean = float(np.mean(col))
        std = float(np.std(col))
        continuous_means[name] = mean
        continuous_stds[name] = std
        if zscore_continuous:
            if np.isfinite(std) and std > 0:
                E[:, idx] = ((col - mean) / std).astype(np.float32)
            else:
                E[:, idx] = np.zeros_like(col, dtype=np.float32)

    block_cols: Dict[str, List[str]] = {block: [] for block in BLOCK_ORDER}
    for feature_name in feat_cols:
        block_cols[_assign_block(feature_name)].append(feature_name)

    block_scales: Dict[str, float] = {}
    for block_name, columns in block_cols.items():
        if not columns:
            block_scales[block_name] = 0.0
            continue
        if block_scaling == "inv_sqrt_dim":
            scale = 1.0 / math.sqrt(len(columns))
            idxs = [feature_to_idx[name] for name in columns]
            E[:, idxs] *= np.float32(scale)
            block_scales[block_name] = float(scale)
        else:
            block_scales[block_name] = 1.0

    preprocess_summary: Dict[str, object] = {
        "zscore_continuous": bool(zscore_continuous),
        "block_scaling": block_scaling,
        "continuous_whitelist": list(CONTINUOUS_FEATURES),
        "continuous_present": continuous_present,
        "continuous_missing": continuous_missing,
        "continuous_means": continuous_means,
        "continuous_stds": continuous_stds,
        "block_counts": {block: len(cols) for block, cols in block_cols.items()},
        "block_scales": block_scales,
    }
    return E.astype(np.float32, copy=False), feat_cols, preprocess_summary
