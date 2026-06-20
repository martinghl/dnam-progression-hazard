import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

WINDOWS = ["2y","3y","5y"]

def beta_to_m(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    beta = np.clip(beta, eps, 1.0 - eps)
    return np.log2(beta / (1.0 - beta))

def load_pair(path_t0: str, path_t1: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = pd.read_csv(path_t0, low_memory=False)
    t1 = pd.read_csv(path_t1, low_memory=False)
    if "RID" not in t0.columns or "RID" not in t1.columns:
        raise ValueError("RID column missing in design files.")
    if (t0["RID"].values != t1["RID"].values).any():
        raise ValueError("RID mismatch between t0 and t1.")
    if "y" in t0.columns and "y" in t1.columns:
        if (t0["y"].values != t1["y"].values).any():
            raise ValueError("y mismatch between t0 and t1.")
    return t0, t1

def get_cpg_cols(df: pd.DataFrame) -> List[str]:
    # Prefer the explicit prefix used in your pipeline
    cpg = [c for c in df.columns if c.startswith("cpg_")]
    if not cpg:
        # fallback: treat all non-meta as CpGs
        meta_like = {"RID","window","status","y","t1_dnam_m"}
        cpg = [c for c in df.columns if c not in meta_like]
    return cpg

def load_all_windows(data_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    out = {}
    for w in WINDOWS:
        t0_path = os.path.join(data_dir, f"design_{w}_t0_top50000.csv.gz")
        t1_path = os.path.join(data_dir, f"design_{w}_t1_top50000.csv.gz")
        if not (os.path.exists(t0_path) and os.path.exists(t1_path)):
            raise FileNotFoundError(f"Missing design files for {w}: {t0_path} / {t1_path}")
        t0, t1 = load_pair(t0_path, t1_path)
        out[w] = {"t0": t0, "t1": t1}
    return out

def build_master_table(allw: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for w in WINDOWS:
        df = allw[w]["t0"][["RID","y"]].copy()
        df = df.rename(columns={"y": f"y_{w}"})
        frames.append(df.set_index("RID"))
    ytbl = pd.concat(frames, axis=1)

    # Choose a source window for features: prefer the longest available label (5y > 3y > 2y)
    src = pd.Series(index=ytbl.index, dtype="object")
    for w in ["5y","3y","2y"]:
        src = src.fillna(w).where(ytbl[f"y_{w}"].notna(), src)
    ytbl["src_window"] = src

    return ytbl.reset_index()

def assemble_features(allw: Dict[str, Dict[str, pd.DataFrame]], master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Use t0/t1 rows from each subject's chosen src_window.
    # Index each window once so lookup scales with the number of selected RIDs
    # instead of rescanning the full design table for every subject.
    rows0 = []
    rows1 = []
    order_col = "_master_order"

    for w in WINDOWS:
        mask = master["src_window"] == w
        if not mask.any():
            continue

        rids = master.loc[mask, "RID"]
        t0 = allw[w]["t0"].set_index("RID", drop=False)
        t1 = allw[w]["t1"].set_index("RID", drop=False)

        if t0.index.has_duplicates or t1.index.has_duplicates:
            raise ValueError(f"RID not unique in window {w}")

        try:
            row0 = t0.loc[rids].copy()
            row1 = t1.loc[rids].copy()
        except KeyError as e:
            raise ValueError(f"RID lookup failed in window {w}: {e}") from e

        if len(row0) != len(rids) or len(row1) != len(rids):
            raise ValueError(f"RID count mismatch while assembling window {w}")

        row0[order_col] = master.index[mask].to_numpy()
        row1[order_col] = master.index[mask].to_numpy()
        rows0.append(row0)
        rows1.append(row1)

    t0_sel = pd.concat(rows0, axis=0).sort_values(order_col).drop(columns=order_col).reset_index(drop=True)
    t1_sel = pd.concat(rows1, axis=0).sort_values(order_col).drop(columns=order_col).reset_index(drop=True)

    # sanity: aligned
    if (t0_sel["RID"].values != t1_sel["RID"].values).any():
        raise ValueError("Assembled t0/t1 RID mismatch.")
    return t0_sel, t1_sel

def build_hazard_labels(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    # hazards
    m["h2"] = np.nan
    m["h3"] = np.nan
    m["h5"] = np.nan

    # h2 defined where y_2y exists
    m.loc[m["y_2y"].notna(), "h2"] = m.loc[m["y_2y"].notna(), "y_2y"]

    # h3 defined only for those with y2==0 and y3 known
    mask_h3 = (m["y_2y"] == 0) & (m["y_3y"].notna())
    m.loc[mask_h3, "h3"] = m.loc[mask_h3, "y_3y"]

    # h5 defined only for those with y3==0 and y5 known
    mask_h5 = (m["y_3y"] == 0) & (m["y_5y"].notna())
    m.loc[mask_h5, "h5"] = m.loc[mask_h5, "y_5y"]

    # masks
    m["mask_h2"] = m["h2"].notna().astype(int)
    m["mask_h3"] = m["h3"].notna().astype(int)
    m["mask_h5"] = m["h5"].notna().astype(int)
    return m
