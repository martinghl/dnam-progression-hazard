from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .explain_common import (
    build_target_maps,
    cpg_norm_from_map,
    cpg_scores_from_map,
    parse_horizons,
    permutation_importance_proj,
    read_json_if_exists,
    validate_state_load,
    write_panel_outputs,
)
from .calibration_layer import MultiHeadPlattCalibrator
from .context import load_context_csv
from .data import (
    assemble_features,
    beta_to_m,
    build_hazard_labels,
    build_master_table,
    get_cpg_cols,
    load_all_windows,
)
from .model import ContextSiameseHazardNet


def fit_scaler_on_train(X0_tr, X1_tr):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(np.vstack([X0_tr, X1_tr]))
    return scaler


def apply_scaler(X, scaler):
    return scaler.transform(X).astype(np.float32)


def _first_present(*values, default=None):
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _resolve_required(args: argparse.Namespace, results_subdir: Path) -> dict:
    run_config = read_json_if_exists(results_subdir / "run_config.json")
    context_summary = read_json_if_exists(results_subdir / "context_preprocess_summary.json")
    cli_cfg = run_config.get("cli_args", {})

    resolved = {
        "data_dir": _first_present(args.data_dir, cli_cfg.get("data_dir")),
        "context_csv": _first_present(args.context_csv, context_summary.get("context_csv"), cli_cfg.get("context_csv")),
        "context_id_col": _first_present(args.context_id_col, context_summary.get("context_id_col"), cli_cfg.get("context_id_col"), default="cpg"),
        "folds": int(_first_present(args.folds, cli_cfg.get("folds"), default=5)),
        "seed": int(_first_present(args.seed, cli_cfg.get("seed"), default=42)),
        "proj_dim": int(_first_present(args.proj_dim, cli_cfg.get("proj_dim"))),
        "ctx_hidden": int(_first_present(args.ctx_hidden, cli_cfg.get("ctx_hidden"))),
        "emb_dim": int(_first_present(args.emb_dim, cli_cfg.get("emb_dim"), default=128)),
        "dropout": float(_first_present(args.dropout, cli_cfg.get("dropout"), default=0.1)),
        "ctx_dropout": float(_first_present(args.ctx_dropout, cli_cfg.get("ctx_dropout"), default=0.0)),
        "projector_type": str(_first_present(args.projector_type, cli_cfg.get("projector_type"), default="residual")),
        "residual_alpha_init": float(_first_present(args.residual_alpha_init, cli_cfg.get("residual_alpha_init"), default=0.05)),
        "free_weight_init_std": float(_first_present(args.free_weight_init_std, cli_cfg.get("free_weight_init_std"), default=0.01)),
        "context_zscore_continuous": bool(int(_first_present(args.context_zscore_continuous, int(context_summary.get("zscore_continuous", 1)), cli_cfg.get("context_zscore_continuous"), default=1))),
        "context_block_scaling": str(_first_present(args.context_block_scaling, context_summary.get("block_scaling"), cli_cfg.get("context_block_scaling"), default="inv_sqrt_dim")),
    }

    missing = [key for key in ["data_dir", "context_csv", "proj_dim", "ctx_hidden"] if not resolved.get(key)]
    if missing:
        raise ValueError(
            f"Unable to resolve required settings {missing}. Pass them explicitly or ensure {results_subdir / 'run_config.json'} exists."
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute CpG-level latent importance for a completed siamese_hazard run.")
    ap.add_argument("--data_dir")
    ap.add_argument("--context_csv")
    ap.add_argument("--context_id_col")
    ap.add_argument("--model_dir", required=True, help="Run model folder containing siamese_hazard/fold_k/best.ckpt.")
    ap.add_argument("--results_dir", required=True, help="Run results folder containing siamese_hazard/calibrators_platt/...")
    ap.add_argument("--folds", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--proj_dim", type=int)
    ap.add_argument("--ctx_hidden", type=int)
    ap.add_argument("--emb_dim", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--ctx_dropout", type=float)
    ap.add_argument("--projector_type")
    ap.add_argument("--residual_alpha_init", type=float)
    ap.add_argument("--free_weight_init_std", type=float)
    ap.add_argument("--context_zscore_continuous", type=int, choices=[0, 1])
    ap.add_argument("--context_block_scaling", choices=["inv_sqrt_dim", "none"])
    ap.add_argument("--horizons", default="all", help="Which horizons to explain: all, 2y, 3y, 5y, or comma-separated subset.")
    ap.add_argument("--panel_size", type=int, default=30)
    ap.add_argument("--min_freq", type=float, default=0.6)
    ap.add_argument("--vote_top_k", type=int, default=200)
    ap.add_argument("--perm_repeats", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--apply_calibrator", type=int, default=1)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    selected_horizons = parse_horizons(args.horizons)
    results_subdir = Path(args.results_dir).resolve() / "siamese_hazard"
    resolved = _resolve_required(args, results_subdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    allw = load_all_windows(resolved["data_dir"])
    master = build_master_table(allw)
    t0_df, t1_df = assemble_features(allw, master)
    master = master.set_index("RID").loc[t0_df["RID"]].reset_index()
    haz = build_hazard_labels(master)

    cpg_cols = get_cpg_cols(t0_df)
    X0 = beta_to_m(t0_df[cpg_cols].to_numpy(np.float32))
    X1 = beta_to_m(t1_df[cpg_cols].to_numpy(np.float32))
    dt = t0_df["t1_dnam_m"].to_numpy(np.float32)
    E, _, _ = load_context_csv(
        resolved["context_csv"],
        cpg_cols,
        cpg_col_name=resolved["context_id_col"],
        zscore_continuous=resolved["context_zscore_continuous"],
        block_scaling=resolved["context_block_scaling"],
    )

    y3 = master["y_3y"].fillna(0).astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=resolved["folds"], shuffle=True, random_state=resolved["seed"])

    cpg_scores_folds = []
    vote_counts = np.zeros(len(cpg_cols), dtype=np.int32)
    cpg_norm_folds = []

    for fold_idx, (tr, va) in enumerate(skf.split(np.arange(len(y3)), y3), start=1):
        ckpt = Path(args.model_dir).resolve() / "siamese_hazard" / f"fold_{fold_idx}" / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        scaler = fit_scaler_on_train(X0[tr], X1[tr])
        X0_va = apply_scaler(X0[va], scaler)
        X1_va = apply_scaler(X1[va], scaler)

        model = ContextSiameseHazardNet(
            context_features=torch.from_numpy(E),
            proj_dim=resolved["proj_dim"],
            ctx_hidden=resolved["ctx_hidden"],
            emb_dim=resolved["emb_dim"],
            dropout=resolved["dropout"],
            ctx_dropout=resolved["ctx_dropout"],
            projector_type=resolved["projector_type"],
            residual_alpha_init=resolved["residual_alpha_init"],
            free_weight_init_std=resolved["free_weight_init_std"],
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        load_res = model.load_state_dict(state, strict=False)
        validate_state_load(load_res)
        model.eval()

        cal = None
        if args.apply_calibrator == 1:
            cal_path = results_subdir / "calibrators_platt" / f"fold_{fold_idx}_platt.pt"
            if cal_path.exists():
                pack = torch.load(cal_path, map_location="cpu")
                cal = MultiHeadPlattCalibrator()
                cal.load_state_dict(pack["state_dict"])

        with torch.no_grad():
            cpg_map = model.get_V().detach().cpu().numpy().astype(np.float32)
        Z0 = (X0_va @ cpg_map).astype(np.float32)
        Z1 = (X1_va @ cpg_map).astype(np.float32)

        targets, masks = build_target_maps(
            haz["h2"].to_numpy(np.float32)[va],
            haz["h3"].to_numpy(np.float32)[va],
            haz["h5"].to_numpy(np.float32)[va],
            haz["mask_h2"].to_numpy(np.int64)[va],
            haz["mask_h3"].to_numpy(np.int64)[va],
            haz["mask_h5"].to_numpy(np.int64)[va],
        )
        imps = permutation_importance_proj(
            model.core,
            Z0,
            Z1,
            dt[va],
            targets,
            masks,
            selected_horizons,
            cal=cal,
            device=device,
            seed=resolved["seed"] + 1000 * fold_idx,
            repeats=args.perm_repeats,
            batch_size=args.batch_size,
        )

        cpg_score = cpg_scores_from_map(cpg_map, imps)
        cpg_scores_folds.append(cpg_score)
        cpg_norm_folds.append(cpg_norm_from_map(cpg_map))

        top_k = min(args.vote_top_k, len(cpg_score))
        top_idx = np.argpartition(-cpg_score, top_k - 1)[:top_k]
        fold_vote = np.zeros(len(cpg_score), dtype=bool)
        fold_vote[top_idx] = True
        vote_counts += fold_vote.astype(np.int32)

    score = np.mean(np.vstack(cpg_scores_folds), axis=0)
    freq = vote_counts / max(1, len(cpg_scores_folds))
    cpg_norm = np.mean(np.vstack(cpg_norm_folds), axis=0)
    out_dir = write_panel_outputs(
        results_dir=args.results_dir,
        model_subdir="siamese_hazard",
        selected_horizons=selected_horizons,
        cpg_cols=cpg_cols,
        score=score,
        freq=freq,
        cpg_norm=cpg_norm,
        panel_size=args.panel_size,
        min_freq=args.min_freq,
    )

    print(f"Saved CpG importance (routeA) -> {out_dir}")
    print(f"Horizons={','.join(selected_horizons)} panel_size={min(args.panel_size, len(cpg_cols))} min_freq={args.min_freq}")


if __name__ == "__main__":
    main()
