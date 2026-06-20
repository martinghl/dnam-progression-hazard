import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .calibration_layer import MultiHeadPlattCalibrator, fit_platt_calibrator
from .context import CONTINUOUS_FEATURES, load_context_csv
from .data import (
    assemble_features,
    beta_to_m,
    build_hazard_labels,
    build_master_table,
    get_cpg_cols,
    load_all_windows,
)
from .metrics import brier, decision_curve, ece_uniform, safe_auc, safe_pr
from .model import ContextSiameseHazardNet


class PairHazardDataset(Dataset):
    def __init__(self, X0, X1, dt, h2, h3, h5, m2, m3, m5, RID):
        self.X0 = X0.astype(np.float32)
        self.X1 = X1.astype(np.float32)
        self.dt = dt.astype(np.float32)
        self.h2 = h2.astype(np.float32)
        self.h3 = h3.astype(np.float32)
        self.h5 = h5.astype(np.float32)
        self.m2 = m2.astype(np.int64)
        self.m3 = m3.astype(np.int64)
        self.m5 = m5.astype(np.int64)
        self.RID = np.array(RID)

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, i):
        return (
            self.X0[i],
            self.X1[i],
            self.dt[i],
            self.h2[i],
            self.h3[i],
            self.h5[i],
            self.m2[i],
            self.m3[i],
            self.m5[i],
            self.RID[i],
        )


def bce_masked(logits, targets, mask, pos_weight=None):
    m = (mask == 1) & ~torch.isnan(targets)
    if m.sum() == 0:
        return logits.new_tensor(0.0)
    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits[m], targets[m])
    return nn.functional.binary_cross_entropy_with_logits(logits[m], targets[m], reduction="mean")


def posw(y, m):
    y = y[m == 1]
    y = y[~torch.isnan(y)]
    if y.numel() == 0:
        return torch.tensor(1.0)
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def semi_hard_triplet(e0, e1, y, m, dt, margin=0.2, dt_tol=1.0):
    m_valid = (m == 1) & ~torch.isnan(y)
    if m_valid.sum() < 2:
        return e0.new_tensor(0.0)
    yv = y[m_valid].long()
    e0v = e0[m_valid]
    e1v = e1[m_valid]
    dtv = dt[m_valid]

    d_ap = (e0v - e1v).pow(2).sum(dim=1).sqrt()
    dist = torch.cdist(e0v, e0v)
    loss = 0.0
    cnt = 0

    for i in range(len(yv)):
        cand = (yv != yv[i]) & (torch.abs(dtv - dtv[i]) <= dt_tol)
        if cand.sum() == 0:
            continue
        d_an = dist[i, cand]
        semi = d_an[(d_an > d_ap[i]) & (d_an < d_ap[i] + margin)]
        d_an_h = semi.min() if semi.numel() > 0 else d_an.min()
        loss += torch.relu(d_ap[i] - d_an_h + margin)
        cnt += 1

    if cnt == 0:
        return e0.new_tensor(0.0)
    return loss / cnt


def enforce_monotone(p2, p3, p5):
    p3 = np.maximum(p2, p3)
    p5 = np.maximum(p3, p5)
    return p2, p3, p5


def fit_scaler_on_train(X0_tr, X1_tr):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(np.vstack([X0_tr, X1_tr]))
    return scaler


def apply_scaler(X, scaler):
    return scaler.transform(X).astype(np.float32)


def _safe_auc_on_nonmissing(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    keep = ~np.isnan(y_true)
    if keep.sum() == 0:
        return float("nan")
    return float(safe_auc(y_true[keep].astype(int), y_prob[keep].astype(float)))


def _selection_auc(auc: float) -> float:
    return float(auc) if np.isfinite(auc) else 0.5


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def compute_validation_metrics(model, dl_va, y2_va, y3_va, y5_va, device, use_cuda):
    logits2 = []
    logits3 = []
    logits5 = []

    model.eval()
    with torch.no_grad():
        for X0b, X1b, dtb, _, _, _, _, _, _, _ in dl_va:
            X0b = X0b.to(device, non_blocking=use_cuda)
            X1b = X1b.to(device, non_blocking=use_cuda)
            dtb = dtb.to(device, non_blocking=use_cuda)
            (l2, l3, l5), _, _ = model(X0b, X1b, dtb)
            logits2.append(l2.detach().cpu().numpy())
            logits3.append(l3.detach().cpu().numpy())
            logits5.append(l5.detach().cpu().numpy())

    if not logits2:
        p2 = np.zeros(0, dtype=np.float32)
        p3 = np.zeros(0, dtype=np.float32)
        p5 = np.zeros(0, dtype=np.float32)
    else:
        l2 = np.concatenate(logits2).astype(np.float32, copy=False)
        l3 = np.concatenate(logits3).astype(np.float32, copy=False)
        l5 = np.concatenate(logits5).astype(np.float32, copy=False)

        h2_prob = _sigmoid_np(l2)
        h3_prob = _sigmoid_np(l3)
        h5_prob = _sigmoid_np(l5)
        p2 = h2_prob
        p3 = 1.0 - (1.0 - h2_prob) * (1.0 - h3_prob)
        p5 = 1.0 - (1.0 - h2_prob) * (1.0 - h3_prob) * (1.0 - h5_prob)
        p2, p3, p5 = enforce_monotone(p2, p3, p5)

    val_p2_auc = _safe_auc_on_nonmissing(y2_va, p2)
    val_p3_auc = _safe_auc_on_nonmissing(y3_va, p3)
    val_p5_auc = _safe_auc_on_nonmissing(y5_va, p5)
    val_mean_auc_235 = float(
        np.mean([
            _selection_auc(val_p2_auc),
            _selection_auc(val_p3_auc),
            _selection_auc(val_p5_auc),
        ])
    )
    return {
        "val_p2_auc": val_p2_auc,
        "val_p3_auc": val_p3_auc,
        "val_p5_auc": val_p5_auc,
        "val_mean_auc_235": val_mean_auc_235,
    }


def train_fold(args, fold_id, data_dict, context_E, X0_all, X1_all, dt_all, tr, va, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    use_cuda = device == "cuda"

    master = data_dict["master"]
    haz = data_dict["haz"]
    X0 = X0_all
    X1 = X1_all
    dt = dt_all

    h2 = haz["h2"].to_numpy(np.float32)
    m2 = haz["mask_h2"].to_numpy(np.int64)
    h3 = haz["h3"].to_numpy(np.float32)
    m3 = haz["mask_h3"].to_numpy(np.int64)
    h5 = haz["h5"].to_numpy(np.float32)
    m5 = haz["mask_h5"].to_numpy(np.int64)

    scaler = fit_scaler_on_train(X0[tr], X1[tr])
    X0_tr = apply_scaler(X0[tr], scaler)
    X1_tr = apply_scaler(X1[tr], scaler)
    X0_va = apply_scaler(X0[va], scaler)
    X1_va = apply_scaler(X1[va], scaler)

    ds_tr = PairHazardDataset(
        X0_tr,
        X1_tr,
        dt[tr],
        h2[tr],
        h3[tr],
        h5[tr],
        m2[tr],
        m3[tr],
        m5[tr],
        master["RID"].values[tr],
    )
    ds_va = PairHazardDataset(
        X0_va,
        X1_va,
        dt[va],
        h2[va],
        h3[va],
        h5[va],
        m2[va],
        m3[va],
        m5[va],
        master["RID"].values[va],
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
    )

    model = ContextSiameseHazardNet(
        context_features=torch.from_numpy(context_E),
        proj_dim=args.proj_dim,
        ctx_hidden=args.ctx_hidden,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        ctx_dropout=args.ctx_dropout,
        projector_type=args.projector_type,
        residual_alpha_init=args.residual_alpha_init,
        free_weight_init_std=args.free_weight_init_std,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pw2 = posw(torch.from_numpy(h2[tr]), torch.from_numpy(m2[tr])).to(device)
    pw3 = posw(torch.from_numpy(h3[tr]), torch.from_numpy(m3[tr])).to(device)
    pw5 = posw(torch.from_numpy(h5[tr]), torch.from_numpy(m5[tr])).to(device)

    y2_va = master["y_2y"].to_numpy(np.float32)[va]
    y3_va = master["y_3y"].to_numpy(np.float32)[va]
    y5_va = master["y_5y"].to_numpy(np.float32)[va]

    best_score = -float("inf")
    best_state = None
    bad_epochs = 0
    history_rows = []

    for ep in range(1, args.max_epochs + 1):
        model.train()
        for X0b, X1b, dtb, h2b, h3b, h5b, m2b, m3b, m5b, _ in dl_tr:
            X0b = X0b.to(device, non_blocking=use_cuda)
            X1b = X1b.to(device, non_blocking=use_cuda)
            dtb = dtb.to(device, non_blocking=use_cuda)
            h2b = h2b.to(device, non_blocking=use_cuda)
            h3b = h3b.to(device, non_blocking=use_cuda)
            h5b = h5b.to(device, non_blocking=use_cuda)
            m2b = m2b.to(device, non_blocking=use_cuda)
            m3b = m3b.to(device, non_blocking=use_cuda)
            m5b = m5b.to(device, non_blocking=use_cuda)

            (l2, l3, l5), (_, _, _), (e0, e1) = model(X0b, X1b, dtb)
            loss = (
                bce_masked(l2, h2b, m2b, pos_weight=pw2)
                + bce_masked(l3, h3b, m3b, pos_weight=pw3)
                + bce_masked(l5, h5b, m5b, pos_weight=pw5)
            )
            if args.triplet_weight > 0:
                loss = loss + args.triplet_weight * (
                    semi_hard_triplet(e0, e1, h2b, m2b, dtb, margin=args.triplet_margin)
                    + semi_hard_triplet(e0, e1, h3b, m3b, dtb, margin=args.triplet_margin)
                    + semi_hard_triplet(e0, e1, h5b, m5b, dtb, margin=args.triplet_margin)
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_metrics = compute_validation_metrics(model, dl_va, y2_va, y3_va, y5_va, device, use_cuda)
        if args.stop_metric == "mean_auc_235":
            stop_metric_value = val_metrics["val_mean_auc_235"]
        else:
            stop_metric_value = _selection_auc(val_metrics["val_p2_auc"])

        is_best = stop_metric_value > best_score
        if is_best:
            best_score = stop_metric_value
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
                if k != "ctx.E"
            }
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": ep,
                "val_p2_auc": val_metrics["val_p2_auc"],
                "val_p3_auc": val_metrics["val_p3_auc"],
                "val_p5_auc": val_metrics["val_p5_auc"],
                "val_mean_auc_235": val_metrics["val_mean_auc_235"],
                "stop_metric_name": args.stop_metric,
                "stop_metric_value": stop_metric_value,
                "best_so_far": best_score,
                "is_best": int(is_best),
            }
        )

        if args.patience > 0 and bad_epochs >= args.patience:
            break

    history_path = os.path.join(out_dir, "epoch_history.csv")
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    ckpt_path = os.path.join(out_dir, "best.ckpt")
    torch.save(best_state, ckpt_path)
    return best_state, scaler


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--context_csv", required=True, help="CpG context feature CSV aligned by cg id (column 'cpg').")
    ap.add_argument("--context_id_col", default="cpg", help="Column name for CpG id in context_csv (default: cpg)")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--triplet_weight", type=float, default=0.1)
    ap.add_argument("--triplet_margin", type=float, default=0.2)
    ap.add_argument("--proj_dim", type=int, default=64, help="Latent projection dim produced by context projector.")
    ap.add_argument("--ctx_hidden", type=int, default=32, help="Hidden dim for context weight generator g(e).")
    ap.add_argument("--ctx_dropout", type=float, default=0.0)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--calibration_mode", type=str, default="per_head_then_compose", choices=["per_head_then_compose"])
    ap.add_argument("--cal_max_iter", type=int, default=200)
    ap.add_argument("--cal_l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop_metric", type=str, default="mean_auc_235", choices=["p2_auc", "mean_auc_235"])
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--context_zscore_continuous", type=int, default=1)
    ap.add_argument("--context_block_scaling", type=str, default="inv_sqrt_dim", choices=["inv_sqrt_dim", "none"])
    ap.add_argument("--projector_type", type=str, default="residual", choices=["residual", "context_only"])
    ap.add_argument("--residual_alpha_init", type=float, default=0.05)
    ap.add_argument("--free_weight_init_std", type=float, default=0.01)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir = os.path.join(args.results_dir, "siamese_hazard")
    os.makedirs(res_dir, exist_ok=True)

    allw = load_all_windows(args.data_dir)
    master = build_master_table(allw)
    t0_df, t1_df = assemble_features(allw, master)
    master = master.set_index("RID").loc[t0_df["RID"]].reset_index()
    haz = build_hazard_labels(master)

    data_dict = {"t0": t0_df, "t1": t1_df, "master": master, "haz": haz}
    cpg_cols = get_cpg_cols(t0_df)
    X0_all = beta_to_m(t0_df[cpg_cols].to_numpy(np.float32))
    X1_all = beta_to_m(t1_df[cpg_cols].to_numpy(np.float32))
    dt_all = t0_df["t1_dnam_m"].to_numpy(np.float32)

    E, feat_names, preprocess_summary = load_context_csv(
        args.context_csv,
        cpg_cols,
        cpg_col_name=args.context_id_col,
        zscore_continuous=bool(args.context_zscore_continuous),
        block_scaling=args.context_block_scaling,
    )
    E = np.array(E, dtype=np.float32, copy=True)

    preprocess_summary = {
        **preprocess_summary,
        "context_csv": args.context_csv,
        "context_id_col": args.context_id_col,
        "n_cpg": int(E.shape[0]),
        "n_features": int(E.shape[1]),
        "continuous_whitelist": list(CONTINUOUS_FEATURES),
    }
    _write_json(os.path.join(res_dir, "context_preprocess_summary.json"), preprocess_summary)

    run_config = {
        "cli_args": vars(args),
        "resolved": {
            "output_subdir": "siamese_hazard",
            "device": device,
            "context_zscore_continuous": bool(args.context_zscore_continuous),
            "context_block_scaling": args.context_block_scaling,
            "projector_type": args.projector_type,
            "residual_alpha_init": args.residual_alpha_init,
            "free_weight_init_std": args.free_weight_init_std,
            "stop_metric": args.stop_metric,
            "patience": args.patience,
            "n_context_features": len(feat_names),
        },
    }
    _write_json(os.path.join(res_dir, "run_config.json"), run_config)

    y3 = master["y_3y"].fillna(0).astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = pd.DataFrame({"RID": master["RID"].values})
    oof["y2"] = master["y_2y"].values
    oof["y3"] = master["y_3y"].values
    oof["y5"] = master["y_5y"].values
    oof["mask_h2"] = haz["mask_h2"].values
    oof["mask_h3"] = haz["mask_h3"].values
    oof["mask_h5"] = haz["mask_h5"].values

    for hz in ["h2", "h3", "h5"]:
        oof[f"{hz}_logit"] = np.nan
        oof[f"{hz}_prob"] = np.nan
        oof[f"{hz}_logit_cal"] = np.nan
        oof[f"{hz}_cal"] = np.nan
    oof["p2"] = np.nan
    oof["p3"] = np.nan
    oof["p5"] = np.nan

    cal_dir = os.path.join(res_dir, "calibrators_platt")
    os.makedirs(cal_dir, exist_ok=True)
    cal_meta = []

    for k, (tr, va) in enumerate(skf.split(np.arange(len(y3)), y3), start=1):
        fold_model_dir = os.path.join(args.model_dir, "siamese_hazard", f"fold_{k}")
        fold_res_dir = os.path.join(res_dir, f"fold_{k}")
        os.makedirs(fold_model_dir, exist_ok=True)
        os.makedirs(fold_res_dir, exist_ok=True)

        state, scaler = train_fold(
            args,
            k,
            data_dict,
            E,
            X0_all,
            X1_all,
            dt_all,
            tr,
            va,
            fold_model_dir,
            device,
        )

        X0_va = apply_scaler(X0_all[va], scaler)
        X1_va = apply_scaler(X1_all[va], scaler)

        model = ContextSiameseHazardNet(
            context_features=torch.from_numpy(E),
            proj_dim=args.proj_dim,
            ctx_hidden=args.ctx_hidden,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            ctx_dropout=args.ctx_dropout,
            projector_type=args.projector_type,
            residual_alpha_init=args.residual_alpha_init,
            free_weight_init_std=args.free_weight_init_std,
        ).to(device)
        load_res = model.load_state_dict(state, strict=False)
        missing = set(load_res.missing_keys)
        if missing - {"ctx.E"} or load_res.unexpected_keys:
            raise RuntimeError(
                f"Unexpected checkpoint mismatch: missing={load_res.missing_keys}, unexpected={load_res.unexpected_keys}"
            )
        model.eval()

        with torch.no_grad():
            X0t = torch.from_numpy(X0_va).to(device, non_blocking=(device == "cuda"))
            X1t = torch.from_numpy(X1_va).to(device, non_blocking=(device == "cuda"))
            dtt = torch.from_numpy(dt_all[va]).to(device, non_blocking=(device == "cuda"))
            (l2, l3, l5), _, _ = model(X0t, X1t, dtt)
            l2 = l2.detach().cpu().numpy()
            l3 = l3.detach().cpu().numpy()
            l5 = l5.detach().cpu().numpy()
            p2 = _sigmoid_np(l2)
            p3 = _sigmoid_np(l3)
            p5 = _sigmoid_np(l5)

        oof.loc[va, "h2_logit"] = l2
        oof.loc[va, "h3_logit"] = l3
        oof.loc[va, "h5_logit"] = l5
        oof.loc[va, "h2_prob"] = p2
        oof.loc[va, "h3_prob"] = p3
        oof.loc[va, "h5_prob"] = p5

        y2_va = master["y_2y"].to_numpy(np.float32)[va]
        y3_va = master["y_3y"].to_numpy(np.float32)[va]
        y5_va = master["y_5y"].to_numpy(np.float32)[va]
        m2_va = haz["mask_h2"].to_numpy(np.int64)[va]
        m3_va = haz["mask_h3"].to_numpy(np.int64)[va]
        m5_va = haz["mask_h5"].to_numpy(np.int64)[va]

        cal = MultiHeadPlattCalibrator()
        meta = fit_platt_calibrator(
            cal,
            torch.from_numpy(l2).float(),
            torch.from_numpy(l3).float(),
            torch.from_numpy(l5).float(),
            torch.from_numpy(y2_va).float(),
            torch.from_numpy(y3_va).float(),
            torch.from_numpy(y5_va).float(),
            torch.from_numpy(m2_va),
            torch.from_numpy(m3_va),
            torch.from_numpy(m5_va),
            max_iter=args.cal_max_iter,
            l2_reg=args.cal_l2,
            device="cpu",
        )
        torch.save({"state_dict": cal.state_dict(), "meta": meta}, os.path.join(cal_dir, f"fold_{k}_platt.pt"))
        cal_meta.append({"fold": int(k), **meta})

        with torch.no_grad():
            l2c, l3c, l5c = cal(
                torch.from_numpy(l2).float(),
                torch.from_numpy(l3).float(),
                torch.from_numpy(l5).float(),
            )
            l2c = l2c.numpy()
            l3c = l3c.numpy()
            l5c = l5c.numpy()
            p2c = _sigmoid_np(l2c)
            p3c = _sigmoid_np(l3c)
            p5c = _sigmoid_np(l5c)

        oof.loc[va, "h2_logit_cal"] = l2c
        oof.loc[va, "h3_logit_cal"] = l3c
        oof.loc[va, "h5_logit_cal"] = l5c
        oof.loc[va, "h2_cal"] = p2c
        oof.loc[va, "h3_cal"] = p3c
        oof.loc[va, "h5_cal"] = p5c

        p2_out = p2c
        p3_out = 1.0 - (1.0 - p2c) * (1.0 - p3c)
        p5_out = 1.0 - (1.0 - p2c) * (1.0 - p3c) * (1.0 - p5c)
        p2_out, p3_out, p5_out = enforce_monotone(p2_out, p3_out, p5_out)
        oof.loc[va, "p2"] = p2_out
        oof.loc[va, "p3"] = p3_out
        oof.loc[va, "p5"] = p5_out

    pd.DataFrame(cal_meta).to_csv(os.path.join(cal_dir, "platt_params.csv"), index=False)

    metrics = []
    for tag, yname, pname in [("2y", "y2", "p2"), ("3y", "y3", "p3"), ("5y", "y5", "p5")]:
        m = ~oof[yname].isna()
        y = oof.loc[m, yname].astype(int).to_numpy()
        p = oof.loc[m, pname].astype(float).to_numpy()
        metrics.append(
            {
                "horizon": tag,
                "AUC": safe_auc(y, p),
                "PR_AUC": safe_pr(y, p),
                "Brier": brier(y, p),
                "ECE": ece_uniform(y, p),
                "N": int(m.sum()),
            }
        )
        dca = decision_curve(y, p)
        pd.DataFrame({"threshold": dca[:, 0], "net_benefit": dca[:, 1]}).to_csv(
            os.path.join(res_dir, f"dca_perhead_{tag}.csv"),
            index=False,
        )

    oof.to_csv(os.path.join(res_dir, "oof_hazards_and_risks_perhead.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(res_dir, "metrics_perhead.csv"), index=False)
    print("Done. Outputs:", res_dir)


if __name__ == "__main__":
    main()
