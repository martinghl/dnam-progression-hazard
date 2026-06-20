from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HORIZON_ORDER = ["2y", "3y", "5y"]
_HORIZON_TO_TARGET = {
    "2y": ("h2", "mask_h2"),
    "3y": ("h3", "mask_h3"),
    "5y": ("h5", "mask_h5"),
}


def parse_horizons(spec: str | None) -> list[str]:
    if spec is None:
        return list(HORIZON_ORDER)
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        return list(HORIZON_ORDER)
    raw_tokens = [token.strip().lower() for token in text.split(",") if token.strip()]
    if not raw_tokens:
        raise ValueError("At least one horizon must be provided.")
    invalid = [token for token in raw_tokens if token not in HORIZON_ORDER]
    if invalid:
        raise ValueError(f"Unknown horizons: {sorted(set(invalid))}. Expected subset of {HORIZON_ORDER}.")
    selected = set(raw_tokens)
    return [horizon for horizon in HORIZON_ORDER if horizon in selected]


def validate_state_load(load_res: object, allowed_missing: Iterable[str] = ("ctx.E",)) -> None:
    missing = set(getattr(load_res, "missing_keys", []))
    unexpected = list(getattr(load_res, "unexpected_keys", []))
    disallowed_missing = missing - set(allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={unexpected}"
        )


def masked_bce_sum(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    valid = (mask == 1) & (~torch.isnan(targets))
    if valid.sum() == 0:
        return 0.0
    loss = F.binary_cross_entropy_with_logits(logits[valid], targets[valid].float(), reduction="mean")
    return float(loss.detach().cpu().item()) * float(valid.sum().detach().cpu().item())


def build_target_maps(
    h2: np.ndarray,
    h3: np.ndarray,
    h5: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    m5: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets = {"2y": h2, "3y": h3, "5y": h5}
    masks = {"2y": m2, "3y": m3, "5y": m5}
    return targets, masks


@torch.no_grad()
def loss_on_proj_selected(
    core,
    Z0: np.ndarray,
    Z1: np.ndarray,
    dt: np.ndarray,
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    selected_horizons: Sequence[str],
    *,
    cal=None,
    device: str = "cpu",
    batch_size: int = 256,
) -> float:
    core.eval()
    if cal is not None:
        cal.eval()
    total = 0.0
    for start in range(0, Z0.shape[0], batch_size):
        end = min(Z0.shape[0], start + batch_size)
        z0 = torch.from_numpy(Z0[start:end]).to(device)
        z1 = torch.from_numpy(Z1[start:end]).to(device)
        dtt = torch.from_numpy(dt[start:end]).to(device)
        (l2, l3, l5), _, _ = core(z0, z1, dtt)
        logits_map = {"2y": l2, "3y": l3, "5y": l5}
        if cal is not None:
            l2c, l3c, l5c = cal(l2.detach().cpu(), l3.detach().cpu(), l5.detach().cpu())
            logits_map = {"2y": l2c.to(device), "3y": l3c.to(device), "5y": l5c.to(device)}
        for horizon in selected_horizons:
            total += masked_bce_sum(
                logits_map[horizon],
                torch.from_numpy(targets[horizon][start:end]).to(device),
                torch.from_numpy(masks[horizon][start:end]).to(device),
            )
    return total


def permutation_importance_proj(
    core,
    Z0: np.ndarray,
    Z1: np.ndarray,
    dt: np.ndarray,
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    selected_horizons: Sequence[str],
    *,
    cal=None,
    device: str = "cpu",
    seed: int = 0,
    repeats: int = 1,
    batch_size: int = 256,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = loss_on_proj_selected(
        core,
        Z0,
        Z1,
        dt,
        targets,
        masks,
        selected_horizons,
        cal=cal,
        device=device,
        batch_size=batch_size,
    )
    imps = np.zeros(Z0.shape[1], dtype=np.float32)
    for col in range(Z0.shape[1]):
        deltas = []
        for _ in range(repeats):
            perm = rng.permutation(Z0.shape[0])
            Z0p = Z0.copy()
            Z1p = Z1.copy()
            Z0p[:, col] = Z0p[perm, col]
            Z1p[:, col] = Z1p[perm, col]
            loss = loss_on_proj_selected(
                core,
                Z0p,
                Z1p,
                dt,
                targets,
                masks,
                selected_horizons,
                cal=cal,
                device=device,
                batch_size=batch_size,
            )
            deltas.append(loss - base)
        imps[col] = max(0.0, float(np.mean(deltas)))
    return imps


def estimate_jacobian_cpg_sensitivity(
    project_pair_fn,
    X0: np.ndarray,
    X1: np.ndarray,
    proj_dim: int,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    n_cpg = int(X0.shape[1])
    accum = np.zeros((n_cpg, proj_dim), dtype=np.float64)
    total = 0
    for start in range(0, X0.shape[0], batch_size):
        end = min(X0.shape[0], start + batch_size)
        x0 = torch.from_numpy(X0[start:end]).to(device).requires_grad_(True)
        x1 = torch.from_numpy(X1[start:end]).to(device).requires_grad_(True)
        z0, z1 = project_pair_fn(x0, x1)
        if z0.shape[1] != proj_dim or z1.shape[1] != proj_dim:
            raise ValueError(
                f"Latent shape mismatch: expected proj_dim={proj_dim}, got {tuple(z0.shape)} and {tuple(z1.shape)}"
            )
        batch_n = end - start
        for latent_idx in range(proj_dim):
            scalar = z0[:, latent_idx].sum() + z1[:, latent_idx].sum()
            grad0, grad1 = torch.autograd.grad(
                scalar,
                (x0, x1),
                retain_graph=(latent_idx < proj_dim - 1),
            )
            mean_abs = 0.5 * (grad0.detach().abs().mean(dim=0) + grad1.detach().abs().mean(dim=0))
            accum[:, latent_idx] += mean_abs.cpu().numpy() * batch_n
        total += batch_n
    if total == 0:
        return np.zeros((n_cpg, proj_dim), dtype=np.float32)
    return (accum / float(total)).astype(np.float32)


@torch.no_grad()
def project_pair_to_numpy(
    model,
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    z0_rows = []
    z1_rows = []
    for start in range(0, X0.shape[0], batch_size):
        end = min(X0.shape[0], start + batch_size)
        x0 = torch.from_numpy(X0[start:end]).to(device)
        x1 = torch.from_numpy(X1[start:end]).to(device)
        z0, z1 = model.project_pair(x0, x1)
        z0_rows.append(z0.detach().cpu().numpy().astype(np.float32))
        z1_rows.append(z1.detach().cpu().numpy().astype(np.float32))
    if not z0_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    return np.vstack(z0_rows), np.vstack(z1_rows)


def build_panel_output_dir(results_dir: str | Path, model_subdir: str, selected_horizons: Sequence[str]) -> Path:
    results_root = Path(results_dir).resolve() / model_subdir
    if list(selected_horizons) == HORIZON_ORDER:
        return results_root / "panel_routeA"
    suffix = "_".join(selected_horizons)
    return results_root / f"panel_routeA_{suffix}"


def cpg_scores_from_map(cpg_map: np.ndarray, latent_importance: np.ndarray) -> np.ndarray:
    if cpg_map.shape[1] != latent_importance.shape[0]:
        raise ValueError(
            f"Shape mismatch: cpg_map has {cpg_map.shape[1]} latent dims, "
            f"latent_importance has {latent_importance.shape[0]}"
        )
    return (np.abs(cpg_map) @ latent_importance.reshape(-1, 1)).ravel().astype(np.float32)


def cpg_norm_from_map(cpg_map: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cpg_map, axis=1).astype(np.float32)


def write_panel_outputs(
    *,
    results_dir: str | Path,
    model_subdir: str,
    selected_horizons: Sequence[str],
    cpg_cols: Sequence[str],
    score: np.ndarray,
    freq: np.ndarray,
    cpg_norm: np.ndarray,
    panel_size: int,
    min_freq: float,
) -> Path:
    out_dir = build_panel_output_dir(results_dir, model_subdir, selected_horizons)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cpg = [f"cpg_{cg}" if not str(cg).startswith("cpg_") else str(cg) for cg in cpg_cols]
    df_imp = pd.DataFrame(
        {
            "cpg": out_cpg,
            "cg": [str(c).replace("cpg_", "") for c in out_cpg],
            "score": score,
            "freq": freq,
            "ctx_norm": cpg_norm,
        }
    ).sort_values(["freq", "score"], ascending=[False, False])
    panel = df_imp[df_imp["freq"] >= min_freq].head(panel_size)
    df_imp.to_csv(out_dir / "cpg_importance.csv", index=False)
    panel.to_csv(out_dir / "panel_topN.csv", index=False)
    return out_dir


def read_json_if_exists(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))
