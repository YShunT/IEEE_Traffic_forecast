"""Task 4：各パネルの経路交通量を、区間交通量と事前値から非負最小二乗（双対ニュートン法）で解く。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.config import work_path
from core.runtime import Runtime, log, make_runtime, section, to_numpy
from data.panel import family_mean
from data.release import Release, read_link_counts, read_path_link_incidence, read_path_set, read_weak_prior

ARMIJO = 1e-4
MIN_STEP = 1e-10
STALL_LIMIT = 10


def incidence(path_set: pd.DataFrame, links: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    path_ids = path_set["path_id"].tolist()
    link_ids = links["link_id"].drop_duplicates().tolist()
    rows = links["link_id"].map({link: i for i, link in enumerate(link_ids)}).to_numpy()
    cols = links["path_id"].map({path: j for j, path in enumerate(path_ids)})
    if cols.isna().any():
        raise ValueError("path_link_incidence.csv references paths missing from path_set.csv")
    matrix = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    matrix[rows, cols.to_numpy(dtype=np.int64)] = 1.0
    return path_ids, link_ids, matrix


def align(values: pd.Series, ids: list[str]) -> np.ndarray:
    return pd.to_numeric(values.reindex(ids), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


def solve(
    A: np.ndarray, counts: np.ndarray, prior: np.ndarray, lam: float, runtime: Runtime, max_iter: int, tol: float
) -> tuple[np.ndarray, int, str]:
    A_t, c, b = runtime.floats(A), runtime.floats(counts), runtime.floats(prior)
    eye = torch.eye(A_t.shape[0], dtype=runtime.dtype, device=runtime.device)
    tol = max(tol, 100.0 * torch.finfo(runtime.dtype).eps)
    scale = max(1.0, float(torch.linalg.vector_norm(c)))

    def primal(y):
        return (b - A_t.T @ y / (2.0 * lam)).clamp(min=0.0)

    def dual(y, f):
        return -(y @ y) / 4.0 - c @ y + lam * ((f - b) @ (f - b)) + (A_t.T @ y) @ f

    y = torch.zeros_like(c)
    f = primal(y)
    gradient = A_t @ f - c - y / 2.0
    status, best, since_best = "max_iter", float("inf"), 0
    for iterations in range(1, max_iter + 1):
        residual = float(torch.linalg.vector_norm(gradient)) / scale
        if residual <= tol:
            status = "converged"
            break
        best, since_best = (residual, 0) if residual < 0.5 * best else (best, since_best + 1)
        if since_best >= STALL_LIMIT:
            status = "stalled"
            break
        active = (f > 0).to(runtime.dtype)
        hessian = eye / 2.0 + (A_t * active[None, :]) @ A_t.T / (2.0 * lam)
        direction = torch.cholesky_solve(gradient[:, None], torch.linalg.cholesky(hessian))[:, 0]
        value, slope, step = dual(y, f), float(gradient @ direction), 1.0
        while True:
            y_new = y + step * direction
            f_new = primal(y_new)
            if float(dual(y_new, f_new)) >= float(value) + ARMIJO * step * slope or step < MIN_STEP:
                break
            step /= 2.0
        if step < MIN_STEP:
            status = "stalled"
            break
        y, f = y_new, f_new
        gradient = A_t @ f - c - y / 2.0
    if float(torch.linalg.vector_norm(gradient)) / scale <= tol:
        status = "converged"
    return to_numpy(f), iterations, status


SPLITS = ("validation", "private")
COLUMNS = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]


def solve_panel(release, panel, split, t, runtime):
    path_set = read_path_set(release, panel)
    path_ids, link_ids, matrix = incidence(path_set, read_path_link_incidence(release, panel))
    prior_table = read_weak_prior(release, panel, split)
    prior = align(prior_table.set_index("path_id")["path_flow"], path_ids)
    counts = read_link_counts(release, panel, split)
    measured = [link for link in link_ids if link in counts.index]
    A = matrix[[link in counts.index for link in link_ids]]
    target = align(counts, measured)
    flows, iterations, status = solve(A, target, prior, t["reg_lambda"], runtime, t["max_iter"], t["tol"])
    if not (np.isfinite(flows).all() and (flows >= 0).all()):
        raise RuntimeError(f"{panel} {split}: path flows must be finite and non-negative")
    gap = float((runtime.floats(A) @ runtime.floats(flows) - runtime.floats(target)).abs().sum())
    s_link = max(0.0, 1.0 - gap / max(float(target.sum()), 1e-9))
    table = path_set[["path_id", "origin_zone", "destination_zone"]].reset_index(drop=True)
    table = table.assign(panel=panel, departure_time=prior_table["departure_time"].iloc[0], path_flow=flows)[COLUMNS]
    return table, s_link, iterations, status


def run(c, run_id):
    t = c["task4"]
    release = Release(Path(c["data"]))
    runtime = make_runtime("cpu", "float64")
    for split in SPLITS:
        section(f"task4 {split}")
        flows, s_link = [], {}
        for panel in release.panels():
            table, s_link[panel], iterations, status = solve_panel(release, panel, split, t, runtime)
            if status != "converged" or s_link[panel] < 0.999:
                raise ValueError(f"{panel} {split}: {status}, S_link {s_link[panel]:.6f}")
            flows.append(table)
            log(f"{panel:<10} | paths: {len(table):,} | iters: {iterations} | S_link: {s_link[panel]:.6f}")
        log(f"  [done]    S_link: {family_mean(s_link, release.family_of()):.6f}")
        pd.concat(flows, ignore_index=True).to_parquet(work_path(c, run_id, f"path_flow_{split}.parquet"), index=False)
