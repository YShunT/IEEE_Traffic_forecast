"""ongoing の共通部分：窓、渋滞の塊、損失、確率の表、0.5 で切る規則。"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.config import DATES, FIELDS, MONTH, TRAIN_DAYS, work_path
from core.runtime import log
from data.panel import Panel

FIRST_SLOT, LAST_SLOT = 12, 281
MIN_COVERAGE, MAX_PERSISTENCE_IOU = 0.7, 0.9


def ongoing_slots(speed, threshold):
    valid = np.isfinite(speed)
    queued = valid & (speed <= threshold)
    n = speed.shape[1]
    cq = np.vstack([np.zeros((1, n), int), queued.cumsum(0)])
    cv = np.vstack([np.zeros((1, n), int), valid.cumsum(0)])
    slots = np.arange(FIRST_SLOT, LAST_SLOT + 1)
    established = ((cq[slots] - cq[slots - 12]) >= 2).any(1)
    coverage = (cv[slots] - cv[slots - 12]).sum(1) / (12 * n)
    ahead = slots[:, None] + np.arange(1, 7)
    fq, fv = queued[ahead], valid[ahead]
    persist = queued[slots - 1, None, :]
    union = ((persist | fq) & fv).sum((1, 2))
    iou = np.divide((persist & fq & fv).sum((1, 2)), union, out=np.ones(len(slots)), where=union > 0)
    return slots[established & fq.any((1, 2)) & (coverage >= MIN_COVERAGE) & (iou <= MAX_PERSISTENCE_IOU)]


def candidates(c, panels):
    path = Path(c["cache"]) / "origins.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for name, p in panels.items():
        speed = p.eligible("train")
        for day in range(TRAIN_DAYS):
            rows += [(name, day, int(s)) for s in ongoing_slots(speed[day], p.threshold)]
    f = pd.DataFrame(rows, columns=["panel", "day", "slot"])
    f["month"] = MONTH[f.day.to_numpy()]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    f.to_parquet(temp, index=False)
    temp.replace(path)
    return f


def thin(frame, gap):
    keep = []
    for _, g in frame.assign(t=frame.day * 288 + frame.slot).sort_values(["panel", "t"]).groupby("panel", sort=False):
        last = -(10**9)
        for i, t in zip(g.index, g.t):
            if t - last >= gap:
                keep.append(i)
                last = t
    return frame.loc[keep]


def training(c, origins, gap):
    t = thin(origins, gap) if gap > 1 else origins
    t = t[(t.month >= c["fit_months"][0]) & (t.month <= c["fit_months"][1])]
    if c.get("limit_windows"):
        t = t.groupby("panel", sort=False).head(c["limit_windows"])
    t = t.sort_values(["panel", "day", "slot"]).reset_index(drop=True)
    t["window_id"] = [f"P_{p}_{DATES[d]:%Y%m%d}_{s:03d}" for p, d, s in zip(t.panel, t.day, t.slot)]
    return t


def official(c):
    frames = []
    for name in c["panels"]:
        for split in ("validation", "private"):
            f = pd.read_csv(Path(c["data"]) / "task2" / name / split / "window_index.csv")
            frames.append(f[f.condition == "queue_ongoing"])
    w = pd.concat(frames, ignore_index=True)
    origin = pd.to_datetime(w.forecast_origin, utc=True)
    w["day"] = (origin.dt.tz_localize(None).dt.normalize() - DATES[0]).dt.days
    w["slot"] = origin.dt.hour * 12 + origin.dt.minute // 5
    w["month"] = w.date.str[:7]
    return w[["window_id", "panel", "family_id", "split", "date", "month", "day", "slot", "forecast_origin"]]


def labels(p, source, rows):
    speed = p.eligible(source)
    y, e = np.zeros((len(rows), p.n, 6), bool), np.zeros((len(rows), p.n, 6), bool)
    for k, r in enumerate(rows.itertuples()):
        v = speed[r.day, r.slot + 1 : r.slot + 7]
        ok = np.isfinite(v)
        y[k], e[k] = (ok & (v <= p.threshold)).T, ok.T
    return y, e


def released_histories(c, p, rows):
    result = {}
    for split in sorted(rows.split.unique()):
        root = Path(c["data"]) / "task2" / p.name / split
        frame = pd.read_parquet(root / "window_history.parquet")
        wanted = rows[rows.split == split].set_index("window_id")
        for wid, f in frame[frame.window_id.isin(wanted.index)].groupby("window_id", sort=False):
            origin = pd.Timestamp(wanted.loc[wid, "forecast_origin"])
            stamps = pd.to_datetime(f.timestamp, utc=True)
            expected = pd.date_range(origin - pd.Timedelta(minutes=60), periods=12, freq="5min")
            if sorted(stamps.unique()) != list(expected) or f.duplicated(["timestamp", "link_id"]).any():
                raise ValueError(f"invalid official history {wid}")
            t = ((stamps - expected[0]).dt.total_seconds() / 300).to_numpy(int)
            i = f.link_id.map(p.index).to_numpy(int)
            arrays = {}
            for field in FIELDS:
                arrays[field] = np.full((12, p.n), np.nan, np.float32)
                arrays[field][t, i] = f[field].to_numpy(np.float32)
            pct, eligible = np.zeros((12, p.n), np.float32), np.zeros((12, p.n), bool)
            pct[t, i] = np.nan_to_num(f.pct_observed.to_numpy(np.float32))
            eligible[t, i] = (f.is_score_eligible.eq(1) & f.pct_observed.ge(75)).to_numpy()
            result[wid] = (arrays, pct, eligible)
    return result


class Histories:
    def __init__(self, c, p, rows, kind):
        self.p, self.rows, self.kind = p, rows.reset_index(drop=True), kind
        self.source = "train" if kind == "train" else "public"
        self.released = released_histories(c, p, rows) if kind == "official" else None

    def window(self, k, eligible_only):
        r = self.rows.iloc[k]
        day, slot = int(r.day), int(r.slot)
        if self.released is None:
            return self.p.history(self.source, day, slot, eligible_only), self.p.pct(self.source, day, slot)
        arrays, pct, eligible = self.released[r.window_id]
        if eligible_only:
            arrays = {f: np.where(eligible & np.isfinite(a), a, np.nan).astype(np.float32) for f, a in arrays.items()}
        at, at_pct = self.p.at_origin(day, slot, eligible_only)
        return {f: np.vstack([a, at[f][None]]) for f, a in arrays.items()}, np.vstack([pct, at_pct[None]])

    def ramps(self, k):
        r = self.rows.iloc[k]
        return self.p.ramp_totals(int(r.day), int(r.slot))


def prepare(c):
    panels = {n: Panel(c, n) for n in c["panels"]}
    origins = candidates(c, panels)
    official_windows = official(c)
    fit = int(origins.month.between(*c["fit_months"]).sum())
    if fit != c["expected"]["origins"]:
        raise ValueError(f"origins differ from the plan: {fit} against {c['expected']['origins']}")
    if len(official_windows) != c["expected"]["official"]:
        raise ValueError(f"official windows differ from the plan: {len(official_windows)} against {c['expected']['official']}")
    log(
        f"  [data]    bypass links: {sum(int(p.bypass.sum()) for p in panels.values())} | train origins: {fit:,} "
        f"({c['fit_months'][0]}..{c['fit_months'][1]}) | official: {len(official_windows)}"
    )
    return panels, origins, official_windows


def training_windows(c, origins, gap):
    rows = training(c, origins, gap)
    plan = c["expected"]["windows"]
    if not c["trial"] and str(gap) in plan and len(rows) != plan[str(gap)]:
        raise ValueError(f"{len(rows)} training windows {gap} slots apart; the plan has {plan[str(gap)]}")
    return rows


def binarize(prob, threshold=0.5):
    pred = prob >= threshold
    flat = pred.reshape(len(pred), -1)
    empty = ~flat.any(1)
    flat[empty, prob.reshape(len(prob), -1)[empty].argmax(1)] = True
    return pred.astype(np.int8)


def cell_table(p, rows, prob):
    w, n = prob.shape[:2]
    minutes = (rows.slot.to_numpy()[:, None, None] + np.arange(1, 7)[None, None, :]) * 5
    stamp = np.broadcast_to(DATES[rows.day.to_numpy()].values[:, None, None] + minutes.astype("timedelta64[m]"), (w, n, 6)).reshape(-1)
    return pd.DataFrame(
        dict(
            window_id=np.repeat(rows.window_id.to_numpy(), n * 6),
            panel=p.name,
            link_id=np.tile(np.repeat(p.ids, 6), w),
            step=np.tile(np.arange(1, 7), w * n),
            timestamp=pd.DatetimeIndex(stamp).strftime("%Y-%m-%dT%H:%M:%SZ"),
            probability=prob.reshape(-1).astype(np.float32),
            queue_pred=binarize(prob).reshape(-1),
        )
    )


def cube(frame, p, rows):
    f = frame[frame.window_id.isin(rows.window_id)]
    a = np.full((len(rows), p.n, 6), np.nan, np.float32)
    a[
        f.window_id.map({w: k for k, w in enumerate(rows.window_id)}).to_numpy(), f.link_id.map(p.index).to_numpy(), f.step.to_numpy() - 1
    ] = f.probability.to_numpy()
    if np.isnan(a).any():
        raise ValueError(f"probabilities are missing for some cells of {p.name}")
    return a


def save_cells(c, run_id, name, frames):
    pd.concat(frames, ignore_index=True).to_parquet(work_path(c, run_id, f"official_{name}.parquet"), index=False, compression="zstd")


def read_cells(c, run_id, name):
    return pd.read_parquet(work_path(c, run_id, f"official_{name}.parquet"))


def loss_terms(logits, y, e, start, weight, margin, side):
    s = start[:, :, None]
    stay = (s < 0) & (y < 0.5)
    if side == "all":
        stay = stay | ((s > 0) & (y > 0.5))
    w = torch.where(stay, weight, 1.0) * e
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1.0)
    b = logits.shape[0]
    z, t, v = logits.reshape(b, -1), y.reshape(b, -1), e.reshape(b, -1)
    errors = torch.where(v, torch.where(stay, margin, 1.0).reshape(b, -1) - z * (2.0 * t - 1.0), z.new_full((), -1e4))
    errors, order = torch.sort(errors, dim=1, descending=True)
    gt, ok = torch.gather(t * v, 1, order), torch.gather(v.to(t.dtype), 1, order)
    total = gt.sum(1, keepdim=True)
    jaccard = 1.0 - (total - gt.cumsum(1)) / (total + (ok * (1.0 - gt)).cumsum(1)).clamp_min(1.0)
    grad = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], 1)
    return bce + (torch.relu(errors) * grad).sum(1)


NAMES = ["tail_km", "head_km", "tail_velocity", "tail_arrival", "to_head_km", "recovery", "balance", "length_km"]
REACH_KM, UPSTREAM_KM, RECENT = 20.0, 3.0, 3
SCALES = {
    "tail_km": (10.0, 2.0),
    "head_km": (10.0, 2.0),
    "tail_velocity": (30.0, 2.0),
    "tail_arrival": (60.0, 4.0),
    "to_head_km": (10.0, 2.0),
    "recovery": (60.0, 4.0),
    "balance": (1.0, 2.0),
    "length_km": (10.0, 3.0),
}
CHANNELS = [f"shock_{n}" for n in NAMES] + ["shock_upstream_of_queue", "shock_downstream_of_queue"]


def _recent(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[-RECENT:], 0)


def last_state(speed, threshold):
    ok = np.isfinite(speed)
    at = speed.shape[0] - 1 - ok[::-1].argmax(0)
    last = speed[at, np.arange(speed.shape[1])]
    return np.where(ok.any(0), (last <= threshold).astype(np.int8), np.int8(-1))


def queues(state, order):
    s, runs, first, last = state[order], [], None, None
    for k, v in enumerate(s):
        if v == 1:
            first = k if first is None else first
            last = k
        elif v == 0 and first is not None:
            runs.append((first, last))
            first = None
    if first is not None:
        runs.append((first, last))
    return runs


def features(p, speed, flow):
    n, order = p.n, p.road
    out = np.full((n, len(NAMES)), np.nan, np.float32)
    state = last_state(speed, p.threshold)
    runs = queues(state, order)
    if not runs:
        return out
    v, q = _recent(speed), _recent(flow)
    k = np.divide(q, v, out=np.full(n, np.nan), where=np.isfinite(q) & np.isfinite(v) & (v > 1))
    tails, heads, segs = [], [], []
    for first, last in runs:
        tail, head = order[first], order[last]
        inside = order[first : last + 1]
        inside = inside[state[inside] == 1]
        up = [j for j in order[:first][::-1] if p.x_start[tail] - p.x_end[j] <= UPSTREAM_KM]
        up = [j for j in up if state[j] == 0 and np.isfinite(k[j])]
        near = [j for j in inside[:3] if np.isfinite(k[j])]
        w = np.nan
        if up and near:
            q_in, k_in, q_q, k_q = q[up[0]], k[up[0]], np.mean(q[near]), np.mean(k[near])
            if k_q - k_in > 1.0:
                w = float(np.clip((q_in - q_q) / (k_in - k_q), -60.0, 60.0))
        out_flow = [q[j] for j in order[last:] if np.isfinite(q[j])]
        balance = (q[up[0]] - out_flow[0]) / p.capacity[head] if up and out_flow else np.nan
        tails.append(p.x_start[tail])
        heads.append(p.x_end[head])
        segs.append((w, balance, p.x_end[head] - p.x_start[tail], order[first : last + 1]))
    tails, heads = np.array(tails), np.array(heads)
    gap = tails[None] - p.x_end[:, None]
    gap = np.where(gap >= -1e-6, np.maximum(gap, 0.0), np.inf)
    member = np.full(n, -1)
    for g, (w, balance, length, members) in enumerate(segs):
        gap[members, g] = 0.0
        member[members] = g
    near = gap.argmin(1)
    dist = gap[np.arange(n), near]
    use = dist <= REACH_KM
    out[use, 0] = dist[use]
    for j in np.flatnonzero(use):
        w, balance, length, members = segs[near[j]]
        out[j, 2], out[j, 6], out[j, 7] = w, balance, length
        if member[j] < 0 and np.isfinite(w) and w < -0.5:
            out[j, 3] = min(out[j, 0] / -w * 60.0, 240.0)
        if member[j] >= 0:
            out[j, 4] = max(heads[near[j]] - p.x_end[j], 0.0)
            out[j, 5] = min(out[j, 4] / (p.capacity[j] / max(p.k_jam[j] - p.k_crit[j], 1.0)) * 60.0, 240.0)
    after = p.x_start[:, None] - heads[None]
    after = np.where(after >= -1e-6, after, np.inf)
    for g, (_, _, _, members) in enumerate(segs):
        after[members, g] = np.inf
    dist = after.min(1)
    use = dist <= REACH_KM
    out[use, 1] = np.maximum(dist[use], 0.0)
    return out[p.anchor]


def channels(f):
    out = np.zeros((len(f), len(CHANNELS)), np.float32)
    for i, name in enumerate(NAMES):
        scale, cap = SCALES[name]
        out[:, i] = np.clip(np.nan_to_num(f[:, i] / scale), -cap, cap)
    out[:, -2], out[:, -1] = np.isfinite(f[:, 0]), np.isfinite(f[:, 1])
    return out


def reach_margin(f, steps=6):
    speed = np.where(np.isfinite(f[:, 2]) & (f[:, 2] < 0), -f[:, 2], np.nan)
    return speed[:, None] * (np.arange(1, steps + 1)[None] * 5.0 / 60.0) - f[:, 0:1]


def downstream_of_queue(p, speed, margin_km, reach_km):
    state = last_state(speed, p.threshold)
    runs = queues(state, p.road)
    out = np.zeros(p.n, bool)
    if not runs:
        return out
    member = np.zeros(p.n, bool)
    tails = np.array([p.x_start[p.road[first]] for first, _ in runs])
    heads = np.array([p.x_end[p.road[last]] for _, last in runs])
    for first, last in runs:
        member[p.road[first : last + 1]] = True
    before = tails[None] - p.x_end[:, None]
    after = p.x_start[:, None] - heads[None]
    up = np.where(before >= -1e-6, np.maximum(before, 0.0), np.inf).min(1)
    down = np.where(after >= -1e-6, np.maximum(after, 0.0), np.inf).min(1)
    return (~member & (down < up) & (down > margin_km) & (down <= reach_km))[p.anchor]
