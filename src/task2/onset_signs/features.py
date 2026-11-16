"""onset_signs の共通部分：窓、GBM の列と DCRNN 型モデルのチャネル、確率の校正。"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DATES, FIELDS, MIN_SPEED_KMH, MONTH, SLOTS_PER_DAY, TRAIN_DAYS, WEEKDAY
from core.runtime import log
from task2.ongoing.common import FIRST_SLOT, MIN_COVERAGE, Histories
from task2.onset_mix import save_probs

HORIZON = 6
STEPS = 13


def centres(p):
    return (np.asarray(p.x_start) + np.asarray(p.x_end)) / 2.0


def near(p, km):
    x = centres(p)
    return np.abs(x[:, None] - x[None, :]) <= km


def train_day(p, day, slots):
    idx = slots[:, None] + np.arange(-STEPS + 1, 1)[None]
    hist = {}
    for f in FIELDS:
        values = p.eligible("train", f)[day]
        at = p.eligible("public", f)[day]
        hist[f] = np.concatenate([values[idx[:, :-1]], at[slots][:, None]], 1)
    pct = np.concatenate([np.asarray(p.grid["train_pct"][day])[idx[:, :-1]], np.asarray(p.grid["public_pct"][day])[slots][:, None]], 1)
    on, off = np.asarray(p.grid["ramp_or"][day])[idx], np.asarray(p.grid["ramp_fr"][day])[idx]
    label = p.eligible("train")[day][slots + HORIZON]
    return hist, pct, on, off, (label <= p.threshold) & np.isfinite(label), np.isfinite(label)


def qualifying(p, speed, v, close, min_steps):
    queued = (np.isfinite(speed) & (speed <= p.threshold)).any(1)
    blocked = (queued.astype(np.float32) @ close.astype(np.float32)) > 0
    return ~blocked & (np.isfinite(speed).sum(1) >= min_steps) & v


def example_days(c, o):
    days = np.flatnonzero((MONTH[:TRAIN_DAYS] >= c["fit_months"][0]) & (MONTH[:TRAIN_DAYS] <= c["fit_months"][1]))
    return days[:: o["local"].get("day_step", 1)]


def origins(c):
    return np.arange(FIRST_SLOT, SLOTS_PER_DAY - HORIZON, c["onset_signs"]["local"]["origin_every"])


def onset_slots(speed, threshold):
    valid = np.isfinite(speed)
    queued = valid & (speed <= threshold)
    n = speed.shape[1]
    cq = np.vstack([np.zeros((1, n), int), queued.cumsum(0)])
    cv = np.vstack([np.zeros((1, n), int), valid.cumsum(0)])
    slots = np.arange(FIRST_SLOT, SLOTS_PER_DAY - HORIZON)
    stable = ((cq[slots] - cq[slots - 12]) >= 2).any(1)
    coverage = (cv[slots] - cv[slots - 12]).sum(1) / (12 * n)
    ahead = queued[slots[:, None] + np.arange(1, HORIZON + 1)]
    base = ~stable & ahead.any((1, 2)) & (coverage >= MIN_COVERAGE)
    first = base & ~np.r_[False, base[:-1]]
    return slots[first & ~ahead[:, : HORIZON - 1].any((1, 2)) & ahead[:, HORIZON - 1].any(1)]


def pseudo(c, p):
    speed, rows = p.eligible("train"), []
    for day in example_days(c, c["onset_signs"]):
        rows += [dict(day=int(day), slot=int(s)) for s in onset_slots(speed[day], p.threshold)]
    return frame(p.name, pd.DataFrame(rows, columns=["day", "slot"]), "pseudo")


def frame(panel, rows, split):
    rows = rows.copy()
    rows["panel"], rows["family_id"], rows["split"] = panel, panel.rsplit("_", 1)[0], split
    rows["date"] = [str(DATES[d].date()) for d in rows.day]
    rows["month"] = MONTH[rows.day.to_numpy()] if len(rows) else []
    rows["forecast_origin"] = [(DATES[d] + pd.Timedelta(minutes=5 * s)).tz_localize("UTC").isoformat() for d, s in zip(rows.day, rows.slot)]
    if "window_id" not in rows:
        rows["window_id"] = [f"P_{panel}_{DATES[d]:%Y%m%d}_{s:03d}_onset" for d, s in zip(rows.day, rows.slot)]
    return rows.reset_index(drop=True)


def official(c, p):
    frames = []
    for split in ("validation", "private"):
        f = pd.read_csv(Path(c["data"]) / "task2" / p.name / split / "window_index.csv")
        f = f[f.condition == "queue_onset"].copy()
        origin = pd.to_datetime(f.forecast_origin, utc=True)
        f["day"] = (origin.dt.tz_localize(None).dt.normalize() - DATES[0]).dt.days.to_numpy()
        f["slot"] = (origin.dt.hour * 12 + origin.dt.minute // 5).to_numpy()
        f["split"] = split
        frames.append(f[["window_id", "day", "slot", "split", "forecast_origin"]])
    f = pd.concat(frames, ignore_index=True)
    out = frame(p.name, f[["window_id", "day", "slot"]], "official")
    out["split"], out["forecast_origin"] = f.split.to_numpy(), f.forecast_origin.to_numpy()
    return out


HISTORY_KIND = {"pseudo": "train", "official": "official"}


def window_histories(c, p, rows, kind):
    if not len(rows):
        empty = np.zeros((0, STEPS, p.n), np.float32)
        return {f: empty for f in FIELDS}, empty, empty, empty
    h = Histories(c, p, rows, HISTORY_KIND[kind])
    hist, pct, on, off = {f: [] for f in FIELDS}, [], [], []
    for k in range(len(rows)):
        values, share = h.window(k, True)
        for f in FIELDS:
            hist[f].append(values[f])
        pct.append(share)
        a, b = h.ramps(k)
        on.append(a)
        off.append(b)
    return {f: np.stack(v) for f, v in hist.items()}, np.stack(pct), np.stack(on), np.stack(off)


def labels(p, rows, source):
    speed = p.eligible(source)[rows.day.to_numpy(), rows.slot.to_numpy() + HORIZON]
    return (speed <= p.threshold) & np.isfinite(speed), np.isfinite(speed)


def settings(c):
    o = copy.deepcopy(c["onset_signs"])
    if c["trial"]:
        t = c["trial_settings"]["onset_signs"]
        o["gbm"]["rounds"] = t["rounds"]
        o["dcrnn"]["seeds"] = o["dcrnn"]["seeds"][:1]
        o["dcrnn"]["base"].update(epochs=1, steps_per_epoch=t["steps_per_epoch"], teacher_epochs=1, warmup_epochs=0)
        o["dcrnn"]["tune"].update(epochs=1, windows_per_panel=t["windows_per_panel"])
        o["local"]["day_step"] = t["day_step"]
    return o


def predicted(c, p):
    return {"pseudo": pseudo(c, p), "official": official(c, p)}


def calibrate(logits, y, v):
    z = np.concatenate([np.asarray(a, np.float64)[m] for a, m in zip(logits, v)])
    target = float(sum(int((a & m).sum()) for a, m in zip(y, v)))
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1.0 / (1.0 + np.exp(-(z + mid)))).sum() < target:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    log(f"  [calib]   onset pseudo-windows: {len(logits):,} | queued links {target:.0f} | logit shift {b:+.3f}")
    return b


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def save_calibrated(c, run_id, name, panels, logits, threshold):
    pseudo_z, pseudo_y, pseudo_v = [], [], []
    for rows, z in logits["pseudo"]:
        if len(rows):
            y, v = labels(panels[rows.panel.iloc[0]], rows, "train")
            pseudo_z += list(z)
            pseudo_y += list(y)
            pseudo_v += list(v)
    shift = calibrate(pseudo_z, pseudo_y, pseudo_v)
    parts = [(rows, z) for rows, z in logits["official"] if len(rows)]
    rows = pd.concat([r for r, _ in parts], ignore_index=True)
    probs = [q for _, z in parts for q in sigmoid(z + shift)]
    save_probs(
        c,
        run_id,
        name,
        list(zip(rows.window_id, rows.split, rows.panel, rows.forecast_origin)),
        probs,
        {n: list(p.ids) for n, p in panels.items()},
    )
    log(
        f"  [predict] official: {len(rows)} windows | links with p >= {threshold}: {np.mean([(q >= threshold).sum() for q in probs]):.2f} per window"
    )


GROUPS = {
    "now": ["now_speed", "now_speed_drop", "now_occ", "now_flow"],
    "dev": ["dev_speed", "dev_flow"],
    "phy": ["phy_density_ratio", "phy_load", "phy_down_speed", "phy_up_flow_trend"],
    "net": ["net_bottleneck"],
    "win": ["win_rank_dev_speed", "win_rank_load"],
    "cal": ["cal_slot"],
}
COLUMNS = [name for names in GROUPS.values() for name in names]

HISTORY = [
    "now_speed",
    "now_flow",
    "now_occ",
    "now_queue",
    "dev_speed",
    "obs_mask",
    "obs_age",
    "obs_quality",
    "ramp_on",
    "ramp_off",
    "ramp_mask",
]
CONTEXT = ["net_bottleneck", "net_on_ramp", "net_off_ramp", "cal_sin", "cal_cos"]
DECODER = ["cal_sin", "cal_cos", "cal_step"]
H = {name: i for i, name in enumerate(HISTORY)}

DOWN_HOPS = 3
DOWN_KM = 3.0
TREND_STEPS = 6
AGE_CAP = 12


def quiet(function):
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return function(*args, **kwargs)

    return wrapped


class Normal:
    def __init__(self, p):
        speed = p.eligible("train") / p.free
        flow = p.eligible("train", "flow_vph") / p.capacity
        months = sorted(set(MONTH[:TRAIN_DAYS]))
        self.months, self.sums = months, []
        for month in months:
            days = np.flatnonzero(MONTH[:TRAIN_DAYS] == month)
            s = np.zeros((2, 7, SLOTS_PER_DAY, p.n))
            n = np.zeros((7, SLOTS_PER_DAY, p.n))
            for wd in range(7):
                d = days[WEEKDAY[days] == wd]
                if len(d):
                    ok = np.isfinite(speed[d]) & np.isfinite(flow[d])
                    s[0, wd] = np.where(ok, speed[d], 0.0).sum(0)
                    s[1, wd] = np.where(ok, flow[d], 0.0).sum(0)
                    n[wd] = ok.sum(0)
            self.sums.append((s, n))
        self._cache = {}

    def before(self, day):
        month = MONTH[min(day, TRAIN_DAYS - 1)] if day < TRAIN_DAYS else None
        k = self.months.index(month) if month is not None else len(self.months)
        if k not in self._cache:
            s = sum((x for x, _ in self.sums[:k]), np.zeros_like(self.sums[0][0]))
            n = sum((m for _, m in self.sums[:k]), np.zeros_like(self.sums[0][1]))
            with np.errstate(invalid="ignore", divide="ignore"):
                self._cache[k] = (
                    np.where(n > 0, s[0] / n, np.nan).astype(np.float32),
                    np.where(n > 0, s[1] / n, np.nan).astype(np.float32),
                )
        return self._cache[k]


class Around:
    def __init__(self, p):
        index, hops = p.down
        x = centres(p)
        ok = index >= 0
        cap = np.where(ok & (hops <= DOWN_HOPS), p.capacity[np.clip(index, 0, None)], np.nan)
        self.down_capacity = quiet(np.nanmin)(cap, 1)
        self.bottleneck = self.down_capacity / p.capacity
        self.down = np.where(ok & (np.abs(x[np.clip(index, 0, None)] - x[:, None]) <= DOWN_KM), index, -1)
        self.up = p.up[0]
        self.on, self.off = (p.n_on > 0).astype(np.float32), (p.n_off > 0).astype(np.float32)


def first_valued(index, values):
    v = np.where(index[None] >= 0, values[:, np.clip(index, 0, None)], np.nan)
    has = np.isfinite(v)
    pick = has.argmax(2)
    out = np.take_along_axis(v, pick[..., None], 2)[..., 0]
    return np.where(has.any(2), out, np.nan)


def last_valid(a):
    ok = np.isfinite(a)
    steps = a.shape[1]
    at = steps - 1 - ok[:, ::-1].argmax(1)
    value = np.take_along_axis(a, at[:, None], 1)[:, 0]
    has = ok.any(1)
    return np.where(has, value, np.nan), np.where(has, steps - 1 - at, steps)


@quiet
def slope(a):
    x = np.arange(a.shape[1], dtype=np.float64)[None, :, None]
    ok = np.isfinite(a)
    n = ok.sum(1)
    mx = np.where(ok, x, 0.0).sum(1) / np.maximum(n, 1)
    my = np.nanmean(a, 1)
    num = np.where(ok, (x - mx[:, None]) * (a - my[:, None]), 0.0).sum(1)
    den = np.where(ok, (x - mx[:, None]) ** 2, 0.0).sum(1)
    return np.where((n >= 2) & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)


def window_rank(values, larger_first):
    ok = np.isfinite(values)
    key = np.where(ok, -values if larger_first else values, np.inf)
    rank = key.argsort(1, kind="stable").argsort(1, kind="stable") + 1.0
    return np.where(ok, rank / np.maximum(ok.sum(1, keepdims=True), 1), np.nan)


@quiet
def columns(p, around, normal, days, slots, hist, on, off):
    speed = hist["speed_kmh"] / p.free
    flow = hist["flow_vph"] / p.capacity
    occ = hist["occupancy"]
    now_speed, now_flow = last_valid(speed)[0], last_valid(flow)[0]
    usual = [normal.before(int(d)) for d in days]
    wd = WEEKDAY[np.asarray(days)]
    dev_speed = now_speed - np.stack([u[0][w, s] for u, w, s in zip(usual, wd, slots)])
    dev_flow = now_flow - np.stack([u[1][w, s] for u, w, s in zip(usual, wd, slots)])
    both = np.isfinite(hist["speed_kmh"]) & np.isfinite(hist["flow_vph"])
    k = last_valid(np.where(both, hist["flow_vph"] / np.maximum(hist["speed_kmh"], MIN_SPEED_KMH), np.nan))[0]
    q = hist["flow_vph"]
    q_up = first_valued(around.up, last_valid(q)[0])
    ramps = np.nan_to_num(last_valid(on)[0]) - np.nan_to_num(last_valid(off)[0])
    load = (q_up + ramps) / around.down_capacity[None]
    down_speed = np.nanmin(np.where(around.down[None] >= 0, now_speed[:, np.clip(around.down, 0, None)], np.nan), 2)
    up_trend = first_valued(around.up, slope(flow[:, -TREND_STEPS:]))
    col = dict(
        now_speed=now_speed,
        now_speed_drop=now_speed - np.nanmax(speed, 1),
        now_occ=last_valid(occ[:, -3:])[0],
        now_flow=now_flow,
        dev_speed=dev_speed,
        dev_flow=dev_flow,
        phy_density_ratio=k / p.k_crit[None],
        phy_load=load,
        phy_down_speed=down_speed,
        phy_up_flow_trend=up_trend,
        net_bottleneck=np.broadcast_to(around.bottleneck[None], now_speed.shape),
        win_rank_dev_speed=window_rank(dev_speed, False),
        win_rank_load=window_rank(load, True),
        cal_slot=np.broadcast_to(np.asarray(slots, np.float64)[:, None], now_speed.shape),
    )
    out = np.stack([col[name] for name in COLUMNS], -1).astype(np.float32)
    if np.isinf(out).any():
        raise ValueError("infinite onset column")
    return out


def forward_fill(a):
    ok = np.isfinite(a)
    steps = a.shape[1]
    at = np.maximum.accumulate(np.where(ok, np.arange(steps)[None, :, None], -1), 1)
    filled = np.take_along_axis(np.where(ok, a, 0.0), np.clip(at, 0, None), 1)
    return np.where(at >= 0, filled, 0.0), np.where(at >= 0, np.arange(steps)[None, :, None] - at, steps)


@quiet
def channels(p, around, normal, days, slots, hist, pct, on, off):
    w, steps = len(slots), hist["speed_kmh"].shape[1]
    speed = hist["speed_kmh"] / p.free
    out = np.zeros((w, steps, p.n, len(HISTORY)), np.float32)
    filled, age = forward_fill(speed)
    out[..., H["now_speed"]] = filled
    out[..., H["now_flow"]] = forward_fill(hist["flow_vph"] / p.capacity)[0]
    out[..., H["now_occ"]] = forward_fill(hist["occupancy"])[0]
    valid = np.isfinite(speed)
    out[..., H["now_queue"]] = valid & (np.nan_to_num(speed, nan=9.0) <= 0.6)
    wd = WEEKDAY[np.asarray(days)]
    step_slots = np.asarray(slots)[:, None] + np.arange(-steps + 1, 1)[None]
    usual = np.stack([normal.before(int(d))[0][w_, s_] for d, w_, s_ in zip(days, wd, step_slots)])
    out[..., H["dev_speed"]] = np.where((age < steps) & np.isfinite(usual), filled - usual, 0.0)
    out[..., H["obs_mask"]] = valid
    out[..., H["obs_age"]] = np.minimum(age, AGE_CAP) / AGE_CAP
    out[..., H["obs_quality"]] = np.where(valid, pct / 100.0, 0.0)
    out[..., H["ramp_on"]] = forward_fill(on / p.capacity)[0]
    out[..., H["ramp_off"]] = forward_fill(off / p.capacity)[0]
    out[..., H["ramp_mask"]] = np.isfinite(on) & np.isfinite(off)
    angle = 2 * np.pi * np.asarray(slots, np.float64) / SLOTS_PER_DAY
    ctx = np.zeros((w, p.n, len(CONTEXT)), np.float32)
    ctx[..., 0] = np.nan_to_num(around.bottleneck, nan=1.0)[None]
    ctx[..., 1], ctx[..., 2] = around.on[None], around.off[None]
    ctx[..., 3], ctx[..., 4] = np.sin(angle)[:, None], np.cos(angle)[:, None]
    target = 2 * np.pi * (np.asarray(slots)[:, None] + np.arange(7)[None]) / SLOTS_PER_DAY
    dec = np.zeros((w, 7, p.n, len(DECODER)), np.float32)
    dec[..., 0], dec[..., 1] = np.sin(target)[..., None], np.cos(target)[..., None]
    dec[..., 2] = (np.arange(7) / 6.0)[None, :, None]
    last = last_valid(speed)[0]
    start = np.where(np.isfinite(last), np.where(last <= 0.6, 1.0, -1.0), 0.0).astype(np.float32)
    return out, ctx, dec, start
