"""工程 onset_place_gbm（CPU）：過去の onset の頻度を読む LightGBM を学習し、公式の onset 窓の T+30 確率を出す。"""

from __future__ import annotations

import fcntl
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import work_path
from core.runtime import log, peak_gb, section
from task2.onset_mix import save_probs
from task2.onset_place import data as grid
from task2.onset_place.data import DATES, TRAIN_DAYS, Grid, day_candidates, history_arrays, labels, official, settings, with_origin

GROUPS = {
    "last 60 min": ["speed_last", "speed_mean", "speed_min", "speed_slope", "flow_last", "age_minutes", "up_speed", "down_speed"],
    "past freq": ["freq_wd_all", "freq_wd_8", "freq_any_14", "freq_origin"],
    "ramp": ["ramp_ratio", "ramp_change"],
    "static": ["length", "has_on_ramp", "has_off_ramp"],
    "calendar": ["origin_slot", "weekday"],
    "last 15 min": ["occupancy_last_15", "self_speed_slope_15", "self_flow_slope_15", "self_valid_count_15"],
    "onset freq": ["onset_freq_near", "onset_freq_wd", "onset_freq_all"],
    "in-window": ["freq_rank", "freq_ratio", "freq_sum", "n_links"],
    "timing": ["onset_slot_gap", "queued_day_fraction"],
    "at T": ["speed_change_T", "flow_change_T"],
}

FEATURES = [name for names in GROUPS.values() for name in names]

EXPECTED = {"cv_fit": (1863, 269553, 5736), "cv_eval": (235, 34182, 705), "full_fit": (2098, 303735, 6441)}


def nanstat(a, kind="mean", axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return getattr(np, "nan" + kind)(a, axis=axis)


def frequency(q, valid, days, slot, pool=1, min_days=4):
    slots = np.arange(max(0, slot - pool), min(288, slot + pool + 1))
    v = valid[np.ix_(days, slots, np.arange(q.shape[2]))].any(axis=1)
    y = q[np.ix_(days, slots, np.arange(q.shape[2]))].any(axis=1)
    n = v.sum(axis=0)
    s = y.sum(axis=0)
    result = np.divide(s, n, out=np.full(q.shape[2], np.nan), where=n >= min_days)
    return result, n


def last_valid(a):
    valid = np.isfinite(a)
    has = valid.any(axis=0)
    index = a.shape[0] - 1 - valid[::-1].argmax(axis=0)
    value = a[index, np.arange(a.shape[1])].copy()
    value[~has] = np.nan
    age = (a.shape[0] - 1 - index).astype(float) * 5
    age[~has] = np.nan
    return value, age


def slope(a):
    x = np.arange(len(a))[:, None] * 5.0
    valid = np.isfinite(a)
    count = valid.sum(axis=0)
    mx = np.divide((valid * x).sum(axis=0), count, out=np.zeros(a.shape[1]), where=count > 0)
    my = nanstat(a)
    num = np.nansum((x - mx) * (a - my), axis=0)
    den = ((x - mx) ** 2 * valid).sum(axis=0)
    return np.divide(num, den, out=np.full(a.shape[1], np.nan), where=(count >= 2) & (den > 0))


def nearest(neighbours, values):
    result = np.full(len(neighbours), np.nan)
    indices = np.full(len(neighbours), -1, int)
    for i, candidates in enumerate(neighbours):
        for j, hops in candidates:
            if np.isfinite(values[j]):
                result[i] = values[j]
                indices[i] = j
                break
    return result, indices


class PastWindows:
    def __init__(self, days, slots, q, valid, weekday):
        self.days = np.asarray(days, dtype=int)
        self.slots = np.asarray(slots, dtype=int)
        self.weekday = weekday[self.days] if len(self.days) else np.zeros(0, int)
        self.queued = q[self.days, self.slots + 6].astype(np.float64)
        self.valid = valid[self.days, self.slots + 6].astype(np.float64)

    def share(self, keep):
        if not keep.any():
            return np.full(self.queued.shape[1], np.nan)
        n = self.valid[keep].sum(axis=0)
        return np.divide(self.queued[keep].sum(axis=0), n, out=np.full(len(n), np.nan), where=n > 0)


def build(net, weekday, q, valid, first, ramp, past, day, slot, history, near_slots=12):
    n = len(net["ids"])
    speed, flow, occupancy = (np.asarray(history[k], dtype=np.float64) for k in ("speed_kmh", "flow_vph", "occupancy"))
    if speed.shape != (13, n):
        raise ValueError("the history must hold thirteen slots per link")
    vr, fr = speed / net["free"], flow / net["capacity"]
    last, age = last_valid(vr)

    before = np.arange(min(day, TRAIN_DAYS))
    same = before[weekday[before] == weekday[day]]
    target = slot + 6

    def freq(days, at):
        return frequency(q, valid, days, at, 1, 1)[0] if len(days) else np.full(n, np.nan)

    freq_wd = freq(same, target)
    filled = np.nan_to_num(freq_wd, nan=0.0)
    top = filled.max()
    recent = np.asarray(ramp[day, slot - 5 : slot + 1], dtype=np.float64)
    ramp_now = nanstat(recent[-3:]) / net["capacity"]
    ramp_before = nanstat(recent[:3]) / net["capacity"]
    starts = first[same] if len(same) else np.full((1, n), np.nan)
    earlier = past.days < min(day, TRAIN_DAYS)
    both = np.isfinite(speed[-3:]) & np.isfinite(flow[-3:])

    columns = dict(
        speed_last=last,
        speed_mean=nanstat(vr),
        speed_min=nanstat(vr, "min"),
        speed_slope=slope(vr),
        flow_last=last_valid(fr)[0],
        age_minutes=age,
        up_speed=nearest(net["up"], last)[0],
        down_speed=nearest(net["down"], last)[0],
        freq_wd_all=freq_wd,
        freq_wd_8=freq(same[-8:], target),
        freq_any_14=freq(before[-14:], target),
        freq_origin=freq(same, slot - 1),
        ramp_ratio=ramp_now,
        ramp_change=ramp_now - ramp_before,
        length=net["static"][:, 1],
        has_on_ramp=net["static"][:, 4],
        has_off_ramp=net["static"][:, 5],
        origin_slot=np.full(n, float(slot)),
        weekday=np.full(n, float(weekday[day])),
        occupancy_last_15=last_valid(occupancy[-3:])[0],
        self_speed_slope_15=slope(vr[-3:]),
        self_flow_slope_15=slope(fr[-3:]),
        self_valid_count_15=both.sum(axis=0).astype(float),
        onset_freq_near=past.share(earlier & (np.abs(past.slots - slot) <= near_slots)),
        onset_freq_wd=past.share(earlier & (past.weekday == weekday[day])),
        onset_freq_all=past.share(earlier),
        freq_rank=(-filled).argsort(kind="stable").argsort().astype(float),
        freq_ratio=filled / top if top > 0 else np.zeros(n),
        freq_sum=np.full(n, filled.sum()),
        n_links=np.full(n, float(n)),
        onset_slot_gap=target - nanstat(starts, "median"),
        queued_day_fraction=np.isfinite(starts).mean(axis=0) if len(same) else np.full(n, np.nan),
        speed_change_T=vr[-1] - last_valid(vr[:-1])[0],
        flow_change_T=fr[-1] - last_valid(fr[:-1])[0],
    )
    result = np.column_stack([columns[name] for name in FEATURES]).astype(np.float32)
    if result.shape != (n, len(FEATURES)) or np.isinf(result).any():
        raise ValueError("invalid onset feature matrix")
    return result


def out(c, *parts):
    path = Path(c["gbm_cache"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


WEEKDAY = np.array([d.weekday() for d in DATES])

MONTH = np.array([d.strftime("%Y-%m") for d in DATES])


class Panel:
    def __init__(self, c, name):
        self.name = name
        self.grid = Grid(c, name)
        self.net = self.grid.net
        self.n = self.grid.n
        self.q, self.valid = labels(np.asarray(self.grid.raw["speed_kmh"]), self.net["threshold"])
        self.first = np.where(self.q.any(axis=1), self.q.argmax(axis=1), np.nan).astype(float)
        days, slots = [], []
        for day in range(TRAIN_DAYS):
            found = day_candidates(self.grid.raw["speed_kmh"][day], self.net["threshold"])
            for slot in found["slots"][found["onset"]]:
                days.append(day)
                slots.append(int(slot))
        self.days, self.slots = np.array(days, dtype=int), np.array(slots, dtype=int)
        self.past = PastWindows(self.days, self.slots, self.q, self.valid, WEEKDAY)

    def features(self, c, window):
        history = with_origin(history_arrays(c, self.grid, window), self.grid.masked, int(window.day), int(window.slot))
        return build(
            self.net,
            WEEKDAY,
            self.q,
            self.valid,
            self.first,
            self.grid.ramp,
            self.past,
            int(window.day),
            int(window.slot),
            history,
            c["near_slots"],
        )


def pseudo_windows(panel):
    rows = []
    for day, slot in zip(panel.days, panel.slots):
        origin = (DATES[day] + pd.Timedelta(minutes=int(slot) * 5)).tz_localize("UTC")
        rows.append(
            dict(
                panel=panel.name,
                family_id=panel.name.rsplit("_", 1)[0],
                window_id=f"P_{panel.name}_{DATES[day]:%Y%m%d}_{slot:03d}_onset",
                month=MONTH[day],
                day=int(day),
                slot=int(slot),
                date=str(DATES[day].date()),
                split="pseudo",
                forecast_origin=origin.isoformat(),
            )
        )
    return pd.DataFrame(rows)


def official_windows(c, panel):
    frame = official(c, panel.name)
    frame = frame[frame.split.isin(["validation", "private"]) & frame.condition.eq("onset")]
    return frame[["panel", "family_id", "window_id", "month", "day", "slot", "date", "split", "forecast_origin"]].copy()


def phase_of(month, split):
    if split != "pseudo":
        return "official"
    if month < "2030-07":
        return "history_only"
    return "cv_eval" if month == "2031-02" else "cv_fit"


def prepare(c):
    windows, blocks, rows = [], [], []
    for name in c["panels"]:
        panel = Panel(c, name)
        frame = pd.concat([pseudo_windows(panel), official_windows(c, panel)], ignore_index=True)
        frame["phase"] = [phase_of(m, s) for m, s in zip(frame.month, frame.split)]
        frame = frame[frame.phase.ne("history_only")].reset_index(drop=True)
        for window in frame.itertuples():
            x = panel.features(c, window)
            if window.split == "pseudo":
                target = panel.q[window.day, window.slot + 6].astype(np.int8)
                eligible = panel.valid[window.day, window.slot + 6]
            else:
                target = np.full(panel.n, -1, np.int8)
                eligible = np.zeros(panel.n, bool)
            blocks.append(x)
            rows.append(
                pd.DataFrame(
                    dict(
                        window_id=window.window_id, link_id=panel.net["ids"], position=np.arange(panel.n), target=target, eligible=eligible
                    )
                )
            )
        windows.append(frame)
        log(f"  [cache]  {name}: {panel.n} links | pseudo onset windows: {len(panel.days):,} (all Train days) | featured: {len(frame):,}")
    windows = pd.concat(windows, ignore_index=True)
    rows = pd.concat(rows, ignore_index=True)
    x = np.concatenate(blocks).astype(np.float32)
    if len(rows) != len(x) or rows.duplicated(["window_id", "link_id"]).any():
        raise ValueError("feature rows and keys disagree")
    windows.to_parquet(out(c, "windows.parquet"), index=False)
    rows.to_parquet(out(c, "rows.parquet"), index=False)
    np.save(out(c, "X.npy"), x)
    return check_counts(c, windows, rows)


def load_cache(c):
    windows = pd.read_parquet(out(c, "windows.parquet"))
    rows = pd.read_parquet(out(c, "rows.parquet"))
    x = np.load(out(c, "X.npy"))
    rows = rows.merge(
        windows[["window_id", "panel", "family_id", "phase", "split", "month", "day", "slot"]],
        on="window_id",
        how="left",
        validate="many_to_one",
    )
    return windows, rows, x


def check_counts(c, windows, rows):
    merged = rows.merge(windows[["window_id", "phase"]], on="window_id")
    counts = {}
    for name, phases in (("cv_fit", ["cv_fit"]), ("cv_eval", ["cv_eval"]), ("full_fit", ["cv_fit", "cv_eval"])):
        part = merged[merged.phase.isin(phases) & merged.eligible]
        counts[name] = (int(windows.phase.isin(phases).sum()), len(part), int(part.target.sum()))
    if not c["trial"]:
        if counts != EXPECTED:
            raise ValueError(f"window/label counts differ from the plan: {counts}")
    return {k: dict(windows=v[0], rows=v[1], queued=v[2]) for k, v in counts.items()}


def build_cache(c):
    root = Path(c["gbm_cache"])
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "cache.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "X.npy").exists():
            return
        counts = prepare(c)
        log(
            "  [cache]   "
            + " | ".join(f"{k}: {v['windows']:,} windows, {v['rows']:,} rows, {v['queued']:,} queued" for k, v in counts.items())
        )


COL = {name: i for i, name in enumerate(FEATURES)}

ADAPT_DAYS = 3.0

HALF_WIDTH = 6


def month_freq(masked_speed, threshold, day, slot, last_days=None, since=None):
    first = int(np.flatnonzero(MONTH == MONTH[day])[0])
    if since is not None:
        first = min(first, int(since))
    if last_days is not None:
        first = max(first, day - int(last_days))
    n_links = masked_speed.shape[2]
    if first >= day:
        return np.full(n_links, np.nan), np.zeros(n_links)
    target = slot + 6
    x = np.asarray(masked_speed[first:day, max(0, target - HALF_WIDTH) : min(288, target + HALF_WIDTH + 1)])
    ok = np.isfinite(x)
    seen = ok.any(1).sum(0).astype(float)
    hit = (ok & (x <= threshold)).any(1).sum(0)
    return np.divide(hit, seen, out=np.full(n_links, np.nan), where=seen > 0), seen


def adapt(near, freq, n):
    k = n / (n + ADAPT_DAYS)
    moved = (1.0 - k) * np.nan_to_num(near) + k * np.nan_to_num(freq)
    return np.where(n > 0, moved, near)


NEAR = COL["onset_freq_near"]


def describe(p, info, x, y, v):
    x = np.asarray(x, np.float32)
    speed, threshold = p.grid.masked["speed_kmh"], p.net["threshold"]
    freq, n = month_freq(speed, threshold, info["day"], info["slot"])
    return dict(**info, x=x, y=np.asarray(y, bool), v=np.asarray(v, bool), near=np.nan_to_num(x[:, NEAR]), freq=freq, n=n)


def cached_windows(c, panels, phases):
    windows, rows, x = load_cache(c)
    first = rows.reset_index().groupby("window_id", sort=False)["index"].first()
    out = {name: [] for name in c["panels"]}
    for w in windows[windows.phase.isin(phases)].itertuples():
        p = panels[w.panel]
        a = int(first[w.window_id])
        block = rows.iloc[a : a + p.n]
        assert (block.window_id == w.window_id).all() and block.position.tolist() == list(range(p.n))
        info = dict(
            window_id=w.window_id, panel=w.panel, day=int(w.day), slot=int(w.slot), split=w.split, forecast_origin=w.forecast_origin
        )
        out[w.panel].append(describe(p, info, x[a : a + p.n], block.target.to_numpy() == 1, block.eligible.to_numpy()))
    return out


def flat(by_panel):
    return [w for ws in by_panel.values() for w in ws]


def meta(windows):
    return [(w["window_id"], w["split"], w["panel"], w["forecast_origin"]) for w in windows]


def window_weights(panel, window_id):
    frame = pd.DataFrame(dict(panel=np.asarray(panel), window_id=np.asarray(window_id)))
    rows = frame.groupby("window_id").window_id.transform("size")
    windows = frame.groupby("panel").window_id.transform("nunique")
    weight = 1.0 / (windows * rows)
    return (weight / weight.mean()).to_numpy(dtype=np.float64)


def columns(w):
    x = w["x"].copy()
    x[:, NEAR] = adapt(w["x"][:, NEAR], w["freq"], w["n"])
    return x


def feature_names():
    return ["onset_freq_adapt" if n == "onset_freq_near" else n for n in FEATURES]


def training_matrix(train):
    blocks, ys, panel_ids, window_ids = [], [], [], []
    for w in train:
        block, keep = columns(w), w["v"]
        blocks.append(block[keep])
        ys.append(w["y"][keep])
        panel_ids += [w["panel"]] * int(keep.sum())
        window_ids += [w["window_id"]] * int(keep.sum())
    return np.concatenate(blocks).astype(np.float32), np.concatenate(ys).astype(np.int8), np.array(panel_ids), np.array(window_ids)


def fit_gbm(c, train, seed, path):
    values, y, panel_ids, window_ids = training_matrix(train)
    data = lgb.Dataset(values, y, weight=window_weights(panel_ids, window_ids), feature_name=feature_names(), free_raw_data=False)
    history = {}
    model = lgb.train(
        {**c["lightgbm"], "learning_rate": c["learning_rate"], "seed": seed},
        data,
        num_boost_round=c["rounds"],
        valid_sets=[data],
        valid_names=["train"],
        callbacks=[lgb.record_evaluation(history)],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return model, dict(rows=len(y), loss=float(history["train"]["binary_logloss"][-1]))


def predict_gbm(models, windows):
    return [np.mean([m.predict(columns(w), num_threads=4) for m in models], 0) for w in windows]


NAME = "onset_place_gbm"


def windows(o, panels):
    train = flat(cached_windows(o, panels, ["cv_fit", "cv_eval"]))
    official = flat(cached_windows(o, panels, ["official"]))
    rows, queued = sum(int(w["v"].sum()) for w in train), sum(int((w["y"] & w["v"]).sum()) for w in train)
    assert (len(train), rows, queued, len(official)) == (2098, 303735, 6441, 80), (len(train), rows, queued, len(official))
    log(f"  [data]    train: {len(train):,} windows, {rows:,} rows, {queued:,} queued | official: {len(official)} windows")
    return train, official


def run(c, run_id):
    o = settings(c)
    section(NAME)
    started = time.monotonic()
    grid.build(o)
    build_cache(o)
    panels = {n: Panel(o, n) for n in o["panels"]}
    ids = {n: list(p.net["ids"]) for n, p in panels.items()}
    train, official = windows(o, panels)
    path = work_path(c, run_id, f"{NAME}_seed{o['seed']}.txt")
    if path.exists():
        model = lgb.Booster(model_file=str(path))
        log(f"  [model]   loaded {path}")
    else:
        model, info = fit_gbm(o, train, o["seed"], path)
        log(f"  [model]   {info['rows']:,} rows | loss: tr {info['loss']:.5f} | {time.monotonic() - started:.0f}s")
    probs = predict_gbm([model], official)
    save_probs(c, run_id, NAME, meta(official), probs, ids)
    gain = model.feature_importance("gain")
    log(
        "  [gain]    "
        + " | ".join(f"{n}: {g:.1%}" for n, g in sorted(zip(feature_names(), gain / gain.sum()), key=lambda t: -t[1])[:5])
        + f" | RSS peak {peak_gb():.1f} GB"
    )
