"""onset_place の共通部分：設定、パネルの配列、onset・ongoing の窓。"""

from __future__ import annotations

import copy
import fcntl
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import write_json
from core.runtime import log
from data.release import Road, attach


def settings(c):
    o = copy.deepcopy(c["onset_place"])
    root = Path(c["cache"]) / "onset_place"
    s = dict(
        data=c["data"],
        panels=c["panels"],
        trial=c["trial"],
        threshold=o["threshold"],
        windows=o["windows"],
        fit_months=c["fit_months"],
        grid_dir=str(root / "grid"),
        windows_dir=str(root / "windows"),
        gbm_cache=str(root / "gbm"),
        ramp_dir=str(root / "ramp"),
        grid_all_dir=str(root / "grid_all"),
        cache_dir=str(root / ("dcrnn_trial" if c["trial"] else "dcrnn")),
        **o["gbm"],
        **o["dcrnn"],
    )
    if c["trial"]:
        t = c["trial_settings"]["onset_place"]
        s["rounds"], s["seeds"] = t["rounds"], s["seeds"][:1]
        s["stage2"].update(epochs=t["epochs"], teacher_epochs=1)
        s["windows_per_panel"] = t["windows_per_panel"]
    return s


DATES = pd.date_range("2030-06-01", "2031-04-30", freq="D")

TRAIN_DAYS = 273

FIELDS = ["speed_kmh", "flow_vph", "occupancy"]


def load_network(root, panel, max_hops=10):
    data_root, root = root, Path(root) / "corridors" / panel / "network"
    links = pd.read_csv(root / "links.csv").sort_values("link_id").reset_index(drop=True)
    links.free_speed_kmh = links.free_speed_kmh.fillna(105.0)
    for col in ["free_speed_kmh", "capacity_vph", "lanes", "length_km"]:
        if not (np.isfinite(links[col]) & (links[col] > 0)).all():
            raise ValueError(f"Invalid {panel} {col}")
    ids = links.link_id.tolist()
    index = {v: i for i, v in enumerate(ids)}
    along = Road(data_root, panel, ids)
    down, up = defaultdict(set, along.chain()), defaultdict(set)
    for a, targets in down.items():
        for b in targets:
            up[b].add(a)

    def neighbours(graph):
        result = []
        for start in ids:
            seen = {start: 0}
            q = deque([start])
            while q:
                a = q.popleft()
                if seen[a] >= max_hops:
                    continue
                for b in sorted(graph[a]):
                    if b not in seen:
                        seen[b] = seen[a] + 1
                        q.append(b)
            result.append(sorted([(index[b], h) for b, h in seen.items() if b in index and h > 0], key=lambda x: (x[1], ids[x[0]])))
        return result

    def side(graph):
        return attach(neighbours(graph), along.anchor, ~along.mainline, max_hops)

    topo = pd.read_csv(root / "lwr_mainline_topology.csv").set_index("link_id").loc[ids]
    ramps = pd.read_csv(root / "ramp_attachment_map.csv")
    ramps = ramps[ramps.nearest_mainline_link_id.isin(ids)].copy()
    ramps["position"] = ramps.nearest_mainline_link_id.map(index)
    return dict(
        links=links,
        ids=ids,
        index=index,
        up=side(up),
        down=side(down),
        ramps=ramps,
        free=links.free_speed_kmh.to_numpy(),
        threshold=0.6 * links.free_speed_kmh.to_numpy(),
        capacity=links.capacity_vph.to_numpy(),
        static=np.column_stack(
            [
                links.lanes,
                links.length_km,
                links.free_speed_kmh,
                links.capacity_vph / links.lanes,
                topo.on_ramp_link_ids.notna(),
                topo.off_ramp_link_ids.notna(),
            ]
        ).astype(np.float32),
    )


def labels(speed, threshold):
    valid = np.isfinite(speed)
    return (speed <= threshold) & valid, valid


def coordinates(frame, index):
    slots = (frame.timestamp.str.slice(11, 13).astype(int) * 12 + frame.timestamp.str.slice(14, 16).astype(int) // 5).to_numpy()
    positions = frame.link_id.map(index)
    if positions.isna().any():
        raise ValueError("Unknown mainline link")
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError("Duplicate mainline cell")
    return slots, positions.to_numpy(dtype=int)


def prepare_panel(c, panel):
    directory = Path(c["grid_dir"]) / panel
    if (directory / "manifest.json").exists():
        log(f"data {panel}: existing complete grid cache")
        return
    start = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    net = load_network(c["data"], panel)
    n = len(net["ids"])
    arrays = {}
    for source, days in [("raw", TRAIN_DAYS), ("masked", len(DATES))]:
        for field in FIELDS:
            a = np.lib.format.open_memmap(directory / f"{source}_{field}.npy", mode="w+", dtype="float32", shape=(days, 288, n))
            a[:] = np.nan
            arrays[source, field] = a
    ramp = np.lib.format.open_memmap(directory / "ramp.npy", mode="w+", dtype="float32", shape=(len(DATES), 288, n))
    ramp[:] = np.nan
    root = Path(c["data"]) / "corridors" / panel
    on = net["ramps"].query("ramp_type == 'OR'").set_index("ramp_link_id").position
    attachment_counts = np.bincount(on.to_numpy(dtype=int), minlength=n)
    for d, date in enumerate(DATES):
        split = "train" if d < TRAIN_DAYS else ("validation" if date.month == 3 else "private")
        suffix = date.strftime("%Y_%m_%d")
        files = {}
        if split == "train":
            files["raw"] = root / split / "mainline_states" / f"year_month={date:%Y-%m}" / f"synthetic_mainline_{suffix}.parquet"
        matches = list((root / split / "mainline_states_masked").glob(f"*/synthetic_mainline_{suffix}.parquet"))
        if len(matches) != 1:
            raise ValueError(f"Expected one masked day {panel} {date}")
        files["masked"] = matches[0]
        for source, path in files.items():
            frame = pd.read_parquet(path, columns=["timestamp", "link_id", *FIELDS, "is_score_eligible", "pct_observed"])
            slots, ix = coordinates(frame, net["index"])
            eligible = (frame.is_score_eligible.eq(1) & frame.pct_observed.ge(75)).to_numpy()
            for field in FIELDS:
                values = frame[field].to_numpy(dtype=np.float32).copy()
                values[~eligible | ~np.isfinite(values)] = np.nan
                arrays[source, field][d, slots, ix] = values
        paths = list((root / split / "ramp_states").glob(f"*/synthetic_ramp_{suffix}.parquet"))
        if len(paths) != 1:
            raise ValueError(f"Expected ramp day {panel} {date}")
        frame = pd.read_parquet(paths[0], columns=["timestamp", "ramp_link_id", "flow_vph", "is_score_eligible", "pct_observed"])
        frame = frame[frame.ramp_link_id.isin(on.index)]
        totals = np.zeros((288, n), dtype=np.float32)
        counts = np.zeros((288, n), dtype=np.int16)
        slots = (frame.timestamp.str.slice(11, 13).astype(int) * 12 + frame.timestamp.str.slice(14, 16).astype(int) // 5).to_numpy()
        ix = frame.ramp_link_id.map(on).to_numpy(dtype=int)
        values = frame.flow_vph.to_numpy(dtype=np.float32)
        valid = (frame.is_score_eligible.eq(1) & frame.pct_observed.ge(75)).to_numpy() & np.isfinite(values)
        np.add.at(totals, (slots[valid], ix[valid]), values[valid])
        np.add.at(counts, (slots[valid], ix[valid]), 1)
        totals[counts < attachment_counts[None, :]] = np.nan
        ramp[d] = totals
    for a in arrays.values():
        a.flush()
    ramp.flush()
    write_json(
        directory / "manifest.json",
        dict(
            panel=panel,
            links=net["ids"],
            dates=DATES.strftime("%Y-%m-%d").tolist(),
            raw_days=TRAIN_DAYS,
            source=str(root.resolve()),
            quality="eligible & pct>=75 & finite",
            ramp="all attached ORs valid per slot; no attachment => zero",
        ),
    )
    log(f"data {panel}: {n} links | 273 raw / {len(DATES)} masked days | {time.monotonic() - start:.1f} s")


class Grid:
    def __init__(self, c, panel):
        self.panel = panel
        self.net = load_network(c["data"], panel)
        self.root = Path(c["grid_dir"]) / panel
        self.raw = {f: np.load(self.root / f"raw_{f}.npy", mmap_mode="r") for f in FIELDS}
        self.masked = {f: np.load(self.root / f"masked_{f}.npy", mmap_mode="r") for f in FIELDS}
        self.ramp = np.load(self.root / "ramp.npy", mmap_mode="r")
        self.dates = DATES
        self.n = len(self.net["ids"])

    def day(self, date):
        return int((pd.Timestamp(date).tz_localize(None).normalize() - DATES[0]).days)


def build(c):
    root = Path(c["grid_dir"])
    root.mkdir(parents=True, exist_ok=True)
    for panel in c["panels"]:
        with open(root / f"{panel}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepare_panel(c, panel)


def day_candidates(speed, threshold, min_coverage=0.7, max_iou=0.9):
    q, v = labels(speed, threshold)
    n = q.shape[1]
    cq = np.vstack([np.zeros((1, n), int), q.cumsum(axis=0)])
    cv = np.vstack([np.zeros((1, n), int), v.cumsum(axis=0)])
    slots = np.arange(12, 282)
    stable = ((cq[slots] - cq[slots - 12]) >= 2).any(axis=1)
    coverage = (cv[slots] - cv[slots - 12]).sum(axis=1) / (12 * n)
    fq = q[slots[:, None] + np.arange(1, 7)]
    fv = v[slots[:, None] + np.arange(1, 7)]
    any_future = fq.any(axis=(1, 2))
    base = (~stable) & any_future & (coverage >= min_coverage)
    previous = np.r_[False, base[:-1]]
    onset = base & ~previous & ~fq[:, :5].any(axis=(1, 2)) & fq[:, 5].any(axis=1)
    persist = q[slots - 1, None, :]
    intersection = (persist & fq & fv).sum(axis=(1, 2))
    union = ((persist | fq) & fv).sum(axis=(1, 2))
    iou = np.divide(intersection, union, out=np.ones_like(union, float), where=union > 0)
    ongoing = stable & any_future & (coverage >= min_coverage) & (iou <= max_iou)
    return dict(slots=slots, onset=onset, ongoing=ongoing, base_onset=base, coverage=coverage, persistence_iou=iou)


def official(c, panel):
    frames = []
    for split in ["train", "validation", "private"]:
        frame = pd.read_csv(Path(c["data"]) / "task2" / panel / split / "window_index.csv")
        dt = pd.to_datetime(frame.forecast_origin, utc=True)
        frame["slot"] = dt.dt.hour * 12 + dt.dt.minute // 5
        frame["day"] = (dt.dt.tz_localize(None).dt.normalize() - DATES[0]).dt.days
        frame["month"] = frame.date.str[:7]
        frame["condition"] = frame.condition.str.replace("queue_", "", regex=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def history_arrays(c, g, w):
    if w.split == "pseudo":
        return {f: np.array(g.raw[f][w.day, w.slot - 12 : w.slot]) for f in FIELDS}
    directory = Path(c["data"]) / "task2" / g.panel / w.split
    key = (g.panel, w.split)
    if not hasattr(g, "official_history"):
        g.official_history = {}
    if key not in g.official_history:
        frame = pd.read_parquet(directory / "window_history.parquet")
        g.official_history[key] = {wid: f for wid, f in frame.groupby("window_id", sort=False)}
    frame = g.official_history[key][w.window_id]
    timestamps = pd.to_datetime(frame.timestamp, utc=True)
    origin = pd.Timestamp(w.forecast_origin)
    expected = pd.date_range(origin - pd.Timedelta(minutes=60), periods=12, freq="5min")
    if sorted(timestamps.unique()) != list(expected):
        raise ValueError("Invalid official history interval")
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError("Duplicate official history")
    ti = ((timestamps - (origin - pd.Timedelta(minutes=60))).dt.total_seconds() / 300).to_numpy(dtype=int)
    li = frame.link_id.map(g.net["index"]).to_numpy(dtype=int)
    eligible = (frame.is_score_eligible.eq(1) & frame.pct_observed.ge(75)).to_numpy()
    result = {}
    for f in FIELDS:
        a = np.full((12, g.n), np.nan, dtype=np.float32)
        values = frame[f].to_numpy(dtype=np.float32).copy()
        values[~eligible | ~np.isfinite(values)] = np.nan
        a[ti, li] = values
        result[f] = a
    return result


def with_origin(history, masked, day, slot):
    return {f: np.vstack([history[f], np.asarray(masked[f][day, slot])[None]]) for f in history}


def greedy(frame, gap):
    selected = []
    for _, g in frame.sort_values(["panel", "absolute_slot"]).groupby("panel"):
        last = -(10**9)
        for i, t in zip(g.index, g.absolute_slot):
            if t - last >= gap:
                selected.append(i)
                last = t
    return frame.loc[selected].copy()


def window_columns(f):
    f = f.copy()
    dt = DATES[f.day.to_numpy()]
    f["date"] = dt.strftime("%Y-%m-%d")
    f["family_id"] = f.panel.str.rsplit("_", n=1).str[0]
    f["forecast_origin"] = [(d + pd.Timedelta(minutes=int(s) * 5)).tz_localize("UTC").isoformat() for d, s in zip(dt, f.slot)]
    f["window_id"] = [f"P_{p}_{d:%Y%m%d}_{s:03d}_ongoing" for p, d, s in zip(f.panel, dt, f.slot)]
    f["split"] = "pseudo"
    f["condition"] = "ongoing"
    return f


def make_candidates(c, guard=lambda: None):
    rows = []
    for panel in c["panels"]:
        g = Grid(c, panel)
        for day in range(TRAIN_DAYS):
            r = day_candidates(
                g.raw["speed_kmh"][day], g.net["threshold"], c["windows"]["min_coverage"], c["windows"]["max_persistence_iou"]
            )
            for j in np.flatnonzero(r["ongoing"]):
                slot = int(r["slots"][j])
                future = g.raw["speed_kmh"][day, slot + 1 : slot + 7]
                valid = np.isfinite(future)
                q = (future <= g.net["threshold"]) & valid
                rows.append(
                    dict(
                        panel=panel,
                        month=f"{DATES[day]:%Y-%m}",
                        day=day,
                        slot=slot,
                        absolute_slot=day * 288 + slot,
                        rows=int(valid.sum()),
                        positives=int(q.sum()),
                        history_coverage=float(r["coverage"][j]),
                        persistence_iou=float(r["persistence_iou"][j]),
                    )
                )
            guard()
        log(f"[candidates] {panel}: complete")
    return window_columns(pd.DataFrame(rows))


def ongoing_windows(c):
    root = Path(c["windows_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "windows.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "training.parquet").exists():
            return
        training = greedy(make_candidates(c), 12)
        training = training[training.month.ge("2030-07")].copy()
        if not c["trial"]:
            assert (len(training), int(training.rows.sum())) == (15497, 14246166)
        training.to_parquet(root / "training.parquet", index=False)
        log(f"[data] ongoing training windows: {len(training)}")
