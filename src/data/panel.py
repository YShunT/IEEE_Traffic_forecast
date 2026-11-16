"""1 パネルの区間・道路・配列（memmap）と、時刻・車線数・集計の補助。"""

from __future__ import annotations

import fcntl
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DATES, FIELDS, HISTORY_SLOTS, MAX_LANES, MIN_LANES, SLOTS_PER_DAY, TIMESTAMP_FORMAT, TRAIN_DAYS
from core.runtime import log
from data.release import Road, attach

SOURCES = {"train": ("mainline_states", TRAIN_DAYS), "public": ("mainline_states_masked", len(DATES))}


def split_of(day):
    return "train" if day < TRAIN_DAYS else ("validation" if DATES[day].month == 3 else "private")


def slot_of(timestamps):
    return (timestamps.str.slice(11, 13).astype(int) * 12 + timestamps.str.slice(14, 16).astype(int) // 5).to_numpy()


def day_file(root, split, kind, date, stem):
    paths = list((root / split / kind).glob(f"*/{stem}_{date:%Y_%m_%d}.parquet"))
    if len(paths) != 1:
        raise ValueError(f"expected one {kind} file for {date:%Y-%m-%d} under {root / split}")
    return paths[0]


def build(c, name, ids, ramps):
    cache = Path(c["cache"]) / "grid" / name
    if (cache / "manifest.json").exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache.parent / f"{name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / "manifest.json").exists():
            _build(c, name, ids, ramps, cache)
    return cache


def _build(c, name, ids, ramps, cache):
    t0, n = time.monotonic(), len(ids)
    cache.mkdir(parents=True, exist_ok=True)
    index = {v: i for i, v in enumerate(ids)}
    root = Path(c["data"]) / "corridors" / name
    new = lambda key, days, dtype: np.lib.format.open_memmap(cache / f"{key}.npy", "w+", dtype, (days, 288, n))
    for source, (kind, days) in SOURCES.items():
        arrays = {f: new(f"{source}_{f}", days, np.float32) for f in FIELDS}
        pct, eligible = new(f"{source}_pct", days, np.float32), new(f"{source}_eligible", days, bool)
        for a in arrays.values():
            a[:] = np.nan
        for day in range(days):
            frame = pd.read_parquet(
                day_file(root, "train" if source == "train" else split_of(day), kind, DATES[day], "synthetic_mainline"),
                columns=["timestamp", "link_id", *FIELDS, "is_score_eligible", "pct_observed"],
            )
            if frame.duplicated(["timestamp", "link_id"]).any():
                raise ValueError(f"duplicate mainline cell in {name} {DATES[day]:%Y-%m-%d}")
            slots, links = slot_of(frame.timestamp), frame.link_id.map(index).to_numpy(int)
            for f in FIELDS:
                arrays[f][day, slots, links] = frame[f].to_numpy(np.float32)
            pct[day, slots, links] = np.nan_to_num(frame.pct_observed.to_numpy(np.float32))
            eligible[day, slots, links] = (frame.is_score_eligible.eq(1) & frame.pct_observed.ge(75)).to_numpy()
        for a in (*arrays.values(), pct, eligible):
            a.flush()
    grids = {k: new(f"ramp_{k.lower()}", len(DATES), np.float32) for k in ("OR", "FR")}
    position = ramps.set_index("ramp_link_id")
    need = {k: np.bincount(ramps[ramps.ramp_type == k].position, minlength=n) for k in ("OR", "FR")}
    for day in range(len(DATES)):
        frame = pd.read_parquet(
            day_file(root, split_of(day), "ramp_states", DATES[day], "synthetic_ramp"),
            columns=["timestamp", "ramp_link_id", "flow_vph", "is_score_eligible", "pct_observed"],
        )
        frame = frame[frame.ramp_link_id.isin(position.index)]
        values = frame.flow_vph.to_numpy(np.float32)
        ok = (frame.is_score_eligible.eq(1) & frame.pct_observed.ge(75)).to_numpy() & np.isfinite(values)
        slots, links = slot_of(frame.timestamp), position.position.reindex(frame.ramp_link_id).to_numpy(int)
        kind = position.ramp_type.reindex(frame.ramp_link_id).to_numpy()
        for k in ("OR", "FR"):
            m = ok & (kind == k)
            total, count = np.zeros((288, n), np.float32), np.zeros((288, n), np.int16)
            np.add.at(total, (slots[m], links[m]), values[m])
            np.add.at(count, (slots[m], links[m]), 1)
            grids[k][day] = np.where((count >= need[k][None]) & (need[k][None] > 0), total, np.nan)
    for a in grids.values():
        a.flush()
    (cache / "manifest.json").write_text(json.dumps(dict(panel=name, links=list(ids), train_days=TRAIN_DAYS, days=len(DATES))))
    log(f"  [cache]   grid {name}: {n} links | {time.monotonic() - t0:.0f}s")


def load(cache):
    keys = [f"{s}_{f}" for s in SOURCES for f in (*FIELDS, "pct", "eligible")] + ["ramp_or", "ramp_fr"]
    return {k: np.load(Path(cache) / f"{k}.npy", mmap_mode="r") for k in keys}


MAX_HOPS = 10
FAR = 11


def invert(edges):
    before = defaultdict(set)
    for a, targets in edges.items():
        for b in targets:
            before[b].add(a)
    return before


def neighbours(graph, ids, bypass, anchor):
    position, found = {v: i for i, v in enumerate(ids)}, []
    for start in ids:
        seen, queue = {start: 0}, deque([start])
        while queue:
            a = queue.popleft()
            if seen[a] < MAX_HOPS:
                for b in sorted(graph.get(a, ())):
                    if b not in seen:
                        seen[b] = seen[a] + 1
                        queue.append(b)
        found.append([(position[b], h) for h, b in sorted(((h, b) for b, h in seen.items() if b in position and h > 0))])
    found = attach(found, anchor, bypass, MAX_HOPS)
    width = max(1, max(map(len, found)))
    index, hops = np.full((len(ids), width), -1), np.full((len(ids), width), FAR)
    for i, row in enumerate(found):
        for k, (b, h) in enumerate(row):
            index[i, k], hops[i, k] = b, h
    return index, hops


class Panel:
    def __init__(self, c, name):
        self.name = name
        net = Path(c["data"]) / "corridors" / name / "network"
        links = pd.read_csv(net / "links.csv").sort_values("link_id").reset_index(drop=True)
        self.ids = links.link_id.tolist()
        self.n, self.index = len(self.ids), {v: i for i, v in enumerate(self.ids)}
        self.free = links.free_speed_kmh.fillna(105.0).to_numpy(np.float64)
        self.capacity, self.lanes = links.capacity_vph.to_numpy(np.float64), links.lanes.to_numpy(np.float64)
        self.length = links.length_km.to_numpy(np.float64)
        self.threshold = 0.6 * self.free
        topo = pd.read_csv(net / "lwr_mainline_topology.csv").set_index("mainline_link_id").loc[self.ids]
        if topo.order_index.isna().any() or topo.order_index.duplicated().any():
            raise ValueError(f"{name}: every published link needs its own order_index")
        self.order_index = topo.order_index.to_numpy()
        road = Road(c["data"], name, self.ids)
        self.x_start, self.x_end = road.start, road.end
        fd = pd.read_csv(net / "fd_parameters.csv").set_index("link_id").reindex(self.ids)
        self.k_jam, self.k_crit = fd.k_jam.to_numpy(np.float64), fd.critical_density.to_numpy(np.float64)
        self.road, self.bypass, self.anchor, self.feeds = road.main_order, ~road.mainline, road.anchor, road.feeds()
        down = road.chain()
        self.edges = {(a, b): 1 for a, targets in down.items() for b in targets}
        self.up, self.down = (neighbours(side, self.ids, self.bypass, self.anchor) for side in (invert(down), down))
        ramps = pd.read_csv(net / "ramp_attachment_map.csv")
        ramps = ramps[ramps.nearest_mainline_link_id.isin(self.ids)].copy()
        ramps["position"] = ramps.nearest_mainline_link_id.map(self.index)
        self.n_on = np.bincount(ramps[ramps.ramp_type == "OR"].position, minlength=self.n)
        self.n_off = np.bincount(ramps[ramps.ramp_type == "FR"].position, minlength=self.n)
        self.has_on_ramp = topo.on_ramp_link_ids.notna().to_numpy(float)
        self.grid = load(build(c, name, self.ids, ramps[["ramp_link_id", "ramp_type", "position"]]))
        self._eligible = {}

    def eligible(self, source, field="speed_kmh"):
        key = source, field
        if key not in self._eligible:
            values = np.asarray(self.grid[f"{source}_{field}"])
            self._eligible[key] = np.where(np.asarray(self.grid[f"{source}_eligible"]) & np.isfinite(values), values, np.nan).astype(
                np.float32
            )
        return self._eligible[key]

    def at_origin(self, day, slot, eligible_only):
        if eligible_only:
            values = {f: self.eligible("public", f)[day, slot] for f in FIELDS}
        else:
            values = {f: np.asarray(self.grid[f"public_{f}"][day, slot]) for f in FIELDS}
        return values, np.asarray(self.grid["public_pct"][day, slot])

    def history(self, source, day, slot, eligible_only):
        span = slice(slot - HISTORY_SLOTS, slot)
        if eligible_only:
            values = {f: self.eligible(source, f)[day, span] for f in FIELDS}
        else:
            values = {f: np.asarray(self.grid[f"{source}_{f}"][day, span]) for f in FIELDS}
        at = self.at_origin(day, slot, eligible_only)[0]
        return {f: np.vstack([values[f], at[f][None]]) for f in FIELDS}

    def pct(self, source, day, slot):
        span = np.asarray(self.grid[f"{source}_pct"][day, slot - HISTORY_SLOTS : slot])
        return np.vstack([span, self.at_origin(day, slot, False)[1][None]])

    def ramp_totals(self, day, slot):
        span = slice(slot - HISTORY_SLOTS, slot + 1)
        return np.asarray(self.grid["ramp_or"][day, span]), np.asarray(self.grid["ramp_fr"][day, span])

    def on_ramp_flow(self, day, slot):
        end = slot + 1
        return np.where(self.n_on[None] > 0, np.asarray(self.grid["ramp_or"][day, end - 6 : end]), 0.0)


def weekday_and_slot(text: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(text, sort=False)
    parsed = pd.to_datetime(pd.Index(uniques), format=TIMESTAMP_FORMAT, utc=True)
    weekday = parsed.weekday.to_numpy(dtype=np.int64)[codes]
    slot = (parsed.hour * 12 + parsed.minute // 5).to_numpy(dtype=np.int64)[codes]
    return weekday, slot


def week_slot(text: pd.Series) -> np.ndarray:
    weekday, slot = weekday_and_slot(text)
    return weekday * SLOTS_PER_DAY + slot


def lanes_by_link(links: pd.DataFrame, fd: pd.DataFrame) -> dict[str, float]:
    lanes: dict[str, float] = {}
    for table in (links, fd):
        values = pd.to_numeric(table["lanes"], errors="coerce")
        valid = values.between(MIN_LANES, MAX_LANES)
        lanes.update(zip(table.loc[valid, "link_id"], values[valid].astype(float)))
    if not lanes:
        raise ValueError("no usable lane count in links.csv or fd_parameters.csv")
    return lanes


def family_mean(panel_scores: dict[str, float], family_of: dict[str, str]) -> float:
    families: dict[str, list[float]] = {}
    for panel, score in panel_scores.items():
        families.setdefault(family_of[panel], []).append(score)
    return float(np.mean([np.mean(scores) for _, scores in sorted(families.items())]))
