"""onset_place の DCRNN 型モデルの入力：道路グラフ、パネル、統計、窓・履歴・ハザードのキャッシュ。"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.runtime import log
from data.release import Road

HISTORY = [
    "speed_ratio",
    "flow_ratio",
    "occupancy",
    "mask_speed",
    "mask_flow",
    "mask_occ",
    "age_speed",
    "age_flow",
    "age_occ",
    "quality",
    "queue_flag",
    "margin",
    "hazard",
    "onramp_ratio",
    "onramp_mask",
    "offramp_ratio",
    "offramp_mask",
]

HAZARD = HISTORY.index("hazard")

TODAY = ["flow_dev_today", "speed_dev_today", "flow_dev_2h", "speed_dev_2h"]

CONTEXT = [
    "lanes",
    "length_km",
    "free_speed",
    "capacity_per_lane",
    "has_on_ramp",
    "has_off_ramp",
    "panel_queue_fraction",
    "panel_queue_change_30",
    "origin_sin",
    "origin_cos",
    *TODAY,
    "today_obs",
]

DECODER = [
    "freq_wd_long",
    "trans_queued_long",
    "trans_free_long",
    "end_gap_long",
    "freq_dev_month",
    "speed_dev_month",
    "n_month",
    "freq_wd_adapt",
    "miss_freq_wd_long",
    "miss_trans_queued_long",
    "miss_trans_free_long",
    "miss_end_gap_long",
    "miss_freq_dev_month",
    "miss_speed_dev_month",
    "target_sin",
    "target_cos",
    "step",
]

assert (len(HISTORY), len(CONTEXT), len(DECODER) + 1) == (17, 15, 18)


RELATIONS = ("up_direct", "up_bridge", "down_direct", "down_bridge")


def matrices(edges, index):
    n = len(index)
    mats = {r: np.zeros((n, n), np.float32) for r in RELATIONS}
    for (a, b), steps in edges.items():
        kind = "direct" if steps == 1 else "bridge"
        mats["down_" + kind][index[a], index[b]] = 1
        mats["up_" + kind][index[b], index[a]] = 1
    for m in mats.values():
        s = m.sum(1, keepdims=True)
        np.divide(m, s, out=m, where=s > 0)
    return mats


def build_graph(root, panel, nodes):
    along = Road(root, panel, list(nodes))
    index = {v: i for i, v in enumerate(nodes)}
    edges = {(a, b): 1 for a, targets in along.chain().items() for b in targets}
    return dict(nodes=list(nodes), index=index, edges=edges, feeds=along.feeds(), mats=matrices(edges, index))


DATES = pd.date_range("2030-06-01", "2031-04-30", freq="D")

TRAIN_DAYS = 273

FIELDS = ["speed_kmh", "flow_vph", "occupancy"]

QUALITY = "is_score_eligible==1 & pct_observed>=75 & finite"


def slot_of(ts):
    return ts.str.slice(11, 13).astype(int) * 12 + ts.str.slice(14, 16).astype(int) // 5


def eligible_mask(frame):
    return (frame.is_score_eligible.eq(1) & frame.pct_observed.ge(75)).to_numpy()


class BasePanel:
    def __init__(self, c, name, with_grids=True):
        self.name = name
        root = Path(c["data"]) / "corridors" / name / "network"
        gdir = Path(c["grid_dir"]) / name
        self.ids = json.loads((gdir / "manifest.json").read_text())["links"]
        self.n = len(self.ids)
        self.index = {v: i for i, v in enumerate(self.ids)}
        links = pd.read_csv(root / "links.csv").set_index("link_id").loc[self.ids]
        free = links.free_speed_kmh.fillna(105.0).to_numpy(np.float32)
        self.free, self.capacity = free, links.capacity_vph.to_numpy(np.float32)
        self.lanes, self.length = links.lanes.to_numpy(np.float32), links.length_km.to_numpy(np.float32)
        self.threshold = (0.6 * free).astype(np.float32)
        ramps = pd.read_csv(root / "ramp_attachment_map.csv")
        ramps = ramps[ramps.nearest_mainline_link_id.isin(self.ids)].copy()
        ramps["position"] = ramps.nearest_mainline_link_id.map(self.index)
        self.ramps = ramps
        self.n_or = np.bincount(ramps[ramps.ramp_type == "OR"].position, minlength=self.n).astype(np.float32)
        self.n_fr = np.bincount(ramps[ramps.ramp_type == "FR"].position, minlength=self.n).astype(np.float32)
        self.graph = build_graph(c["data"], name, self.ids)
        self.cache = Path(c["ramp_dir"]) / name
        if with_grids:
            self.raw = {f: np.load(gdir / f"raw_{f}.npy", mmap_mode="r") for f in FIELDS}
            self.ramp_or = np.load(self.cache / "ramp_or.npy", mmap_mode="r")
            self.ramp_fr = np.load(self.cache / "ramp_fr.npy", mmap_mode="r")
            speed = np.asarray(self.raw["speed_kmh"])
            self.valid = np.isfinite(speed)
            self.queue = (speed <= self.threshold) & self.valid
            self.weekday = np.array([d.weekday() for d in DATES])


def build_ramps(c, name):
    cache = Path(c["ramp_dir"]) / name
    if (cache / "manifest.json").exists():
        log(f"ramp {name}: existing cache")
        return
    t0 = time.monotonic()
    cache.mkdir(parents=True, exist_ok=True)
    p = BasePanel(c, name, with_grids=False)
    grids = {
        k: np.lib.format.open_memmap(cache / f"ramp_{k.lower()}.npy", mode="w+", dtype="float32", shape=(len(DATES), 288, p.n))
        for k in ("OR", "FR")
    }
    for g in grids.values():
        g[:] = np.nan
    need = {"OR": p.n_or, "FR": p.n_fr}
    pos = p.ramps.set_index("ramp_link_id")
    root = Path(c["data"]) / "corridors" / name
    for d, date in enumerate(DATES):
        split = "train" if d < TRAIN_DAYS else ("validation" if date.month == 3 else "private")
        paths = list((root / split / "ramp_states").glob(f"*/synthetic_ramp_{date:%Y_%m_%d}.parquet"))
        if len(paths) != 1:
            raise ValueError(f"Expected one ramp day {name} {date}")
        f = pd.read_parquet(paths[0], columns=["timestamp", "ramp_link_id", "flow_vph", "is_score_eligible", "pct_observed"])
        f = f[f.ramp_link_id.isin(pos.index)]
        values = f.flow_vph.to_numpy(np.float32)
        ok = eligible_mask(f) & np.isfinite(values)
        slots = slot_of(f.timestamp).to_numpy()
        ix = pos.position.reindex(f.ramp_link_id).to_numpy(int)
        kind = pos.ramp_type.reindex(f.ramp_link_id).to_numpy()
        for k in ("OR", "FR"):
            m = ok & (kind == k)
            total = np.zeros((288, p.n), np.float32)
            count = np.zeros((288, p.n), np.int16)
            np.add.at(total, (slots[m], ix[m]), values[m])
            np.add.at(count, (slots[m], ix[m]), 1)
            good = (count >= need[k][None, :]) & (need[k][None, :] > 0)
            grids[k][d] = np.where(good, total, np.nan)
    for g in grids.values():
        g.flush()
    (cache / "manifest.json").write_text(
        json.dumps(dict(panel=name, quality=QUALITY, rule="all attached ramps of the type valid per slot; no attachment => NaN"))
    )
    log(f"ramp {name}: {p.n} links | {time.monotonic() - t0:.1f} s")


def load_windows(c):
    w = Path(c["windows_dir"])
    cols = ["window_id", "panel", "month", "day", "slot", "date", "family_id", "split", "condition"]
    result = pd.read_parquet(w / "training.parquet")[cols]
    result = result[result.panel.isin(c["panels"])].assign(set="train").reset_index(drop=True)
    check = pd.to_datetime(result.date).sub(DATES[0]).dt.days
    if not (check == result.day).all():
        raise ValueError("Window day index does not match its date")
    return result


def targets(p, day, slot):
    v = np.asarray(p.raw["speed_kmh"][day, slot + 1 : slot + 7])
    ok = np.isfinite(v)
    return ((v <= p.threshold) & ok).T, ok.T


def ffill(values, valid):
    b, t, n = values.shape
    pos = np.where(valid, np.arange(t)[None, :, None], -1)
    last = np.maximum.accumulate(pos, axis=1)
    filled = np.take_along_axis(np.where(valid, values, 0.0), np.clip(last, 0, None), axis=1)
    filled = np.where(last >= 0, filled, 0.0)
    age = np.where(last >= 0, np.arange(t)[None, :, None] - last, t)
    return filled, age


WEEKDAY = np.array([d.weekday() for d in DATES])

MONTH_START = np.array([(d.replace(day=1) - DATES[0]).days for d in DATES])


def pool3(a):
    out = a.copy()
    out[:, 1:] |= a[:, :-1]
    out[:, :-1] |= a[:, 1:]
    return out


def mean3(a):
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0).astype(np.float32)
    total, count = v.copy(), ok.astype(np.float32)
    total[:, 1:] += v[:, :-1]
    total[:, :-1] += v[:, 1:]
    count[:, 1:] += ok[:, :-1]
    count[:, :-1] += ok[:, 1:]
    return np.divide(total, count, out=np.full(a.shape, np.nan, np.float32), where=count > 0)


def nanmean0(a):
    ok = np.isfinite(a)
    n = ok.sum(0)
    return np.divide(np.where(ok, a, 0.0).sum(0, dtype=np.float64), n, out=np.full(a.shape[1:], np.nan), where=n > 0).astype(np.float32)


class PanelStats:
    def __init__(self, raw_speed, raw_flow, masked_speed, masked_flow, free, capacity, queue_ratio=0.6):
        self.n = raw_speed.shape[2]
        self.speed = (np.asarray(raw_speed) / free).astype(np.float32)
        self.flow = (np.asarray(raw_flow) / capacity).astype(np.float32)
        self.valid = np.isfinite(self.speed)
        self.queue = self.valid & (self.speed <= queue_ratio)
        self.q3, self.v3, self.sp3 = pool3(self.queue), pool3(self.valid), mean3(self.speed)
        self.m_speed = (np.asarray(masked_speed) / free).astype(np.float32)
        self.m_flow = (np.asarray(masked_flow) / capacity).astype(np.float32)
        m_valid = np.isfinite(self.m_speed)
        self.m_valid = m_valid
        self.m_q3, self.m_v3 = pool3(m_valid & (self.m_speed <= queue_ratio)), pool3(m_valid)
        self.m_sp3 = mean3(self.m_speed)
        slots = np.arange(288)[None, :, None]
        self.last = np.where(self.queue.any(1), np.where(self.queue, slots, -1).max(1), np.nan)
        self._long = {}

    def long(self, cut):
        if cut not in self._long:
            days = np.arange(min(cut, TRAIN_DAYS))
            r = dict(speed=[], flow=[], freq3=[], sp3=[], last=[])
            for wd in range(7):
                d = days[WEEKDAY[days] == wd]
                r["speed"].append(nanmean0(self.speed[d]) if len(d) else np.full((288, self.n), np.nan, np.float32))
                r["flow"].append(nanmean0(self.flow[d]) if len(d) else np.full((288, self.n), np.nan, np.float32))
                r["sp3"].append(nanmean0(self.sp3[d]) if len(d) else np.full((288, self.n), np.nan, np.float32))
                den, num = self.v3[d].sum(0), self.q3[d].sum(0)
                r["freq3"].append(np.divide(num, den, out=np.full((288, self.n), np.nan, np.float32), where=den > 0))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    r["last"].append(np.nanmedian(self.last[d], 0) if len(d) else np.full(self.n, np.nan))
            self._long[cut] = {k: np.stack(v).astype(np.float32) for k, v in r.items()}
        return self._long[cut]

    def cut(self, day):
        return int(min(MONTH_START[day], TRAIN_DAYS))

    def same_day(self, day, slot, f):
        L = self.long(self.cut(day))
        wd = WEEKDAY[day]
        end = slot - f["today_gap_slots"] + 1
        out = np.full((self.n, 5), np.nan, np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k, start in ((0, f["today_start_slot"]), (2, max(0, slot - f["today_2h_slots"]))):
                if end > start:
                    out[:, k] = np.nanmean(self.m_flow[day, start:end] - L["flow"][wd, start:end], 0)
                    out[:, k + 1] = np.nanmean(self.m_speed[day, start:end] - L["speed"][wd, start:end], 0)
        start = max(0, slot - f["today_2h_slots"])
        seen = self.m_valid[day, start:end].sum(0) if end > start else 0
        out[:, 4] = seen / (f["today_2h_slots"] - f["today_gap_slots"] + 1)
        return out

    def decoder(self, day, slot, f):
        L = self.long(self.cut(day))
        wd, slots = WEEKDAY[day], slot + np.arange(7)
        freq = L["freq3"][wd][slots]
        days = np.arange(self.cut(day))
        d = days[WEEKDAY[days] == wd]
        trans = []
        if len(d):
            vs, qs = self.valid[d, slot - 1], self.queue[d, slot - 1]
            vt, qt = self.valid[d][:, slots], self.queue[d][:, slots]
            for state in (qs, ~qs):
                m = (vs & state)[:, None, :] & vt
                den = m.sum(0)
                trans.append(np.divide((qt & m).sum(0), den, out=np.full(den.shape, np.nan, np.float32), where=den > 0))
        else:
            trans = [np.full((7, self.n), np.nan, np.float32)] * 2
        end_gap = (slots[:, None] - L["last"][wd][None, :]) / 288.0
        month = np.arange(MONTH_START[day], day)
        if len(month):
            wm = WEEKDAY[month]
            obs, qd = self.m_v3[month][:, slots], self.m_q3[month][:, slots]
            base = L["freq3"][wm][:, slots]
            ok = obs & np.isfinite(base)
            n_dev = ok.sum(0)
            freq_dev = np.divide(np.where(ok, qd - base, 0.0).sum(0), n_dev, out=np.full(n_dev.shape, np.nan, np.float32), where=n_dev > 0)
            diff = self.m_sp3[month][:, slots] - L["sp3"][wm][:, slots]
            speed_dev = nanmean0(diff)
            n_obs = obs.sum(0)
        else:
            n_dev = n_obs = np.zeros((7, self.n))
            freq_dev = speed_dev = np.full((7, self.n), np.nan, np.float32)
        n_month = np.minimum(n_obs, f["month_cap_days"]) / f["month_cap_days"]
        adapt = np.clip(np.nan_to_num(freq) + n_dev / (n_dev + f["adapt_days"]) * np.nan_to_num(freq_dev), 0.0, 1.0)
        values = [freq, trans[0], trans[1], end_gap, freq_dev, speed_dev]
        angle = 2 * np.pi * slots / 288.0
        rep = lambda v: np.broadcast_to(np.asarray(v, np.float32)[:, None], (7, self.n))
        cols = (
            [np.nan_to_num(v) for v in values[:6]]
            + [n_month, adapt]
            + [np.isnan(v).astype(np.float32) for v in values]
            + [rep(np.sin(angle)), rep(np.cos(angle)), rep(np.arange(7) / 6.0)]
        )
        out = np.stack([cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], *cols[8:14], *cols[14:]], -1)
        assert out.shape == (7, self.n, len(DECODER))
        return out.astype(np.float32)


STEPS = 13


ATTRIBUTES = [CONTEXT.index(n) for n in ("lanes", "length_km", "free_speed", "capacity_per_lane")]

TODAY_COLUMNS = [CONTEXT.index(n) for n in (*TODAY, "today_obs")]

H = {name: i for i, name in enumerate(HISTORY)}


def build_grid_all(c, name):
    cache = Path(c["grid_all_dir"]) / name
    if (cache / "manifest.json").exists():
        return
    t0 = time.monotonic()
    cache.mkdir(parents=True, exist_ok=True)
    ids = json.loads((Path(c["grid_dir"]) / name / "manifest.json").read_text())["links"]
    index = {v: i for i, v in enumerate(ids)}
    arrays = {f: np.lib.format.open_memmap(cache / f"{f}.npy", "w+", np.float32, (TRAIN_DAYS, 288, len(ids))) for f in FIELDS}
    pct = np.lib.format.open_memmap(cache / "pct.npy", "w+", np.float32, (TRAIN_DAYS, 288, len(ids)))
    for a in arrays.values():
        a[:] = np.nan
    pct[:] = 0
    root = Path(c["data"]) / "corridors" / name / "train" / "mainline_states"
    for d in range(TRAIN_DAYS):
        date = DATES[d]
        frame = pd.read_parquet(
            root / f"year_month={date:%Y-%m}" / f"synthetic_mainline_{date:%Y_%m_%d}.parquet",
            columns=["timestamp", "link_id", *FIELDS, "pct_observed"],
        )
        slots, ix = slot_of(frame.timestamp).to_numpy(), frame.link_id.map(index).to_numpy(int)
        for f in FIELDS:
            arrays[f][d, slots, ix] = frame[f].to_numpy(np.float32)
        pct[d, slots, ix] = np.nan_to_num(frame.pct_observed.to_numpy(np.float32))
    for a in (*arrays.values(), pct):
        a.flush()
    (cache / "manifest.json").write_text(json.dumps(dict(panel=name, links=ids, content="every cell with a value; pct_observed")))
    log(f"  [cache]  grid_all {name}: {len(ids)} links | {time.monotonic() - t0:.0f}s")


def masked_layer(c, p):
    cache = Path(c["cache_dir"]) / "masked" / p.name
    if not (cache / "done").exists():
        cache.mkdir(parents=True, exist_ok=True)
        arrays = {f: np.lib.format.open_memmap(cache / f"{f}.npy", "w+", np.float32, (len(DATES), 288, p.n)) for f in FIELDS}
        pct = np.lib.format.open_memmap(cache / "pct.npy", "w+", np.float32, (len(DATES), 288, p.n))
        for a in arrays.values():
            a[:] = np.nan
        pct[:] = 0
        root = Path(c["data"]) / "corridors" / p.name
        for day, date in enumerate(DATES):
            split = "train" if day < TRAIN_DAYS else ("validation" if date.month == 3 else "private")
            paths = list((root / split / "mainline_states_masked").glob(f"*/synthetic_mainline_{date:%Y_%m_%d}.parquet"))
            if len(paths) != 1:
                raise ValueError(f"expected one masked day {p.name} {date}")
            frame = pd.read_parquet(paths[0], columns=["timestamp", "link_id", *FIELDS, "pct_observed"])
            slots, ix = slot_of(frame.timestamp).to_numpy(), frame.link_id.map(p.index).to_numpy(int)
            for f in FIELDS:
                arrays[f][day, slots, ix] = frame[f].to_numpy(np.float32)
            pct[day, slots, ix] = np.nan_to_num(frame.pct_observed.to_numpy(np.float32))
        for a in (*arrays.values(), pct):
            a.flush()
        (cache / "done").write_text("ok")
    return {f: np.load(cache / f"{f}.npy", mmap_mode="r") for f in FIELDS}, np.load(cache / "pct.npy", mmap_mode="r")


class Panel(BasePanel):
    def __init__(self, c, name):
        if not (Path(c["ramp_dir"]) / name / "manifest.json").exists():
            build_ramps(c, name)
        build_grid_all(c, name)
        super().__init__(c, name)
        gdir, adir = Path(c["grid_dir"]) / name, Path(c["grid_all_dir"]) / name
        self.masked = {f: np.load(gdir / f"masked_{f}.npy", mmap_mode="r") for f in FIELDS}
        self.all = {f: np.load(adir / f"{f}.npy", mmap_mode="r") for f in FIELDS}
        self.pct = np.load(adir / "pct.npy", mmap_mode="r")
        self._queue_ratio, self._stats = c["features"]["queue_ratio"], None
        topo = pd.read_csv(Path(c["data"]) / "corridors" / name / "network" / "lwr_mainline_topology.csv")
        self.order = topo.set_index("mainline_link_id").order_index.reindex(self.ids).to_numpy()
        self.static = np.stack(
            [
                self.lanes,
                self.length,
                self.free,
                self.capacity / self.lanes,
                (self.n_or > 0).astype(np.float32),
                (self.n_fr > 0).astype(np.float32),
            ],
            1,
        )


def _stats(self):
    if self._stats is None:
        self._stats = PanelStats(
            self.raw["speed_kmh"],
            self.raw["flow_vph"],
            self.masked["speed_kmh"],
            self.masked["flow_vph"],
            self.free,
            self.capacity,
            self._queue_ratio,
        )
    return self._stats


Panel.stats = property(_stats)


def official_history(c, p, split):
    root = Path(c["data"]) / "task2" / p.name / split
    frame = pd.read_parquet(root / "window_history.parquet")
    index = pd.read_csv(root / "window_index.csv").set_index("window_id")
    result = {}
    for wid, f in frame.groupby("window_id", sort=False):
        origin = pd.Timestamp(index.loc[wid, "forecast_origin"])
        ts = pd.to_datetime(f.timestamp, utc=True)
        expected = pd.date_range(origin - pd.Timedelta(minutes=60), periods=12, freq="5min")
        if sorted(ts.unique()) != list(expected) or f.duplicated(["timestamp", "link_id"]).any():
            raise ValueError(f"invalid official history {wid}")
        ti = ((ts - expected[0]).dt.total_seconds() / 300).to_numpy(int)
        li = f.link_id.map(p.index).to_numpy(int)
        arrays = {}
        for field in FIELDS:
            a = np.full((12, p.n), np.nan, np.float32)
            a[ti, li] = f[field].to_numpy(np.float32)
            arrays[field] = a
        pct = np.zeros((12, p.n), np.float32)
        pct[ti, li] = np.nan_to_num(f.pct_observed.to_numpy(np.float32))
        result[wid] = (arrays, pct)
    return result


def history_channels(speed, flow, occ, pct, orv, frv, p, f):
    ratio = [speed / p.free, flow / p.capacity, occ]
    valid = [np.isfinite(x) for x in ratio]
    out = np.zeros(speed.shape + (len(HISTORY),), np.float32)
    for k, (x, v) in enumerate(zip(ratio, valid)):
        filled, age = ffill(x, v)
        out[..., k], out[..., 3 + k], out[..., 6 + k] = filled, v, np.minimum(age, 12) / 12.0
    out[..., H["quality"]] = np.where(valid[0], pct / 100.0, 0.0)
    out[..., H["queue_flag"]] = valid[0] & (np.nan_to_num(ratio[0], nan=9.0) <= f["queue_ratio"])
    out[..., H["margin"]] = np.where(valid[0], ratio[0] - f["queue_ratio"], 0.0)
    for name, x in (("onramp", orv), ("offramp", frv)):
        m = np.isfinite(x)
        out[..., H[name + "_ratio"]] = ffill(x / p.capacity, m)[0]
        out[..., H[name + "_mask"]] = m
    return out


def last_state(hist, upto=None):
    part = hist[:, :upto]
    known = part[..., H["mask_speed"]].any(1)
    queued = known & (part[:, -1, :, H["speed_ratio"]] <= 0.6)
    return queued, known


def context_columns(p, hist, today, slots):
    b = hist.shape[0]
    out = np.zeros((b, p.n, len(CONTEXT)), np.float32)
    out[:, :, :6] = p.static[None]
    fractions = []
    for upto in (STEPS, STEPS - 6):
        q, k = last_state(hist, upto)
        fractions.append(np.divide(q.sum(1), k.sum(1), out=np.zeros(b), where=k.sum(1) > 0))
    out[:, :, CONTEXT.index("panel_queue_fraction")] = fractions[0][:, None]
    out[:, :, CONTEXT.index("panel_queue_change_30")] = (fractions[0] - fractions[1])[:, None]
    angle = 2 * np.pi * np.asarray(slots) / 288.0
    out[:, :, CONTEXT.index("origin_sin")] = np.sin(angle)[:, None]
    out[:, :, CONTEXT.index("origin_cos")] = np.cos(angle)[:, None]
    out[:, :, TODAY_COLUMNS] = np.nan_to_num(today)
    return out


def read_histories(p, rows, official, masked):
    b = len(rows)
    h = {f: np.empty((b, STEPS, p.n), np.float32) for f in FIELDS}
    pct, orv, frv = (np.empty((b, STEPS, p.n), np.float32) for _ in range(3))
    for k, w in enumerate(rows.itertuples()):
        span = slice(w.slot - 12, w.slot)
        if w.set == "official":
            arrays, pct[k, :-1] = official[w.window_id]
            for f in FIELDS:
                h[f][k, :-1] = arrays[f]
        else:
            for f in FIELDS:
                h[f][k, :-1] = p.all[f][w.day, span]
            pct[k, :-1] = p.pct[w.day, span]
        for f in FIELDS:
            h[f][k, -1] = masked[0][f][w.day, w.slot]
        pct[k, -1] = masked[1][w.day, w.slot]
        orv[k], frv[k] = p.ramp_or[w.day, w.slot - 12 : w.slot + 1], p.ramp_fr[w.day, w.slot - 12 : w.slot + 1]
    return h, pct, orv, frv


def feature_dir(c, name):
    return Path(c["cache_dir"]) / "stage2" / name


def build_panel(c, name, limit=None):
    out = feature_dir(c, name)
    if (out / "manifest.json").exists():
        return
    t0 = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    p, f = Panel(c, name), c["features"]
    wins = load_windows(c)
    wins = wins[wins.panel == name].reset_index(drop=True)
    if limit:
        wins = wins.head(limit)
    masked = masked_layer(c, p)
    w = len(wins)
    hist = np.lib.format.open_memmap(out / "hist.npy", "w+", np.float32, (w, STEPS, p.n, len(HISTORY)))
    dec = np.lib.format.open_memmap(out / "dec.npy", "w+", np.float32, (w, 7, p.n, len(DECODER)))
    ctx = np.zeros((w, p.n, len(CONTEXT)), np.float32)
    today = np.zeros((w, p.n, 5), np.float32)
    y, e = np.zeros((w, p.n, 6), np.uint8), np.ones((w, p.n, 6), bool)
    for a in range(0, w, f["build_batch"]):
        rows = wins.iloc[a : a + f["build_batch"]]
        s = slice(a, a + len(rows))
        h, pct, orv, frv = read_histories(p, rows, {}, masked)
        hist[s] = history_channels(h["speed_kmh"], h["flow_vph"], h["occupancy"], pct, orv, frv, p, f)
        for k, r in enumerate(rows.itertuples()):
            today[a + k] = p.stats.same_day(r.day, r.slot, f)
            dec[a + k] = p.stats.decoder(r.day, r.slot, f)
            y[a + k], e[a + k] = targets(p, r.day, r.slot)
        ctx[s] = context_columns(p, hist[s], today[s], rows.slot.to_numpy())
    hist.flush()
    dec.flush()
    for key, value in (("ctx", ctx), ("today", today), ("y", y), ("e", e)):
        np.save(out / f"{key}.npy", value)
    wins.to_parquet(out / "windows.parquet")
    share = float(hist[..., H["mask_speed"]].mean())
    (out / "manifest.json").write_text(
        json.dumps(dict(panel=name, windows=w, nodes=p.n, history=HISTORY, context=CONTEXT, decoder=DECODER, value_share=share))
    )
    log(f"  [cache]  {name}: {w:,} windows x {p.n} links | cells with a value: {share:.3f} | {time.monotonic() - t0:.0f}s")


def hazard_rows(p, months, horizon, queue_ratio):
    month = np.array([d.strftime("%Y-%m") for d in DATES[:TRAIN_DAYS]])
    days = np.flatnonzero((month >= months[0]) & (month <= months[1]))
    s = np.asarray(p.raw["speed_kmh"])[days] / p.free
    f = np.asarray(p.raw["flow_vph"])[days] / p.capacity
    o = np.asarray(p.raw["occupancy"])[days]
    now, later = s[:, :-horizon], s[:, horizon:]
    keep = np.isfinite(now) & np.isfinite(f[:, :-horizon]) & np.isfinite(o[:, :-horizon]) & (now > queue_ratio) & np.isfinite(later)
    x = np.stack([f[:, :-horizon][keep], o[:, :-horizon][keep], now[keep]], 1).astype(np.float32)
    return x, (later[keep] <= queue_ratio).astype(np.int8)


def fit_hazard(c, panels, path):
    path = Path(path)
    if path.exists():
        return lgb.Booster(model_file=str(path))
    t0, h = time.monotonic(), c["hazard"]
    months = c["fit_months"]
    parts = [hazard_rows(p, months, h["horizon_slots"], c["features"]["queue_ratio"]) for p in panels.values()]
    x, y = np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
    params = dict(
        objective="binary",
        num_leaves=h["num_leaves"],
        learning_rate=h["learning_rate"],
        min_data_in_leaf=h["min_data_in_leaf"],
        num_threads=h["num_threads"],
        seed=h["seed"],
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model = lgb.train(params, lgb.Dataset(x, label=y, feature_name=["flow_ratio", "occupancy", "speed_ratio"]), h["rounds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    log(
        f"  [hazard] {len(y):,} cells ({months[0]}..{months[1]}) | queued 30 min later: {y.mean():.3%} | "
        f"{h['rounds']} rounds | {time.monotonic() - t0:.0f}s"
    )
    return model


def hazard_channel(model, hist, threads=4):
    ok = (
        (hist[..., H["mask_speed"]] > 0)
        & (hist[..., H["mask_flow"]] > 0)
        & (hist[..., H["mask_occ"]] > 0)
        & (hist[..., H["queue_flag"]] == 0)
    )
    out = np.zeros(hist.shape[:-1], np.float32)
    if ok.any():
        x = np.stack([hist[..., H["flow_ratio"]][ok], hist[..., H["occupancy"]][ok], hist[..., H["speed_ratio"]][ok]], 1)
        out[ok] = model.predict(x, num_threads=threads)
    return out


def static_scaler(panels):
    x = np.concatenate([p.static[:, :4] for p in panels.values()])
    return dict(mean=x.mean(0).tolist(), std=np.maximum(x.std(0), 1e-6).tolist())


def load_panel(c, name, scaler):
    d = feature_dir(c, name)
    r = {k: np.load(d / f"{k}.npy") for k in ("hist", "ctx", "today", "dec", "y", "e")}
    r["ctx"][..., ATTRIBUTES] = (r["ctx"][..., ATTRIBUTES] - np.asarray(scaler["mean"], np.float32)) / np.asarray(scaler["std"], np.float32)
    r["windows"] = pd.read_parquet(d / "windows.parquet")
    return r
