"""工程 ongoing_dcrnn（GPU）：ongoing の DCRNN 型モデルを seed ごとに学習し、公式の窓の確率を出す。"""

from __future__ import annotations

import json
import math
import shutil
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from core.config import MONTH, MONTH_START, STEPS, TRAIN_DAYS, WEEKDAY, work_path
from core.runtime import log, peak_gb, section, set_determinism, use_gpu
from task2.dcrnn import DCRNN, parameter_count
from task2.dcrnn import graph_tensors as panel_graph
from task2.ongoing import common as SH
from task2.ongoing import common as W
from task2.ongoing.common import cell_table, loss_terms, prepare, save_cells, training_windows

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


class Stats:
    def __init__(self, train_speed, train_flow, public_speed, public_flow, free, capacity, queue_ratio=0.6):
        self.n = train_speed.shape[2]
        self.speed = (np.asarray(train_speed) / free).astype(np.float32)
        self.flow = (np.asarray(train_flow) / capacity).astype(np.float32)
        self.valid = np.isfinite(self.speed)
        self.queue = self.valid & (self.speed <= queue_ratio)
        self.q3, self.v3, self.sp3 = pool3(self.queue), pool3(self.valid), mean3(self.speed)
        self.m_speed = (np.asarray(public_speed) / free).astype(np.float32)
        self.m_flow = (np.asarray(public_flow) / capacity).astype(np.float32)
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

    def decoder(self, day, slot, f, now):
        L = self.long(self.cut(day))
        wd, slots = WEEKDAY[day], slot + np.arange(7)
        freq = L["freq3"][wd][slots]
        days = np.arange(self.cut(day))
        d = days[WEEKDAY[days] == wd]
        trans = []
        if len(d):
            vs, qs = self.valid[d, now], self.queue[d, now]
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
    *SH.CHANNELS,
]
H = {name: i for i, name in enumerate(HISTORY)}

QUEUE_RATIO = 0.6
AGE_CAP = 12


def forward_fill(values, valid):
    steps = values.shape[1]
    at = np.maximum.accumulate(np.where(valid, np.arange(steps)[None, :, None], -1), 1)
    filled = np.take_along_axis(np.where(valid, values, 0.0), np.clip(at, 0, None), 1)
    return np.where(at >= 0, filled, 0.0), np.where(at >= 0, np.arange(steps)[None, :, None] - at, steps)


def history_channels(p, speed, flow, occ, pct, on, off, hazard):
    out = np.zeros(speed.shape + (len(HISTORY),), np.float32)
    ratios = [speed / p.free, flow / p.capacity, occ]
    valid = [np.isfinite(x) for x in ratios]
    for k, (x, v) in enumerate(zip(ratios, valid)):
        out[..., k], age = forward_fill(x, v)
        out[..., 3 + k], out[..., 6 + k] = v, np.minimum(age, AGE_CAP) / AGE_CAP
    out[..., H["quality"]] = np.where(valid[0], pct / 100.0, 0.0)
    out[..., H["queue_flag"]] = valid[0] & (np.nan_to_num(ratios[0], nan=9.0) <= QUEUE_RATIO)
    out[..., H["margin"]] = np.where(valid[0], ratios[0] - QUEUE_RATIO, 0.0)
    for name, x in (("onramp", on), ("offramp", off)):
        m = np.isfinite(x)
        out[..., H[name + "_ratio"]], out[..., H[name + "_mask"]] = forward_fill(x / p.capacity, m)[0], m
    ok = (out[..., 3] > 0) & (out[..., 4] > 0) & (out[..., 5] > 0) & (out[..., H["queue_flag"]] == 0)
    if ok.any():
        out[..., H["hazard"]][ok] = hazard.predict(np.stack([out[..., 1][ok], out[..., 2][ok], out[..., 0][ok]], 1), num_threads=4)
    return out


def last_state(hist, upto=None):
    upto = hist.shape[1] if upto is None else upto
    known = hist[:, :upto, :, H["mask_speed"]].any(1)
    return known & (hist[:, upto - 1, :, H["speed_ratio"]] <= QUEUE_RATIO), known


def start_state(hist):
    queued, known = last_state(hist)
    return np.where(known, np.where(queued, 1.0, -1.0), 0.0).astype(np.float32)


def context(p, hist, slots, scaler, fronts):
    b, steps = len(hist), hist.shape[1]
    out = np.zeros((b, p.n, len(CONTEXT)), np.float32)
    out[:, :, -len(SH.CHANNELS) :] = fronts
    out[:, :, :4] = ((attributes(p) - scaler["mean"]) / scaler["std"])[None]
    out[:, :, 4], out[:, :, 5] = (p.n_on > 0)[None], (p.n_off > 0)[None]
    share = []
    for upto in (steps, steps - 6):
        queued, known = last_state(hist, upto)
        share.append(np.divide(queued.sum(1), known.sum(1), out=np.zeros(b), where=known.sum(1) > 0))
    angle = 2 * np.pi * np.asarray(slots) / 288.0
    out[:, :, 6], out[:, :, 7] = share[0][:, None], (share[0] - share[1])[:, None]
    out[:, :, 8], out[:, :, 9] = np.sin(angle)[:, None], np.cos(angle)[:, None]
    return out


def attributes(p):
    return np.stack([p.lanes, p.length, p.free, p.capacity / p.lanes], 1)


def attribute_scaler(panels):
    x = np.concatenate([attributes(p) for p in panels.values()])
    return dict(mean=x.mean(0), std=np.maximum(x.std(0), 1e-6))


def make_stats(p):
    return Stats(
        p.eligible("train", "speed_kmh"),
        p.eligible("train", "flow_vph"),
        p.eligible("public", "speed_kmh"),
        p.eligible("public", "flow_vph"),
        p.free.astype(np.float32),
        p.capacity.astype(np.float32),
        QUEUE_RATIO,
    )


def inputs(c, p, stats, hist, hazard, scaler, store=None, batch=128):
    rows, w, steps = hist.rows, len(hist.rows), STEPS
    new = (
        (lambda name, shape: np.lib.format.open_memmap(Path(store) / f"{name}.npy", "w+", np.float32, shape))
        if store
        else (lambda name, shape: np.empty(shape, np.float32))
    )
    h, dec = new("hist", (w, steps, p.n, len(HISTORY))), new("dec", (w, 7, p.n, len(DECODER)))
    ctx = np.empty((w, p.n, len(CONTEXT)), np.float32)
    for a in range(0, w, batch):
        ks = range(a, min(a + batch, w))
        values = [hist.window(k, False) for k in ks]
        ramps = [hist.ramps(k) for k in ks]
        stack = lambda f: np.stack([v[0][f] for v in values])
        s = slice(a, a + len(ks))
        h[s] = history_channels(
            p,
            stack("speed_kmh"),
            stack("flow_vph"),
            stack("occupancy"),
            np.stack([v[1] for v in values]),
            np.stack([r[0] for r in ramps]),
            np.stack([r[1] for r in ramps]),
            hazard,
        )
        fronts = np.stack([SH.channels(SH.features(p, v[0]["speed_kmh"], v[0]["flow_vph"])) for v in values])
        ctx[s] = context(p, h[s], rows.slot.to_numpy()[s], scaler, fronts)
        for k in ks:
            day, slot = int(rows.day.iloc[k]), int(rows.slot.iloc[k])
            dec[k] = stats.decoder(day, slot, c["features"], slot)
    if store:
        h.flush()
        dec.flush()
    return h, ctx, dec


def fit_hazard(c, panels, path):
    path = Path(path)
    if path.exists():
        return lgb.Booster(model_file=str(path))
    t0, h, xs, ys = time.monotonic(), c["dcrnn"]["hazard"], [], []
    days = np.flatnonzero((MONTH[:TRAIN_DAYS] >= c["fit_months"][0]) & (MONTH[:TRAIN_DAYS] <= c["fit_months"][1]))
    for p in panels.values():
        s, f, o = (
            p.eligible("train", "speed_kmh")[days] / p.free,
            p.eligible("train", "flow_vph")[days] / p.capacity,
            p.eligible("train", "occupancy")[days],
        )
        now, later = s[:, :-6], s[:, 6:]
        keep = np.isfinite(now) & np.isfinite(f[:, :-6]) & np.isfinite(o[:, :-6]) & (now > QUEUE_RATIO) & np.isfinite(later)
        xs.append(np.stack([f[:, :-6][keep], o[:, :-6][keep], now[keep]], 1).astype(np.float32))
        ys.append((later[keep] <= QUEUE_RATIO).astype(np.int8))
    x, y = np.concatenate(xs), np.concatenate(ys)
    params = dict(
        objective="binary",
        num_leaves=h["num_leaves"],
        learning_rate=h["learning_rate"],
        min_data_in_leaf=h["min_data_in_leaf"],
        num_threads=4,
        seed=0,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model = lgb.train(params, lgb.Dataset(x, label=y, feature_name=["flow_ratio", "occupancy", "speed_ratio"]), h["rounds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    log(f"  [hazard]  {len(y):,} cells | queued 30 min later: {y.mean():.3%} | {h['rounds']} rounds | {time.monotonic() - t0:.0f}s")
    return model


def training_cache(c, name, p, stats, rows, hazard, scaler, key):
    directory = Path(c["cache"]) / "dcrnn" / key / name
    if not (directory / "manifest.json").exists():
        t0 = time.monotonic()
        directory.mkdir(parents=True, exist_ok=True)
        _, ctx, _ = inputs(c, p, stats, W.Histories(c, p, rows, "train"), hazard, scaler, store=directory)
        y, e = W.labels(p, "train", rows)
        np.save(directory / "ctx.npy", ctx)
        np.save(directory / "y.npy", y)
        np.save(directory / "e.npy", e)
        (directory / "manifest.json").write_text(
            json.dumps(dict(panel=name, windows=len(rows), first=rows.window_id.iloc[0], last=rows.window_id.iloc[-1]))
        )
        log(f"  [cache]   dcrnn inputs {key} {name}: {len(rows):,} windows x {p.n} links | {time.monotonic() - t0:.0f}s")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["windows"], manifest["first"], manifest["last"]) != (len(rows), rows.window_id.iloc[0], rows.window_id.iloc[-1]):
        raise ValueError(f"the cached DCRNN inputs of {name} ({key}) belong to another list of windows")
    return {k: np.load(directory / f"{k}.npy", mmap_mode="r" if k in ("hist", "dec") else None) for k in ("hist", "ctx", "dec", "y", "e")}


def build_model(c):
    d = c["dcrnn"]
    return DCRNN(
        len(HISTORY),
        len(CONTEXT),
        len(DECODER) + 1,
        d["hidden"],
        d["layers"],
        d["diffusion"],
        d["context_hidden"],
        d["heads"],
        d["attention"],
        d["dropout"],
        c["features"]["z0"],
    )


def graph_tensors(c, panels, device="cuda"):
    return {n: panel_graph(p.n, p.index, p.edges, p.feeds, c["dcrnn"]["diffusion"], device) for n, p in panels.items()}


def learning_rate(step, d):
    warm, total = d["warmup_epochs"] * d["steps_per_epoch"], d["epochs"] * d["steps_per_epoch"]
    if step < warm:
        return d["lr_start"] + (d["lr_peak"] - d["lr_start"]) * step / warm
    return d["lr_end"] + 0.5 * (d["lr_peak"] - d["lr_end"]) * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def train(c, seed, data, graphs, tag, device="cuda"):
    d = c["dcrnn"]
    set_determinism(seed)
    rng, names = np.random.default_rng(seed), sorted(data)
    model = build_model(c).to(device)
    decay = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=d["weight_decay"]), dict(params=plain, weight_decay=0.0)], lr=d["lr_start"])
    noise = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t0, step = time.monotonic(), 0
    g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    for epoch in range(1, d["epochs"] + 1):
        model.train()
        te, total = time.monotonic(), 0.0
        teacher = max(0.0, 1.0 - (epoch - 1) / d["teacher_epochs"])
        for _ in range(d["steps_per_epoch"]):
            lr = learning_rate(step, d)
            for group in opt.param_groups:
                group["lr"] = lr
            name = names[rng.integers(len(names))]
            x = data[name]
            pick = np.sort(rng.integers(len(x["y"]), size=d["batch"]))
            hist = np.asarray(x["hist"][pick])
            y, e, start = g(x["y"][pick].astype(np.float32)), g(x["e"][pick]), g(start_state(hist))
            teach = e & (torch.rand((len(pick), 1, 6), device=device, generator=noise) < teacher)
            logits = model(g(hist), g(x["ctx"][pick]), g(x["dec"][pick]), start, *graphs[name], teacher=y, teach=teach)
            loss = loss_terms(logits, y, e, start, d["stay_weight"], d["stay_margin"], "free").mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), d["clip"])
            opt.step()
            total += float(loss.detach())
            step += 1
        log(
            f"ep {epoch:2d}/{d['epochs']} | {tag} | loss: tr {total / d['steps_per_epoch']:.4f} | lr: {lr:.2e} | tf: {teacher:.2f} "
            f"({time.monotonic() - te:.0f}s)"
        )
    log(
        f"  [done]    {tag} | {parameter_count(model):,} parameters | peak VRAM {torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved | "
        f"{time.monotonic() - t0:.0f}s"
    )
    return model.eval()


def fit_or_load(c, path, seed, data, graphs, tag, device="cuda"):
    path = Path(path)
    if path.exists():
        ck = torch.load(path, map_location=device, weights_only=False)
        model = build_model(c).to(device)
        model.load_state_dict(ck["state"])
        log(f"{tag} | loaded {path}")
        return model.eval()
    model = train(c, seed, data, graphs, tag, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(dict(state=model.state_dict()), temp)
    temp.replace(path)
    return model


@torch.no_grad()
def predict(models, hist, ctx, dec, graph, device="cuda", size=64):
    out = []
    for a in range(0, len(hist), size):
        g = lambda x: torch.as_tensor(np.ascontiguousarray(x[a : a + size]), device=device)
        h = np.asarray(hist[a : a + size])
        start = torch.as_tensor(start_state(h), device=device)
        out.append(np.mean([torch.sigmoid(m(g(hist), g(ctx), g(dec), start, *graph)).cpu().numpy() for m in models], 0))
    return np.concatenate(out)


NAME = "dcrnn"


def run(c, run_id):
    d = c["dcrnn"]
    section(NAME)
    torch = use_gpu(c, "ongoing_dcrnn")
    panels, origins, official = prepare(c)
    stats = {n: make_stats(p) for n, p in panels.items()}
    scaler, graphs = attribute_scaler(panels), graph_tensors(c, panels)
    hazard = fit_hazard(c, panels, Path(c["cache"]) / "dcrnn" / "hazard.txt")
    official_inputs = {}
    for n, p in panels.items():
        r = official[official.panel == n].reset_index(drop=True)
        official_inputs[n] = (r, *inputs(c, p, stats[n], W.Histories(c, p, r, "official"), hazard, scaler))
    rows = training_windows(c, origins, d["gap_slots"])
    paths = {seed: work_path(c, run_id, f"dcrnn_seed{seed}.pt") for seed in d["seeds"]}
    key = f"gap{d['gap_slots']}{'_trial' if c['trial'] else ''}"
    data = (
        None
        if all(p.exists() for p in paths.values())
        else {
            n: training_cache(c, n, p, stats[n], rows[rows.panel == n].reset_index(drop=True), hazard, scaler, key)
            for n, p in panels.items()
        }
    )
    log(
        f"  [model]   {d['steps_per_epoch']} steps/ep x {d['epochs']} ep (batch {d['batch']}) x {len(d['seeds'])} seeds | "
        f"windows: {len(rows):,} ({d['gap_slots'] * 5} min apart)"
    )
    models = [fit_or_load(c, paths[seed], seed, data, graphs, f"dcrnn | seed {seed}") for seed in d["seeds"]]
    del data
    shutil.rmtree(Path(c["cache"]) / "dcrnn" / key, ignore_errors=True)
    torch.cuda.empty_cache()
    frames = []
    for n, p in panels.items():
        r, hist, ctx, dec = official_inputs[n]
        prob = np.mean([predict([m], hist, ctx, dec, graphs[n], size=d["eval_batch"]) for m in models], 0)
        frames.append(cell_table(p, r, prob))
    save_cells(c, run_id, NAME, frames)
    log(f"  [memory]  RSS peak: {peak_gb():.1f} GB")
