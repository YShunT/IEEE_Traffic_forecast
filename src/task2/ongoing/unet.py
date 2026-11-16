"""工程 ongoing_unet（GPU）：道路に沿う 1 次元 U-Net を seed ごとに学習し、公式の窓の確率を出す。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.config import FIELDS, HISTORY_SLOTS, HORIZONS, STEPS, work_path
from core.runtime import log, peak_gb, section, set_determinism, use_gpu
from data.release import Road
from task2.ongoing import common as SH
from task2.ongoing import common as W
from task2.ongoing.common import cell_table, loss_terms, prepare, save_cells, training_windows

CELL_KM = 0.1
CELLS = 1024


@dataclass
class Space:
    panel: str
    order: np.ndarray
    start: np.ndarray
    length: np.ndarray
    cells: int
    weight: np.ndarray

    @property
    def inside(self):
        return (np.arange(CELLS) < self.cells).astype(np.float32)

    def roi(self, samples=4):
        out = np.zeros((CELLS, len(self.start)))
        for k in range(len(self.start)):
            a, b = self.start[k], self.start[k] + self.length[k]
            u = (a + (np.arange(samples) + 0.5) / samples * (b - a)) / CELL_KM - 0.5
            u = np.clip(u, 0.0, self.cells - 1.0)
            low = np.floor(u).astype(int)
            frac = u - low
            high = np.minimum(low + 1, self.cells - 1)
            np.add.at(out[:, k], low, (1.0 - frac) / samples)
            np.add.at(out[:, k], high, frac / samples)
        return out.astype(np.float32)

    def main_link(self):
        return np.where(self.weight.sum(1) > 0, self.weight.argmax(1), -1)


def load(data_root, panel, ids):
    road = Road(data_root, panel, list(ids))
    main = road.mainline
    start = road.start - np.floor(road.start[main].min() / CELL_KM) * CELL_KM
    cells = int(np.ceil(round((start + road.length)[main].max() / CELL_KM, 9)))
    if cells > CELLS:
        raise ValueError(f"{panel}: {cells} cells do not fit in {CELLS}")
    edges = np.arange(CELLS + 1) * CELL_KM
    overlap = np.minimum(edges[1:, None], (start + road.length)[None]) - np.maximum(edges[:-1, None], start[None])
    weight = np.where(overlap > 1e-9, overlap, 0.0) * main[None]
    return Space(panel, road.order, start, road.length, cells, weight.astype(np.float32))


DYNAMIC = ["speed_ratio", "flow", "occupancy", "queue_flag", "has_value"]
STATIC = ["capacity", "lanes", "free_speed", "link_length", "inside"]
TIME = ["origin_sin", "origin_cos"]
FLOW_SCALE, LANES_SCALE, FREE_SCALE, LENGTH_CAP = 10000.0, 6.0, 120.0, 2.0
LINK_STATIC = ["capacity", "lanes", "free_speed", "link_length"]
CHANNELS = [f"{name}_{5 * (HISTORY_SLOTS - k)}" for name in DYNAMIC for k in range(STEPS)] + STATIC + TIME + SH.CHANNELS
LINK_CHANNELS = (
    [f"link_{name}_{5 * (HISTORY_SLOTS - k)}" for name in DYNAMIC for k in range(STEPS)]
    + [f"link_{n}" for n in LINK_STATIC]
    + [f"link_{n}" for n in SH.CHANNELS]
)


class RoadCells:
    def __init__(self, c, p):
        self.p, self.name, self.n = p, p.name, p.n
        self.space = load(c["data"], p.name, p.ids)
        self.values = np.stack([np.asarray(p.grid[f"train_{f}"]) for f in FIELDS], -1)
        w = self.space.weight
        share = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        main = self.space.main_link()
        self.static = np.stack(
            [
                share @ (p.capacity / FLOW_SCALE),
                share @ (p.lanes / LANES_SCALE),
                share @ (p.free / FREE_SCALE),
                np.where(main >= 0, np.minimum(p.length[main], LENGTH_CAP) / LENGTH_CAP, 0.0),
                self.space.inside,
            ]
        ).astype(np.float32)

    def tensors(self, device):
        g = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        return dict(
            weight=g(self.space.weight.T),
            static=g(self.static),
            free=g(self.p.free),
            threshold=g(self.p.threshold),
            inside=g(self.space.inside),
            link_mass=g(self.space.weight.sum(1)),
            roi=g(self.space.roi()),
            link_static=g(
                np.stack(
                    [
                        self.p.capacity / FLOW_SCALE,
                        self.p.lanes / LANES_SCALE,
                        self.p.free / FREE_SCALE,
                        np.minimum(self.p.length, LENGTH_CAP) / LENGTH_CAP,
                    ]
                )
            ),
        )

    def fronts(self, hist):
        return np.stack([SH.channels(SH.features(self.p, h[..., 0], h[..., 1])).T for h in hist]).astype(np.float32)

    def train_fronts(self, days, slots, batch=2048):
        out = np.empty((len(days), len(SH.CHANNELS), self.n), np.float16)
        for a in range(0, len(days), batch):
            out[a : a + batch] = self.fronts(self.history(days[a : a + batch], slots[a : a + batch]))
        return out

    def at_origin(self, days, slots):
        return np.stack([np.asarray(self.p.grid[f"public_{f}"][days, slots]) for f in FIELDS], -1)

    def history(self, days, slots):
        days, slots = np.asarray(days), np.asarray(slots)
        h = self.values[days[:, None], slots[:, None] + np.arange(-HISTORY_SLOTS, 0)[None]]
        return np.concatenate([h, self.at_origin(days, slots)[:, None]], 1)

    def official_history(self, c, rows):
        hist = W.Histories(c, self.p, rows, "official")
        return np.stack([np.stack([hist.window(k, False)[0][f] for f in FIELDS], -1) for k in range(len(rows))]).astype(np.float32)

    def targets(self, days, slots):
        v = self.p.eligible("train")[np.asarray(days)[:, None], np.asarray(slots)[:, None] + np.arange(1, 7)[None]]
        ok = np.isfinite(v)
        return (ok & (v <= self.p.threshold)).transpose(0, 2, 1), ok.transpose(0, 2, 1)


def channels(t, hist, slots, fronts):
    steps = hist.shape[1]
    speed = hist[..., 0]
    valid = torch.isfinite(speed)
    queued = valid & (speed <= t["threshold"])
    v = valid.to(hist.dtype)
    values = torch.stack([speed / t["free"], hist[..., 1] / FLOW_SCALE, hist[..., 2], queued.to(hist.dtype)], 1)
    values = torch.nan_to_num(values) * v[:, None]
    mass = v @ t["weight"]
    has = (mass > 0).to(hist.dtype)
    cells = (values @ t["weight"]) / mass.clamp_min(1e-9)[:, None] * has[:, None]
    b = hist.shape[0]
    angle = 2 * torch.pi * slots.to(hist.dtype) / 288.0
    time_ = torch.stack([torch.sin(angle), torch.cos(angle)], 1)[:, :, None] * t["inside"]
    fronts = (fronts.to(hist.dtype) @ t["weight"]) / t["link_mass"].clamp_min(1e-9)
    x = torch.cat([cells.reshape(b, -1, cells.shape[-1]), has, t["static"].expand(b, -1, -1), time_, fronts], 1)
    last = steps - 1 - valid.flip(1).to(torch.int8).argmax(1)
    state = torch.gather(queued, 1, last[:, None]).squeeze(1)
    start = torch.where(valid.any(1), torch.where(state, 1.0, -1.0), 0.0).to(hist.dtype)
    return x, start


def link_inputs(t, hist, fronts):
    speed = hist[..., 0]
    valid = torch.isfinite(speed)
    queued = valid & (speed <= t["threshold"])
    v = valid.to(hist.dtype)
    values = torch.stack([speed / t["free"], hist[..., 1] / FLOW_SCALE, hist[..., 2], queued.to(hist.dtype)], 1)
    values = torch.nan_to_num(values) * v[:, None]
    b, n = hist.shape[0], hist.shape[2]
    return torch.cat([values.reshape(b, -1, n), v, t["link_static"].expand(b, -1, -1), fronts.to(hist.dtype)], 1)


WIDTHS = (32, 48, 64, 96, 96, 96, 96, 96)


class CellNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv1d(inputs, outputs, 3, padding=1),
        CellNorm(outputs),
        nn.ReLU(),
        nn.Conv1d(outputs, outputs, 3, padding=1),
        CellNorm(outputs),
        nn.ReLU(),
    )


class RoadUNet(nn.Module):
    def __init__(self, channels, link_channels, halvings, horizons=6, z0=1.0, widths=WIDTHS):
        super().__init__()
        w = list(widths[: halvings + 1])
        self.halvings = halvings
        self.down = nn.ModuleList(block(a, b) for a, b in zip([channels] + w[:-1], w))
        self.up = nn.ModuleList(block(w[k + 1] + w[k], w[k]) for k in reversed(range(halvings)))
        self.head = nn.Sequential(
            nn.Conv1d(w[0] + link_channels, w[0], 1), nn.ReLU(), nn.Conv1d(w[0], w[0], 1), nn.ReLU(), nn.Conv1d(w[0], horizons, 1)
        )
        self.start = nn.Parameter(torch.full((1,), float(z0)))

    def features(self, x):
        skips = []
        for k, layer in enumerate(self.down):
            x = layer(x)
            if k < self.halvings:
                skips.append(x)
                x = nn.functional.max_pool1d(x, 2)
        for layer in self.up:
            x = layer(torch.cat([x.repeat_interleave(2, dim=2), skips.pop()], 1))
        return x

    def forward(self, x, start, roi, links):
        change = self.head(torch.cat([self.features(x) @ roi, links], 1))
        return self.start * start[:, :, None] + change.transpose(1, 2)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def sampling_plan(counts, epochs, steps, batch, seed):
    rng, names = np.random.default_rng(seed), sorted(counts)
    plan = []
    for _ in range(epochs):
        epoch = []
        for _ in range(steps):
            name = names[rng.integers(len(names))]
            epoch.append((name, rng.integers(counts[name], size=batch)))
        plan.append(epoch)
    return plan


def schedule(step, spe, s):
    warm, total = s["warmup_epochs"] * spe, s["epochs"] * spe
    if step < warm:
        return s["lr_start"] + (s["lr_peak"] - s["lr_start"]) * step / warm
    return s["lr_end"] + 0.5 * (s["lr_peak"] - s["lr_end"]) * (1 + np.cos(np.pi * (step - warm) / max(total - warm, 1)))


def make_optimizer(model, s):
    decay, plain = [], []
    for _, p in model.named_parameters():
        (plain if p.ndim == 1 else decay).append(p)
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=s["weight_decay"]), dict(params=plain, weight_decay=0.0)],
        lr=s["lr_start"],
        betas=tuple(s["betas"]),
        eps=s["eps"],
    )


def build_model(u):
    return RoadUNet(len(CHANNELS), len(LINK_CHANNELS), u["halvings"], HORIZONS, u["z0"], tuple(u["widths"]))


def logits(model, t, hist, slots, fronts):
    x, start = channels(t, hist, slots, fronts)
    return model(x, start, t["roi"], link_inputs(t, hist, fronts)), start


def train(u, seed, roads, tensors, index, fronts, tag, device="cuda"):
    s = u["train"]
    set_determinism(seed)
    plan = sampling_plan({n: len(d) for n, (d, _) in index.items()}, s["epochs"], s["steps_per_epoch"], s["batch"], seed)
    model = build_model(u).to(device)
    opt, spe = make_optimizer(model, s), s["steps_per_epoch"]
    torch.cuda.reset_peak_memory_stats()
    t0, step = time.monotonic(), 0
    g = lambda a: torch.as_tensor(a, device=device)
    for epoch, batches in enumerate(plan, 1):
        model.train()
        te, total = time.monotonic(), 0.0
        for name, pick in batches:
            lr = schedule(step, spe, s)
            for group in opt.param_groups:
                group["lr"] = lr
            days, slots = index[name][0][pick], index[name][1][pick]
            road = roads[name]
            z, start = logits(model, tensors[name], g(road.history(days, slots)), g(np.asarray(slots)), g(fronts[name][pick]))
            y, e = road.targets(days, slots)
            loss = loss_terms(z, g(y.astype(np.float32)), g(e), start, s["stay_weight"], s["stay_margin"], "all").mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s["clip"])
            opt.step()
            total += float(loss.detach())
            step += 1
        log(f"ep {epoch:2d}/{s['epochs']} | {tag} | loss: tr {total / spe:.4f} | lr: {lr:.2e} ({time.monotonic() - te:.0f}s)")
    log(
        f"  [done]    {tag} | {parameter_count(model):,} parameters | peak VRAM {torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved | "
        f"{time.monotonic() - t0:.0f}s"
    )
    return model.eval()


def load_model(u, path, device="cuda"):
    model = build_model(u).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state"])
    return model.eval()


def fit_or_load(u, path, seed, roads, tensors, index, fronts, tag, device="cuda"):
    path = Path(path)
    if path.exists():
        log(f"{tag} | loaded {path}")
        return load_model(u, path, device)
    model = train(u, seed, roads, tensors, index, fronts(), tag, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(dict(state=model.state_dict()), temp)
    temp.replace(path)
    return model


@torch.no_grad()
def predict(models, t, hist, slots, fronts, device="cuda", size=64):
    out = {}
    for seed, model in models.items():
        parts = []
        for a in range(0, len(hist), size):
            z, _ = logits(
                model,
                t,
                torch.as_tensor(np.ascontiguousarray(hist[a : a + size]), device=device),
                torch.as_tensor(np.asarray(slots[a : a + size]), device=device),
                torch.as_tensor(fronts[a : a + size], device=device),
            )
            parts.append(torch.sigmoid(z).cpu().numpy())
        out[seed] = np.concatenate(parts)
    return out


NAME = "unet"


def run(c, run_id):
    u = c["unet"]
    section(NAME)
    torch = use_gpu(c, "ongoing_unet")
    panels, origins, official = prepare(c)
    roads = {n: RoadCells(c, p) for n, p in panels.items()}
    tensors = {n: r.tensors("cuda") for n, r in roads.items()}
    rows = training_windows(c, origins, 1)
    index = {n: (g.day.to_numpy(), g.slot.to_numpy()) for n, g in rows.groupby("panel")}
    log(
        f"  [model]   {u['train']['steps_per_epoch']} steps/ep x {u['train']['epochs']} ep (batch {u['train']['batch']}) x "
        f"{len(u['seeds'])} seeds | origins: {len(rows):,}"
    )
    cache = {}

    def train_fronts():
        if not cache:
            cache.update({n: roads[n].train_fronts(d, s) for n, (d, s) in index.items()})
        return cache

    models = {
        seed: fit_or_load(u, work_path(c, run_id, f"unet_seed{seed}.pt"), seed, roads, tensors, index, train_fronts, f"unet | seed {seed}")
        for seed in u["seeds"]
    }
    cache.clear()
    torch.cuda.empty_cache()
    frames = []
    for n, road in roads.items():
        r = official[official.panel == n].reset_index(drop=True)
        hist = road.official_history(c, r)
        by_seed = predict(models, tensors[n], hist, r.slot.to_numpy(), road.fronts(hist), size=u["train"]["eval_batch"])
        frames.append(cell_table(road.p, r, np.mean(list(by_seed.values()), 0)))
    save_cells(c, run_id, NAME, frames)
    log(f"  [memory]  RSS peak: {peak_gb():.1f} GB")
