"""工程 onset_signs_dcrnn（GPU）：場所を読まない DCRNN 型モデルを学習・微調整し、校正した T+30 確率を出す。"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from core.config import work_path
from core.runtime import log, peak_gb, section, set_determinism, use_gpu
from data.panel import Panel
from task2.dcrnn import DCRNN, graph_tensors, parameter_count
from task2.ongoing import common as OW
from task2.ongoing.common import loss_terms
from task2.onset_signs import features as F
from task2.onset_signs import features as W
from task2.onset_signs.features import logit, predicted, save_calibrated, settings

NAME = "onset_signs_dcrnn"


def build_model(d):
    return DCRNN(
        len(F.HISTORY),
        len(F.CONTEXT),
        len(F.DECODER) + 1,
        d["hidden"],
        d["layers"],
        d["diffusion"],
        d["context_hidden"],
        d["heads"],
        d["attention"],
        d["dropout"],
        d["z0"],
    )


def inputs(c, p, normal, around, rows, kind):
    if not len(rows):
        return dict(
            hist=np.zeros((0, W.STEPS, p.n, len(F.HISTORY)), np.float32),
            ctx=np.zeros((0, p.n, len(F.CONTEXT)), np.float32),
            dec=np.zeros((0, 7, p.n, len(F.DECODER)), np.float32),
            start=np.zeros((0, p.n), np.float32),
        )
    hist, pct, on, off = W.window_histories(c, p, rows, kind)
    h, ctx, dec, start = F.channels(p, around, normal, rows.day.to_numpy(), rows.slot.to_numpy(), hist, pct, on, off)
    return dict(hist=h, ctx=ctx, dec=dec, start=start)


def base_data(c, o, p, normal, around, origins):
    rows = OW.training(c, origins[origins.panel == p.name], o["dcrnn"]["base"]["gap_slots"])
    if c["trial"]:
        rows = rows.head(c["trial_settings"]["onset_signs"]["windows_per_panel"])
    d = inputs(c, p, normal, around, rows.assign(split="train"), "pseudo")
    d["y"], d["e"] = OW.labels(p, "train", rows)
    return d


def tune_data(c, o, p, normal, around):
    local, t = o["local"], o["dcrnn"]["tune"]
    close, slots = W.near(p, local["clear_km"]), W.origins(c)
    found = []
    for day in W.example_days(c, o):
        hist, _, _, _, y, v = W.train_day(p, day, slots)
        keep = W.qualifying(p, hist["speed_kmh"], v, close, local["min_steps"])
        for k in np.flatnonzero(keep.any(1)):
            found.append((int(day), int(slots[k]), bool((y[k] & keep[k]).any())))
    found = pd.DataFrame(found, columns=["day", "slot", "queued"])
    rng = np.random.default_rng([t["seed"], int.from_bytes(p.name.encode(), "little") % 2**32])
    half = t["windows_per_panel"] // 2
    pick = []
    for queued in (True, False):
        pool = found.index[found.queued == queued].to_numpy()
        pick += list(rng.choice(pool, size=min(len(pool), half), replace=False))
    rows = found.loc[sorted(pick)].reset_index(drop=True)
    rows = W.frame(p.name, rows[["day", "slot"]], "pseudo")
    d = inputs(c, p, normal, around, rows, "pseudo")
    y, e = np.zeros((len(rows), p.n, 6), np.uint8), np.zeros((len(rows), p.n, 6), bool)
    for day, k in rows.groupby("day").groups.items():
        hist, _, _, _, yy, vv = W.train_day(p, day, rows.slot.to_numpy()[k])
        keep = W.qualifying(p, hist["speed_kmh"], vv, close, local["min_steps"])
        y[k, :, 5], e[k, :, 5] = yy, keep
    d["y"], d["e"] = y, e
    return d, dict(
        panel=p.name,
        windows=len(rows),
        with_queue=int(found.queued.sum()),
        found=len(found),
        examples=int(e.sum()),
        queued=int((y.astype(bool) & e).sum()),
    )


def rate(step, total, warm, t):
    if step < warm:
        return t["lr_start"] + (t["lr_peak"] - t["lr_start"]) * step / max(warm, 1)
    return t["lr_end"] + 0.5 * (t["lr_peak"] - t["lr_end"]) * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def fit(torch, d, t, seed, data, graphs, tag, last_only, start=None):
    set_determinism(seed)
    rng, names = np.random.default_rng(seed), sorted(data)
    model = build_model(d).to("cuda")
    if start is not None:
        model.load_state_dict(start)
    decay = [q for q in model.parameters() if q.ndim > 1]
    plain = [q for q in model.parameters() if q.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=t["weight_decay"]), dict(params=plain, weight_decay=0.0)], lr=t["lr_start"])
    noise = torch.Generator(device="cuda").manual_seed(seed)
    steps = t.get("steps_per_epoch") or -(-sum(len(x["y"]) for x in data.values()) // t["batch"])
    total, warm = t["epochs"] * steps, t.get("warmup_epochs", 0) * steps
    g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    began, step = time.monotonic(), 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        tick, running = time.monotonic(), 0.0
        teacher = max(0.0, 1.0 - (epoch - 1) / t["teacher_epochs"]) if not last_only else 0.0
        for _ in range(steps):
            lr = rate(step, total, warm, t)
            for group in opt.param_groups:
                group["lr"] = lr
            name = names[rng.integers(len(names))]
            x = data[name]
            pick = np.sort(rng.integers(len(x["y"]), size=t["batch"]))
            y, e, s0 = g(x["y"][pick].astype(np.float32)), g(x["e"][pick]), g(x["start"][pick])
            teach = e & (torch.rand((len(pick), 1, 6), device="cuda", generator=noise) < teacher)
            logits = model(g(x["hist"][pick]), g(x["ctx"][pick]), g(x["dec"][pick]), s0, *graphs[name], teacher=y, teach=teach)
            if last_only:
                logits, y, e = logits[..., 5:], y[..., 5:], e[..., 5:]
            loss = loss_terms(logits, y, e, s0, t["stay_weight"], t["stay_margin"], "free").mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            opt.step()
            running += float(loss.detach())
            step += 1
        if epoch in (1, t["epochs"]) or epoch % 10 == 0:
            log(f"ep {epoch:2d}/{t['epochs']} | {tag} | loss: tr {running / steps:.4f} | lr: {lr:.2e} ({time.monotonic() - tick:.0f}s)")
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(
        f"  [done]    {tag} | {step:,} steps | {parameter_count(model):,} parameters | peak VRAM {peak:.2f} GB | {time.monotonic() - began:.0f}s"
    )
    return model.eval()


def fit_or_load(torch, d, t, seed, data_fn, graphs, tag, path, last_only, start=None):
    if path.exists():
        log(f"{tag} | loaded {path}")
        return torch.load(path, map_location="cuda", weights_only=True)
    model = fit(torch, d, t, seed, data_fn(), graphs, tag, last_only, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temp)
    temp.replace(path)
    return model.state_dict()


def predict(torch, d, state, x, graph, size=64):
    model = build_model(d).to("cuda")
    model.load_state_dict(state)
    model.eval()
    out = []
    g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device="cuda")
    with torch.no_grad():
        for a in range(0, len(x["hist"]), size):
            s = slice(a, a + size)
            z = model(g(x["hist"][s]), g(x["ctx"][s]), g(x["dec"][s]), g(x["start"][s]), *graph)
            out.append(torch.sigmoid(z[..., 5]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, x["hist"].shape[2]), np.float32)


def run(c, run_id):
    o = settings(c)
    d = o["dcrnn"]
    section(NAME)
    torch = use_gpu(c, NAME)
    panels = {n: Panel(c, n) for n in c["panels"]}
    graphs = {n: graph_tensors(p.n, p.index, p.edges, p.feeds, d["diffusion"]) for n, p in panels.items()}
    normals = {n: F.Normal(p) for n, p in panels.items()}
    arounds = {n: F.Around(p) for n, p in panels.items()}
    log(
        f"  [model]   {len(F.HISTORY)} history x 13 steps | {len(F.CONTEXT)} context | {len(F.DECODER)} decoder | "
        f"hidden {d['hidden']} | seeds {d['seeds']}"
    )
    paths = {
        seed: (work_path(c, run_id, f"{NAME}_base_seed{seed}.pt"), work_path(c, run_id, f"{NAME}_seed{seed}.pt")) for seed in d["seeds"]
    }
    cache = {}

    def base_sets():
        if "base" not in cache:
            origins = OW.candidates(c, panels)
            cache["base"] = {n: base_data(c, o, p, normals[n], arounds[n], origins) for n, p in panels.items()}
            log(
                f"  [base]    ongoing training windows: {sum(len(x['y']) for x in cache['base'].values()):,} "
                f"({c['fit_months'][0]}..{c['fit_months'][1]}, {d['base']['gap_slots'] * 5} min apart)"
            )
        return cache["base"]

    def tune_sets():
        if "tune" not in cache:
            parts = {n: tune_data(c, o, p, normals[n], arounds[n]) for n, p in panels.items()}
            cache["tune"] = {n: x for n, (x, _) in parts.items()}
            info = pd.DataFrame([i for _, i in parts.values()])
            log(
                f"  [tune]    windows of local onset examples: {info.windows.sum():,} (of {info.found.sum():,}; "
                f"{info.with_queue.sum():,} with a queued example) | examples: {info.examples.sum():,} | queued: {info.queued.sum():,}"
            )
        return cache["tune"]

    states = {}
    for seed, (base_path, tune_path) in paths.items():
        start = None
        if not tune_path.exists():
            start = fit_or_load(torch, d, d["base"], seed, base_sets, graphs, f"base | seed {seed}", base_path, False)
        states[seed] = fit_or_load(torch, d, d["tune"], seed, tune_sets, graphs, f"tune | seed {seed}", tune_path, True, start)
    cache.clear()
    torch.cuda.empty_cache()

    logits = {}
    for n, p in panels.items():
        for key, rows in predicted(c, p).items():
            x = inputs(c, p, normals[n], arounds[n], rows, key)
            mean = np.mean([predict(torch, d, s, x, graphs[n]) for s in states.values()], 0)
            logits.setdefault(key, []).append((rows, logit(mean)))
    save_calibrated(c, run_id, NAME, panels, logits, o["threshold"])
    log(f"  [memory]  RSS peak {peak_gb():.1f} GB")
