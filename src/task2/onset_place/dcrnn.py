"""工程 onset_place_dcrnn（GPU）：ongoing の窓で学習した土台を onset の窓で微調整し、公式の onset 窓の T+30 確率を出す。"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.config import work_path
from core.runtime import log, peak_gb, section, set_determinism, use_gpu
from task2.dcrnn import DCRNN, parameter_count
from task2.dcrnn import graph_tensors as panel_graph
from task2.onset_mix import save_probs
from task2.onset_place import data as OW
from task2.onset_place import data as grid
from task2.onset_place import dcrnn_data as D
from task2.onset_place.data import settings
from task2.onset_place.dcrnn_data import CONTEXT, DECODER, HAZARD, HISTORY, last_state
from task2.onset_place.gbm import Panel as OnsetPanel
from task2.onset_place.gbm import build_cache, meta
from task2.onset_place.gbm import windows as onset_windows

HIDDEN = 32


def schedule(step, spe, t):
    warm = t["warmup_epochs"] * spe
    total = t["epochs"] * spe
    if step < warm:
        return t["lr_start"] + (t["lr_peak"] - t["lr_start"]) * step / warm
    frac = (step - warm) / max(total - warm, 1)
    return t["lr_end"] + 0.5 * (t["lr_peak"] - t["lr_end"]) * (1 + math.cos(math.pi * frac))


def sampling_plan(counts, epochs, batch, seed):
    rng = np.random.default_rng(seed)
    total = sum(counts.values())
    names = sorted(counts)
    sizes = [batch] * (total // batch) + ([total % batch] if total % batch else [])
    plan = []
    for _ in range(epochs):
        epoch = []
        for s in sizes:
            name = names[rng.integers(len(names))]
            epoch.append((name, rng.integers(counts[name], size=s)))
        plan.append(epoch)
    return plan, len(sizes)


def make_optimizer(model, t):
    decay, plain = [], []
    for n, p in model.named_parameters():
        (plain if p.ndim == 1 else decay).append(p)
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=t["weight_decay"]), dict(params=plain, weight_decay=0.0)],
        lr=t["lr_start"],
        betas=tuple(t["betas"]),
        eps=t["eps"],
    )


def masked_bce(logits, y, e):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    e = e.to(loss.dtype)
    return (loss * e).sum((1, 2)) / e.sum((1, 2)).clamp_min(1.0)


def lovasz_hinge(logits, y, e):
    b = logits.shape[0]
    z, t, v = logits.reshape(b, -1), y.reshape(b, -1), e.reshape(b, -1)
    errors = torch.where(v, 1.0 - z * (2.0 * t - 1.0), z.new_full((), -1e4))
    errors, order = torch.sort(errors, dim=1, descending=True)
    gt = torch.gather(t * v, 1, order)
    ok = torch.gather(v.to(t.dtype), 1, order)
    total = gt.sum(1, keepdim=True)
    inter = total - gt.cumsum(1)
    union = total + (ok * (1.0 - gt)).cumsum(1)
    jaccard = 1.0 - inter / union.clamp_min(1.0)
    grad = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], 1)
    return (torch.relu(errors) * grad).sum(1)


def loss_terms(logits, y, e):
    return masked_bce(logits, y, e) + lovasz_hinge(logits, y, e)


def stay_cells(start, y):
    start = start[:, :, None]
    return ((start > 0) & (y > 0.5)) | ((start < 0) & (y < 0.5))


def stay_loss_terms(logits, y, e, start, weight=2.0, margin=2.0):
    stay = stay_cells(start, y)
    w = torch.where(stay, weight, 1.0) * e
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1.0)
    b = logits.shape[0]
    z, t, v = logits.reshape(b, -1), y.reshape(b, -1), e.reshape(b, -1)
    m = torch.where(stay, margin, 1.0).reshape(b, -1)
    errors = torch.where(v, m - z * (2.0 * t - 1.0), z.new_full((), -1e4))
    errors, order = torch.sort(errors, dim=1, descending=True)
    gt, ok = torch.gather(t * v, 1, order), torch.gather(v.to(t.dtype), 1, order)
    total = gt.sum(1, keepdim=True)
    jaccard = 1.0 - (total - gt.cumsum(1)) / (total + (ok * (1.0 - gt)).cumsum(1)).clamp_min(1.0)
    grad = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], 1)
    return bce + (torch.relu(errors) * grad).sum(1)


def graph_tensors(panels, diffusion, device):
    return {
        n: panel_graph(len(p.graph["nodes"]), p.graph["index"], p.graph["edges"], p.graph.get("feeds", {}), diffusion, device)
        for n, p in panels.items()
    }


def start_state(hist):
    queued, known = last_state(hist)
    return np.where(known, np.where(queued, 1.0, -1.0), 0.0).astype(np.float32)


def batch(d, idx, device, hist_key="hist"):
    hist = d[hist_key][idx]
    g = lambda a: torch.as_tensor(a, device=device)
    return (g(hist), g(d["ctx"][idx]), g(d["dec"][idx]), g(start_state(hist)), g(d["y"][idx].astype(np.float32)), g(d["e"][idx]))


def build_model(c):
    s = c["stage2"]
    return DCRNN(
        len(HISTORY),
        len(CONTEXT),
        len(DECODER) + 1,
        HIDDEN,
        s["layers"],
        s["diffusion"],
        s["context_hidden"],
        s["heads"],
        True,
        s["dropout"],
        c["features"]["z0"],
    )


def teacher_ratio(epoch, teacher_epochs):
    return max(0.0, 1.0 - (epoch - 1) / teacher_epochs)


def train_model(c, seed, data, graphs, tag, device="cuda"):
    s = c["stage2"]
    set_determinism(seed)
    counts = {n: len(d["windows"]) for n, d in data.items()}
    plan, spe = sampling_plan(counts, s["epochs"], s["batch"], seed)
    model = build_model(c).to(device)
    opt = make_optimizer(model, s)
    rng = torch.Generator(device=device).manual_seed(seed)
    stay = (float(s["stay_weight"]), float(s["stay_margin"]))
    t0, step = time.monotonic(), 0
    for epoch, batches in enumerate(plan, 1):
        model.train()
        te, total = time.monotonic(), 0.0
        tf = teacher_ratio(epoch, s["teacher_epochs"])
        for name, pick in batches:
            lr = schedule(step, spe, s)
            for g in opt.param_groups:
                g["lr"] = lr
            hist, ctx, dec, start, y, e = batch(data[name], pick, device)
            teach = e & (torch.rand((len(pick), 1, 6), device=device, generator=rng) < tf)
            logits = model(hist, ctx, dec, start, *graphs[name], teacher=y, teach=teach)
            loss = (loss_terms(logits, y, e) if stay == (1.0, 1.0) else stay_loss_terms(logits, y, e, start, *stay)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s["clip"])
            opt.step()
            total += float(loss.detach())
            step += 1
        log(
            f"ep {epoch:2d}/{s['epochs']} | {tag} | loss: tr {total / spe:.4f} | lr: {lr:.2e} | tf: {tf:.2f} ({time.monotonic() - te:.0f}s)"
        )
    log(
        f"  [done]    {tag} | {parameter_count(model):,} parameters | peak VRAM {torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved | "
        f"{time.monotonic() - t0:.0f}s"
    )
    return model.eval()


def fit_or_load(c, path, seed, data, graphs, tag):
    path = Path(path)
    if path.exists():
        log(f"{tag} | {path} is there")
        return
    model = train_model(c, seed, data, graphs, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(dict(state=model.state_dict()), temp)
    temp.replace(path)


BASE_DECODER = len(DECODER) + 1


def window_inputs(c, p, rows, scaler, hazard, kind, official, masked):
    f = c["features"]
    frame = rows.assign(set="official" if kind == "official" else "train")
    h, pct, orv, frv = D.read_histories(p, frame, official, masked)
    hist = D.history_channels(h["speed_kmh"], h["flow_vph"], h["occupancy"], pct, orv, frv, p, f)
    dec = np.stack([p.stats.decoder(int(r.day), int(r.slot), f) for r in rows.itertuples()])
    ctx = D.context_columns(p, hist, np.zeros((len(rows), p.n, 5), np.float32), rows.slot.to_numpy())
    ctx[..., D.ATTRIBUTES] = (ctx[..., D.ATTRIBUTES] - np.asarray(scaler["mean"], np.float32)) / np.asarray(scaler["std"], np.float32)
    ctx[..., D.TODAY_COLUMNS] = 0.0
    hist[..., HAZARD] = D.hazard_channel(hazard, hist, c["hazard"]["num_threads"])
    return hist.astype(np.float32), ctx.astype(np.float32), dec.astype(np.float32)


def make_model(c):
    s = c["stage2"]
    return DCRNN(
        len(HISTORY),
        len(CONTEXT),
        BASE_DECODER,
        32,
        s["layers"],
        s["diffusion"],
        s["context_hidden"],
        s["heads"],
        True,
        s["dropout"],
        c["features"]["z0"],
    )


def load_weights(model, path, device="cuda"):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state"])
    return model


def onset_loss(logits, y, e):
    return loss_terms(logits[..., 5:6], y[..., None], e[..., None])


def tune_rate(step, total, start=1e-4, end=1e-5):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / max(total, 1)))


def fit(c, data, graphs, seed, tag, start, device="cuda"):
    s = dict(c["stage2"])
    set_determinism(seed)
    model = make_model(c).to(device)
    load_weights(model, start, device)
    counts = {n: len(d["hist"]) for n, d in data.items()}
    plan, spe = sampling_plan(counts, s["epochs"], s["batch"], seed)
    opt = make_optimizer(model, s)
    torch.cuda.reset_peak_memory_stats()
    began, step, last = time.monotonic(), 0, 0.0
    for epoch, batches in enumerate(plan, 1):
        model.train()
        tick, total = time.monotonic(), 0.0
        for name, pick in batches:
            lr = tune_rate(step, spe * s["epochs"])
            for g in opt.param_groups:
                g["lr"] = lr
            d = data[name]
            hist = d["hist"][pick]
            dec = torch.as_tensor(d["dec"][pick], device=device)
            logits = model(
                torch.as_tensor(hist, device=device),
                torch.as_tensor(d["ctx"][pick], device=device),
                dec,
                torch.as_tensor(start_state(hist), device=device),
                *graphs[name],
            )
            loss = onset_loss(logits, torch.as_tensor(d["y"][pick], device=device), torch.as_tensor(d["e"][pick], device=device)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s["clip"])
            opt.step()
            total += float(loss.detach())
            step += 1
        last = total / spe
        if epoch in (1, s["epochs"]):
            log(
                f"ep {epoch:2d}/{s['epochs']} | {tag} | loss: tr {last:.4f} | lr: {lr:.2e}{' | final [saved]' if epoch == s['epochs'] else ''} ({time.monotonic() - tick:.0f}s)"
            )
    return model.eval(), dict(
        params=parameter_count(model),
        steps=step,
        steps_per_epoch=spe,
        loss=last,
        seconds=time.monotonic() - began,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
    )


@torch.no_grad()
def predict(model, d, graph, device="cuda", size=64):
    model.eval()
    out = []
    for a in range(0, len(d["hist"]), size):
        idx = np.arange(a, min(a + size, len(d["hist"])))
        hist = d["hist"][idx]
        logits = model(
            torch.as_tensor(hist, device=device),
            torch.as_tensor(d["ctx"][idx], device=device),
            torch.as_tensor(d["dec"][idx], device=device),
            torch.as_tensor(start_state(hist), device=device),
            *graph,
        )
        out.append(torch.sigmoid(logits[..., 5]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, d["hist"].shape[2]), np.float32)


def save(path, model, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), info=info), path)


def load(c, path, device="cuda"):
    return load_weights(make_model(c).to(device), path, device).eval()


NAME = "onset_place_dcrnn"


def base_models(c, o, run_id, scaler, graphs, hazard):
    paths = {seed: work_path(c, run_id, f"{NAME}_base_seed{seed}.pt") for seed in o["seeds"]}
    if all(p.exists() for p in paths.values()):
        return paths
    limit = o.get("windows_per_panel")
    for name in o["panels"]:
        D.build_panel(o, name, limit)
    data = {}
    for name in o["panels"]:
        d = D.load_panel(o, name, scaler)
        d["hist"][..., HAZARD] = D.hazard_channel(hazard, d["hist"], o["hazard"]["num_threads"])
        d["ctx"][..., D.TODAY_COLUMNS] = 0.0
        data[name] = d
    n_train = sum(len(d["windows"]) for d in data.values())
    if not o["trial"]:
        assert n_train == 15497, n_train
    s = o["stage2"]
    log(
        f"  [base]    train: {n_train:,} windows | {-(-n_train // s['batch'])} steps/ep x {s['epochs']} ep (batch {s['batch']}) x {len(o['seeds'])} seeds | "
        f"loss: BCE + Lovasz hinge, stay cells x{s['stay_weight']:g} / margin {s['stay_margin']:g}"
    )
    for seed, path in paths.items():
        fit_or_load(o, path, seed, data, graphs, f"base | seed {seed}")
    return paths


def onset_sets(o, panels, scaler, hazard, sets):
    cache = Path(o["cache_dir"]) / "tune"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, ws in sets.items():
        out[kind] = {}
        for name in o["panels"]:
            mine = [w for w in ws if w["panel"] == name]
            if not mine:
                continue
            p, frame = panels[name], pd.DataFrame([{k: w[k] for k in ("window_id", "day", "slot", "split")} for w in mine])
            path = cache / f"{kind}_{name}.npz"
            if path.exists():
                z = np.load(path)
                hist, ctx, dec = z["hist"], z["ctx"], z["dec"]
                assert z["window_id"].tolist() == frame.window_id.tolist()
            else:
                history = {}
                if kind == "official":
                    for split in sorted(frame.split.unique()):
                        history.update(D.official_history(o, p, split))
                hist, ctx, dec = window_inputs(o, p, frame, scaler, hazard, kind, history, D.masked_layer(o, p))
                np.savez(path, hist=hist, ctx=ctx, dec=dec, window_id=frame.window_id.to_numpy().astype(str))
            y, e = np.stack([w["y"] for w in mine]).astype(np.float32), np.stack([w["v"] for w in mine])
            out[kind][name] = dict(hist=hist, ctx=ctx, dec=dec, y=y, e=e)
    return out


def run(c, run_id):
    o = settings(c)
    section(NAME)
    torch = use_gpu(c, NAME)
    grid.build(o)
    OW.ongoing_windows(o)
    build_cache(o)
    panels = {n: D.Panel(o, n) for n in o["panels"]}
    scaler = D.static_scaler(panels)
    graphs = graph_tensors(panels, o["stage2"]["diffusion"], "cuda")
    hazard = D.fit_hazard(o, panels, work_path(c, run_id, f"{NAME}_hazard.txt"))
    base = base_models(c, o, run_id, scaler, graphs, hazard)
    torch.cuda.empty_cache()

    onset_panels = {n: OnsetPanel(o, n) for n in o["panels"]}
    ids = {n: list(p.net["ids"]) for n, p in onset_panels.items()}
    assert all(list(panels[n].ids) == ids[n] for n in ids)
    train, official = onset_windows(o, onset_panels)
    data = onset_sets(o, panels, scaler, hazard, {"train": train, "official": official})
    spe = -(-len(train) // o["stage2"]["batch"])
    log(
        f"  [tune]    {spe} steps/ep x {o['stage2']['epochs']} ep (batch {o['stage2']['batch']}) x {len(o['seeds'])} seeds | from the base models"
    )
    by_seed = []
    for seed in o["seeds"]:
        path = work_path(c, run_id, f"{NAME}_seed{seed}.pt")
        if path.exists():
            model = load(o, path)
            log(f"tune | seed {seed} | loaded {path}")
        else:
            model, info = fit(o, data["train"], graphs, seed, f"tune | seed {seed}", base[seed])
            save(path, model, info)
        by_seed.append(
            [q for name in o["panels"] if name in data["official"] for q in predict(model, data["official"][name], graphs[name])]
        )
        del model
    probs = [np.mean(p, 0) for p in zip(*by_seed)]
    save_probs(c, run_id, NAME, meta(official), probs, ids)
    log(f"  [predict] official: {len(official)} windows | RSS peak {peak_gb():.1f} GB")
