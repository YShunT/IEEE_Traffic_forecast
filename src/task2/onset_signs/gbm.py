"""工程 onset_signs_gbm（CPU）：場所を読まない LightGBM を学習し、校正した T+30 確率を出す。"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import work_path
from core.runtime import log, peak_gb, section
from data.panel import Panel
from task2.onset_signs import features as F
from task2.onset_signs import features as W
from task2.onset_signs.features import predicted, save_calibrated, settings

NAME = "onset_signs_gbm"


def examples(c, o, p, normal, around):
    local, days, slots = o["local"], W.example_days(c, o), W.origins(c)
    close = W.near(p, local["clear_km"])
    counts = np.zeros(2, np.int64)
    for day in days:
        hist, _, _, _, y, v = W.train_day(p, day, slots)
        keep = W.qualifying(p, hist["speed_kmh"], v, close, local["min_steps"])
        counts += np.bincount(y[keep].astype(int), minlength=2)
    rate = min(1.0, local["negatives_per_positive"] * counts[1] / max(counts[0], 1))
    rng = np.random.default_rng([local["seed"], int.from_bytes(p.name.encode(), "little") % 2**32])
    xs, ys = [], []
    for day in days:
        hist, _, on, off, y, v = W.train_day(p, day, slots)
        keep = W.qualifying(p, hist["speed_kmh"], v, close, local["min_steps"])
        keep &= y | (rng.random(keep.shape) < rate)
        x = F.columns(p, around, normal, np.full(len(slots), day), slots, hist, on, off)
        xs.append(x[keep])
        ys.append(y[keep])
    x, y = np.concatenate(xs), np.concatenate(ys)
    return x, y, np.where(y, 1.0, 1.0 / rate), dict(panel=p.name, examples=int(counts.sum()), positives=int(counts[1]), negative_share=rate)


def answer_rows(c, p, normal, around, rows, kind):
    if not len(rows):
        return np.zeros((0, p.n, len(F.COLUMNS)), np.float32)
    hist, _, on, off = W.window_histories(c, p, rows, kind)
    return F.columns(p, around, normal, rows.day.to_numpy(), rows.slot.to_numpy(), hist, on, off)


def run(c, run_id):
    o = settings(c)
    g = o["gbm"]
    section(NAME)
    started = time.monotonic()
    path = work_path(c, run_id, f"{NAME}_seed{g['seed']}.txt")
    panels, inputs, counts, xs, ys, ws = {}, {}, [], [], [], []
    for name in c["panels"]:
        p = panels[name] = Panel(c, name)
        normal, around = F.Normal(p), F.Around(p)
        if not path.exists():
            x, y, w, n = examples(c, o, p, normal, around)
            xs.append(x)
            ys.append(y)
            ws.append(w)
            counts.append(n)
        inputs[name] = {key: (rows, answer_rows(c, p, normal, around, rows, key)) for key, rows in predicted(c, p).items()}
    if path.exists():
        model = lgb.Booster(model_file=str(path))
        log(f"  [model]   loaded {path}")
    else:
        table = pd.DataFrame(counts)
        log(
            f"  [data]    local onset examples ({c['fit_months'][0]}..{c['fit_months'][1]}, every {o['local']['origin_every'] * 5} min, "
            f"clear {o['local']['clear_km']:g} km): {table.examples.sum():,} | queued at T+30: {table.positives.sum():,} | "
            f"rows kept: {sum(len(y) for y in ys):,} | "
            + ", ".join(f"{r.panel} {r.positives:,}/{r.negative_share:.3f}" for r in table.itertuples())
        )
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        data = lgb.Dataset(x, y.astype(np.int8), weight=w, feature_name=F.COLUMNS, free_raw_data=False)
        log(
            f"  [recipe]  BCE | {g['rounds']} rounds fixed | lr: {g['learning_rate']} | leaves: {g['lightgbm']['num_leaves']} | "
            f"{len(F.COLUMNS)} columns | seed {g['seed']} | LightGBM {lgb.__version__}"
        )
        model = lgb.train({**g["lightgbm"], "learning_rate": g["learning_rate"], "seed": g["seed"]}, data, num_boost_round=g["rounds"])
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
        log(f"  [model]   {len(y):,} rows | {time.monotonic() - started:.0f}s")
    logits = {key: [] for key in next(iter(inputs.values()))}
    for name, sets in inputs.items():
        for key, (rows, x) in sets.items():
            z = model.predict(x.reshape(-1, x.shape[-1]), raw_score=True, num_threads=c["threads"]).reshape(x.shape[:2])
            logits[key].append((rows, z))
    save_calibrated(c, run_id, NAME, panels, logits, o["threshold"])
    gain = model.feature_importance("gain")
    log(
        "  [gain]    "
        + " | ".join(f"{n}: {s:.1%}" for n, s in sorted(zip(F.COLUMNS, gain / gain.sum()), key=lambda t: -t[1]))
        + f" | RSS peak {peak_gb():.1f} GB"
    )
