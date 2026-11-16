"""工程 ongoing_gbm（CPU）：ongoing の GBM を学習し、公式の窓の確率を出す。"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from core.config import MONTH_START, STEPS, TRAIN_DAYS, WEEKDAY, work_path
from core.runtime import log, peak_gb, section
from data.panel import FAR
from task2.ongoing import common as SH
from task2.ongoing import common as W
from task2.ongoing.common import cell_table, prepare, save_cells, training_windows


def nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def nanmedian(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


class Habits:
    def __init__(self, p, month_start=False):
        speed = p.eligible("train")
        self.n = p.n
        self.valid = np.isfinite(speed)
        self.queued = self.valid & (speed <= p.threshold)
        any_queue = self.queued.any(1)
        self.first = np.where(any_queue, self.queued.argmax(1), np.nan)
        self.last = np.where(any_queue, 287 - self.queued[:, ::-1].argmax(1), np.nan)
        self._day, self._cache = None, {}
        self.month_start = month_start

    def _memo(self, day, key, compute):
        if day != self._day:
            self._day, self._cache = day, {}
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def days(self, day, weekday=False, last=None):
        days = np.arange(min(day, TRAIN_DAYS, MONTH_START[day] if self.month_start else day))
        if weekday:
            days = days[WEEKDAY[days] == WEEKDAY[day]]
        return days if last is None else days[-last:]

    def frequency(self, day, slot, weekday=False, last=None):
        def compute():
            slots = np.arange(max(0, slot - 1), min(288, slot + 2))
            pick = np.ix_(self.days(day, weekday, last), slots, np.arange(self.n))
            seen, hit = self.valid[pick].any(1).sum(0), self.queued[pick].any(1).sum(0)
            return np.divide(hit, seen, out=np.full(self.n, np.nan), where=seen >= 1)

        return self._memo(day, ("frequency", slot, weekday, last), compute)

    def timing(self, day):
        def compute():
            days = self.days(day, weekday=True)
            if not len(days):
                return (np.full(self.n, np.nan),) * 3
            return nanmedian(self.first[days]), nanmedian(self.last[days]), np.isfinite(self.first[days]).sum(0) / len(days)

        return self._memo(day, ("timing",), compute)

    def continues(self, day, now_slot, target_slot, queued):
        def compute():
            days = self.days(day, weekday=True)
            state = self.queued[days, now_slot]
            ok = self.valid[days, now_slot] & self.valid[days, target_slot] & (state == queued)
            n = ok.sum(0)
            return np.divide((self.queued[days, target_slot] & ok).sum(0), n, out=np.full(self.n, np.nan), where=n > 0)

        return self._memo(day, ("continues", now_slot, target_slot, queued), compute)


GROUPS = {
    "state": [
        "queue_filled",
        "age_minutes",
        "queue_fraction_15",
        "queue_fraction_30",
        "queue_fraction_60",
        "speed_last",
        "speed_mean",
        "speed_min",
        "speed_slope",
        "flow_last",
    ],
    "spatial": [
        "up_queue_1",
        "up_queue_3",
        "up_queue_distance",
        "up_speed",
        "down_queue_1",
        "down_queue_3",
        "down_queue_distance",
        "down_speed",
    ],
    "history": ["freq_wd_all", "freq_wd_8", "freq_any_14", "freq_origin", "transition_queued", "transition_free"],
    "timing": ["onset_slot_gap", "end_slot_gap", "queued_day_fraction"],
    "window": ["panel_queue_fraction", "panel_queue_change_30"],
    "time": ["horizon", "origin_slot", "weekday"],
    "ramps": ["ramp_ratio", "ramp_change"],
    "static": ["lanes", "length", "free_speed", "capacity_lane", "has_on_ramp", "has_off_ramp"],
    "short_trend": [f"{side}_{field}_slope_15" for side in ("self", "up", "down") for field in ("speed", "flow")],
    "trend_turn": [f"{side}_speed_slope_change" for side in ("self", "up", "down")],
    "neighbor_flow": ["up_flow_last_15", "down_flow_last_15"],
    "occupancy": ["occupancy_last_15", "occupancy_slope_15"],
    "history_quality": [f"{side}_valid_count_15" for side in ("self", "up", "down")],
    "history_connection": ["up_history_hops", "down_history_hops"],
    "speed_projection": [f"{side}_projected_speed_margin" for side in ("self", "up", "down")],
}
BASE_COLUMNS = [name for names in GROUPS.values() for name in names]
assert len(BASE_COLUMNS) == 61
GROUPS["shock"] = [f"shock_{name}" for name in SH.NAMES] + ["shock_reach_margin"]
COLUMNS = BASE_COLUMNS + GROUPS["shock"]


def last_valid(a):
    valid = np.isfinite(a)
    has = valid.any(0)
    at = a.shape[0] - 1 - valid[::-1].argmax(0)
    value = np.where(has, a[at, np.arange(a.shape[1])], np.nan)
    return value, np.where(has, (a.shape[0] - 1 - at) * 5.0, np.nan)


def slope(a):
    x = np.arange(len(a))[:, None] * 5.0
    valid = np.isfinite(a)
    count = valid.sum(0)
    mx = np.divide((valid * x).sum(0), count, out=np.zeros(a.shape[1]), where=count > 0)
    num, den = np.nansum((x - mx) * (a - nanmean(a)), 0), ((x - mx) ** 2 * valid).sum(0)
    return np.divide(num, den, out=np.full(a.shape[1], np.nan), where=(count >= 2) & (den > 0))


def queue_state(speed, threshold):
    last = last_valid(speed)[0]
    return np.where(np.isfinite(last), (last <= threshold).astype(float), np.nan)


def along(side, values):
    return np.where(side[0] >= 0, values[np.clip(side[0], 0, None)], np.nan)


def nearest(side, values):
    v = along(side, values)
    ok = np.isfinite(v)
    return np.where(ok.any(1), v[np.arange(len(v)), ok.argmax(1)], np.nan)


def first(side, values):
    return np.where(side[0][:, 0] >= 0, values[:, np.clip(side[0][:, 0], 0, None)], np.nan)


def queue_share(side, q, hops):
    return nanmean(np.where(side[1] <= hops, along(side, q), np.nan), 1)


def queue_hops(side, q):
    v = along(side, q)
    steps = np.where(v == 1, side[1], FAR).min(1)
    return np.where(np.isfinite(v).any(1), steps.astype(float), np.nan)


def build(p, habits, day, slot, history):
    speed = np.asarray(history["speed_kmh"])
    steps, now = len(speed), slot
    vr, fr, occ = speed / p.free, np.asarray(history["flow_vph"]) / p.capacity, np.asarray(history["occupancy"])
    q, last = queue_state(speed, p.threshold), last_valid(vr)[0]
    share = lambda k: nanmean(np.where(np.isfinite(speed[-k:]), speed[-k:] <= p.threshold, np.nan))
    ramp = p.on_ramp_flow(day, slot)
    panel_now, panel_before = nanmean(q), nanmean(queue_state(speed[: steps - 6], p.threshold))
    col = dict(
        queue_filled=q,
        age_minutes=last_valid(speed)[1],
        queue_fraction_15=share(3),
        queue_fraction_30=share(6),
        queue_fraction_60=share(12),
        speed_last=last,
        speed_mean=nanmean(vr),
        speed_min=np.nanmin(np.where(np.isfinite(vr), vr, np.inf), 0),
        speed_slope=slope(vr),
        flow_last=last_valid(fr)[0],
        panel_queue_fraction=np.full(p.n, panel_now),
        panel_queue_change_30=np.full(p.n, panel_now - panel_before),
        origin_slot=np.full(p.n, float(slot)),
        weekday=np.full(p.n, float(WEEKDAY[day])),
        ramp_ratio=nanmean(ramp[-3:]) / p.capacity,
        lanes=p.lanes,
        length=p.length,
        free_speed=p.free,
        capacity_lane=p.capacity / p.lanes,
        has_on_ramp=p.has_on_ramp,
        has_off_ramp=(p.n_off > 0).astype(float),
        occupancy_last_15=last_valid(occ[-3:])[0],
        occupancy_slope_15=slope(occ[-3:]),
    )
    col["speed_min"] = np.where(np.isfinite(vr).any(0), col["speed_min"], np.nan)
    col["ramp_change"] = col["ramp_ratio"] - nanmean(ramp[:3]) / p.capacity
    sides = {"self": (vr, fr), "up": (first(p.up, vr), first(p.up, fr)), "down": (first(p.down, vr), first(p.down, fr))}
    margin = {}
    for side, (v, f) in sides.items():
        if side != "self":
            nb = getattr(p, side)
            col[f"{side}_queue_1"], col[f"{side}_queue_3"] = queue_share(nb, q, 1), queue_share(nb, q, 3)
            col[f"{side}_queue_distance"], col[f"{side}_speed"] = queue_hops(nb, q), nearest(nb, last)
            col[f"{side}_flow_last_15"] = last_valid(f[-3:])[0]
            col[f"{side}_history_hops"] = np.where(nb[0][:, 0] >= 0, nb[1][:, 0].astype(float), np.nan)
        col[f"{side}_speed_slope_15"], col[f"{side}_flow_slope_15"] = slope(v[-3:]), slope(f[-3:])
        col[f"{side}_speed_slope_change"] = slope(v[-3:]) - slope(v[-6:-3])
        count = (np.isfinite(v[-3:]) & np.isfinite(f[-3:])).sum(0).astype(float)
        col[f"{side}_valid_count_15"] = count if side == "self" else np.where(getattr(p, side)[0][:, 0] >= 0, count, np.nan)
        recent, age = last_valid(v[-3:])
        margin[side] = recent[:, None] + slope(v[-3:])[:, None] * (age[:, None] + np.arange(1, 7)[None] * 5) - 0.60
    first_slot, last_slot, day_share = habits.timing(day)
    col["queued_day_fraction"] = day_share
    per_step = {
        name: np.empty((p.n, 6))
        for name in (
            "freq_wd_all",
            "freq_wd_8",
            "freq_any_14",
            "transition_queued",
            "transition_free",
            "onset_slot_gap",
            "end_slot_gap",
            "horizon",
        )
    }
    for step in range(1, 7):
        target, k = slot + step, step - 1
        per_step["freq_wd_all"][:, k] = habits.frequency(day, target, weekday=True)
        per_step["freq_wd_8"][:, k] = habits.frequency(day, target, weekday=True, last=8)
        per_step["freq_any_14"][:, k] = habits.frequency(day, target, last=14)
        per_step["transition_queued"][:, k] = habits.continues(day, now, target, True)
        per_step["transition_free"][:, k] = habits.continues(day, now, target, False)
        per_step["onset_slot_gap"][:, k], per_step["end_slot_gap"][:, k], per_step["horizon"][:, k] = (
            target - first_slot,
            target - last_slot,
            step,
        )
    col["freq_origin"] = habits.frequency(day, now, weekday=True)
    f = SH.features(p, speed, np.asarray(history["flow_vph"]))
    col.update({f"shock_{name}": f[:, i] for i, name in enumerate(SH.NAMES)})
    col["shock_reach_margin"] = SH.reach_margin(f)
    col.update(per_step)
    col.update({f"{side}_projected_speed_margin": m for side, m in margin.items()})
    out = np.empty((p.n, 6, len(COLUMNS)), np.float32)
    for k, name in enumerate(COLUMNS):
        v = col[name]
        out[:, :, k] = v if v.ndim == 2 else v[:, None]
    if np.isinf(out).any():
        raise ValueError("infinite GBM column")
    return out.reshape(p.n * 6, len(COLUMNS))


def focal(margin, target):
    s = 2 * np.asarray(target, np.float64) - 1
    z = s * np.asarray(margin, np.float64)
    q, r, log_q = expit(z), expit(-z), -np.logaddexp(0.0, -z)
    return -r * log_q, s * r * (q * log_q - r), q * r * ((1 - 2 * q) * log_q + 3 * r)


class Objective:
    def __init__(self, floor, every, rounds, tag):
        self.floor, self.every, self.rounds, self.tag = floor, every, rounds, tag
        self.trees, self.started = 0, time.monotonic()

    def __deepcopy__(self, _):
        return self

    def __call__(self, margin, data):
        loss, grad, hess = focal(margin, data.get_label())
        weight = data.get_weight()
        if self.trees and self.trees % self.every == 0:
            self.note(float(np.average(loss, weights=weight)), self.trees)
        self.trees += 1
        np.maximum(hess, self.floor, out=hess)
        return grad * weight, hess * weight

    def note(self, loss, trees, last=False):
        log(
            f"[{trees:4d}/{self.rounds}] {self.tag} | focal: tr {loss:.5f}{' | final [saved]' if last else ''} ({time.monotonic() - self.started:.0f}s)"
        )


def window_columns(p, habits, hist, k):
    r = hist.rows.iloc[k]
    return build(p, habits, int(r.day), int(r.slot), hist.window(k, True)[0])


def training_rows(c, panels, habits, rows):
    parts, total = {}, 0
    for name, p in panels.items():
        r = rows[rows.panel == name].reset_index(drop=True)
        y, e = W.labels(p, "train", r)
        parts[name] = (r, y, e)
        total += int(e.sum())
    X, target, weight = np.empty((total, len(COLUMNS)), np.float32), np.empty(total, np.int8), np.empty(total, np.float32)
    at = 0
    for name, p in panels.items():
        r, y, e = parts[name]
        hist = W.Histories(c, p, r, "train")
        for k in range(len(r)):
            keep = e[k].reshape(-1)
            n = int(keep.sum())
            X[at : at + n] = window_columns(p, habits[name], hist, k)[keep]
            target[at : at + n], weight[at : at + n] = y[k].reshape(-1)[keep], 1.0 / (len(r) * n)
            at += n
    assert at == total
    return X, target, weight * (total / weight.sum(dtype=np.float64))


def fit(c, X, y, weight, seed, tag):
    g = c["gbm"]
    objective = Objective(g["hessian_floor"], g["log_every"], g["rounds"], tag)
    params = dict(
        g["lightgbm"],
        objective=objective,
        learning_rate=g["learning_rate"],
        boost_from_average=False,
        metric="None",
        seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        data_random_seed=seed,
    )
    data = lgb.Dataset(X, label=y, weight=weight, feature_name=list(COLUMNS), params={k: v for k, v in params.items() if k != "objective"})
    model = lgb.train(params, data, num_boost_round=g["rounds"], keep_training_booster=True)
    final = model.eval_train(
        lambda margin, d: ("focal", float(np.average(focal(margin, d.get_label())[0], weights=d.get_weight())), False)
    )[0][2]
    objective.note(final, g["rounds"], last=True)
    return model


def fit_or_load(c, path, make_rows, seed, tag):
    path = Path(path)
    if path.exists():
        log(f"{tag} | loaded {path}")
        return lgb.Booster(model_file=str(path))
    model = fit(c, *make_rows(), seed, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    model.save_model(str(temp))
    temp.replace(path)
    return model


def predict(c, models, p, habits, hist):
    out = np.empty((len(hist.rows), p.n, 6), np.float32)
    for k in range(len(hist.rows)):
        x = window_columns(p, habits, hist, k)
        out[k] = np.mean([expit(m.predict(x, raw_score=True, num_threads=c["gbm"]["lightgbm"]["num_threads"])) for m in models], 0).reshape(
            p.n, 6
        )
    return out


NAME = "gbm"


def run(c, run_id):
    g = c["gbm"]
    section(NAME)
    panels, origins, official = prepare(c)
    habits = {n: Habits(p) for n, p in panels.items()}
    rows = training_windows(c, origins, g["gap_slots"])
    cache = {}

    def make_rows():
        if not cache:
            cache["rows"] = training_rows(c, panels, habits, rows)
            log(
                f"  [input]   {len(COLUMNS)} columns | history: {STEPS} steps, eligible cells | "
                f"windows: {len(rows):,} ({g['gap_slots'] * 5} min apart) | rows: {len(cache['rows'][1]):,} | RSS peak {peak_gb():.1f} GB"
            )
        return cache["rows"]

    models = [fit_or_load(c, work_path(c, run_id, f"gbm_seed{seed}.txt"), make_rows, seed, f"gbm | seed {seed}") for seed in g["seeds"]]
    cache.clear()
    frames = []
    for n, p in panels.items():
        r = official[official.panel == n].reset_index(drop=True)
        prob = predict(c, models, p, Habits(p, month_start=True), W.Histories(c, p, r, "official"))
        frames.append(cell_table(p, r, prob))
    save_cells(c, run_id, NAME, frames)
    log(f"  [memory]  RSS peak: {peak_gb():.1f} GB")
