"""工程 task1_3（GPU）：U-Net の入力を作り、学習し、Task 1（Task 3 も兼ねる）の答えを予測する。"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.config import SUBMISSION_SPLITS, work_path
from core.runtime import duration, limit_memory, log, make_runtime, section, to_numpy
from data.panel import weekday_and_slot
from data.release import STATE_COLUMNS, STATE_KEYS, Release, mainline_days, read_state_template
from task1_3 import model as RC
from task1_3.config import ExperimentConfig, UNetConfig, build
from task1_3.data import LINK, build_grid, place_on_days, profile, template_share, window_panels
from task1_3.store import GridStore, blackout_candidates, month_levels, read_grids, read_ramp_flows, read_ramp_validity
from task1_3.train import StageRun, load_checkpoint, predict_days, rule, train_stage, validation_set

STAGE = "full"


def state_file(c, run_id, split):
    return work_path(c, run_id, f"state_{split}.parquet")


def checkpoint(c):
    return Path(c["output"]) / "model.pt"


def experiment_config(c) -> ExperimentConfig:
    raw = copy.deepcopy(c["task1_3"])
    if c["trial"]:
        raw["unet"]["optim"].update(c["trial_settings"]["task1_3"]["optim"])
        raw["unet"]["data"].update(c["trial_settings"]["task1_3"]["data"])
    return build(ExperimentConfig, raw)


def holdout_days(train_dates: list[str], config: ExperimentConfig) -> list[str]:
    cv, validation = config.cv, config.unet.validation
    candidates = [d for d in train_dates if not cv.start <= d <= cv.end]
    order = np.random.default_rng(validation.holdout_seed).permutation(len(candidates))
    return sorted(candidates[i] for i in order[: round(validation.holdout_fraction * len(train_dates))])


def build_store(release, grids, config, runtime):
    unet = config.unet
    with_windows = set(window_panels(release))
    store, held, profiles, ramp_flows, ramp_valid, allowed, rho = None, None, [], [], [], {}, {}
    for position, grid in enumerate(grids):
        started = time.perf_counter()
        arrays = read_grids(release, grid.panel, grid.link_ids)
        if store is None:
            store = GridStore(grids, list(range(len(grids))), arrays["dates"], arrays["n_train_days"], runtime)
        elif arrays["dates"] != store.dates or arrays["n_train_days"] != store.n_train_days:
            raise ValueError(f"{grid.panel}: the panels do not share one calendar")
        store.place(position, arrays)
        train_dates = [day.date for day in mainline_days(release, grid.panel)]
        held = holdout_days(train_dates, config)
        fit_days = sorted(set(train_dates) - set(held))
        profiles.append(profile(release, grid.panel, fit_days, runtime.scoring()))
        ramp_flows.append(read_ramp_flows(release, grid.panel, grid.link_ids, store.dates))
        ramp_valid.append(read_ramp_validity(release, grid.panel, grid.link_ids))
        if grid.panel in with_windows:
            allowed[position] = blackout_candidates(release, grid.panel, grid.link_ids, arrays)
            rho[grid.panel] = template_share(release, grid.panel)
        log(
            f"features {grid.panel}: {grid.width} columns ({int((grid.kind == LINK).sum())} links), {len(fit_days)} train days, "
            f"{len(ramp_flows[-1]['is_on'])} ramps in {duration(time.perf_counter() - started)}"
        )
    blackouts = []
    for position, by_kind in allowed.items():
        train_dates = [day.date for day in mainline_days(release, grids[position].panel)]
        day_of = {day: i for i, day in enumerate(train_dates)}
        blackouts += place_on_days(
            grids[position].panel,
            by_kind,
            train_dates,
            [day_of[day] for day in held],
            unet.validation.blackout.count,
            unet.validation.blackout.seed,
        )
    log(f"blackouts: {len(blackouts)} on validation days")
    store.set_candidates(allowed)
    store.set_ramps(ramp_flows)
    store.set_ramp_validity(ramp_valid)
    store.set_history(profiles, fit_days)
    store.standardize_ramps(unet.data.ramps)
    store.level = month_levels(store, unet.data.level)
    return store, held, rho, blackouts


def train(c, run_id, store, grids, held, rho, blackouts, unet, runtime, family_of):
    partial = work_path(c, run_id, "model.partial.pt")
    run = StageRun(STAGE, torch.nonzero(store.tables.fit_day).flatten().tolist(), store.day_index(held), family_of, len(grids), partial)
    train_stage(store, run, unet, runtime, validation_set(store, run.holdout_days, unet, rho, blackouts))
    partial.replace(checkpoint(c))


def predict_rows(model, store, position, targets, config: UNetConfig, moved: RC.Rule) -> pd.DataFrame:
    grid = store.grids[position]
    dates = targets["timestamp"].str[:10]
    days = sorted(dates.unique())
    speed, flow, density, _ = predict_days(
        model,
        store,
        position,
        store.day_index(days),
        config.data.crop,
        config.inference.center,
        config.inference.day_context,
        gap_context=config.data.gap_context,
        tile_batch=config.inference.tile_batch,
    )
    speed, flow = RC.apply(moved, speed, flow, density, RC.free_speed(store, store.panel_columns(position)))
    speed, flow = to_numpy(speed), to_numpy(flow) * grid.lanes[None, None, :]
    column = targets["link_id"].map(grid.column_of())
    if column.isna().any():
        raise KeyError(f"{grid.panel}: target links without a column: {sorted(targets['link_id'][column.isna()].unique())[:5]}")
    day = dates.map({day: k for k, day in enumerate(days)}).to_numpy()
    _, slot = weekday_and_slot(targets["timestamp"])
    column = column.to_numpy(dtype=np.int64)
    out = targets[STATE_KEYS].assign(speed_kmh=speed[day, slot, column], flow_vph=flow[day, slot, column])
    if not (np.isfinite(out["speed_kmh"]).all() and np.isfinite(out["flow_vph"]).all()):
        raise RuntimeError(f"{grid.panel}: non-finite Task 1 predictions")
    return out[STATE_COLUMNS].sort_values(STATE_KEYS, ignore_index=True)


def predict(c, run_id, store, grids, unet, runtime, release):
    model, saved = load_checkpoint(checkpoint(c), unet, len(grids), runtime)
    parts = {split: [] for split in SUBMISSION_SPLITS}
    for position, grid in enumerate(store.grids):
        started = time.perf_counter()
        for split in SUBMISSION_SPLITS:
            parts[split].append(predict_rows(model, store, position, read_state_template(release, grid.panel, split), unet, rule(saved)))
        log(f"predict {grid.panel} in {duration(time.perf_counter() - started)}")
    for split, frames in parts.items():
        pd.concat(frames, ignore_index=True).to_parquet(state_file(c, run_id, split), index=False)


def run(c, run_id):
    config = experiment_config(c)
    unet, r = config.unet, config.runtime
    release = Release(Path(c["data"]))
    runtime = make_runtime(r.device, r.dtype, r.seed)
    limit_memory(c, "task1_3")
    Path(c["output"]).mkdir(parents=True, exist_ok=True)
    grids = [build_grid(release, panel, unet.data.crop.space) for panel in release.panels()]
    section("task1_3 features")
    store, held, rho, blackouts = build_store(release, grids, config, runtime)
    if checkpoint(c).exists():
        log(f"task1_3: {checkpoint(c)} is there; not trained again")
    else:
        train(c, run_id, store, grids, held, rho, blackouts, unet, runtime, release.family_of())
    section("task1_3 predict")
    predict(c, run_id, store, grids, unet, runtime, release)
