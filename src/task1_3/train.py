"""U-Net の学習：切り出しの位置、1 日全体の予測、最良のエポックの選択と保存。"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.config import BLACKOUT_SLOTS, FLOW_NORMALIZER, FLOW_WEIGHT, SLOTS_PER_DAY, SPEED_NORMALIZER, SPEED_WEIGHT, WINDOW_HISTORY_SLOTS
from core.runtime import Runtime, derive_seed, duration, log
from data.panel import family_mean
from task1_3 import model as RC
from task1_3.config import Extent, GapContextConfig, UNetConfig, UNetDataConfig
from task1_3.data import KINDS, LINK, Blackout, accumulation, lwr_score, lwr_sums
from task1_3.model import (
    INPUT_CHANNELS,
    RAMP_CHANNELS,
    ScoreLoss,
    UNet,
    assemble,
    cell_weights,
    hide,
    parameter_count,
    remask,
    residual_targets,
    to_density,
    to_state,
    with_blackout,
)
from task1_3.store import GridStore


def seeded(seed: int, *names, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(derive_seed(seed, *names))


@dataclass
class Crops:
    t: torch.Tensor
    c: torch.Tensor
    rate: torch.Tensor
    blackout: torch.Tensor


@dataclass
class BlackoutStarts:
    start: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    first: torch.Tensor
    count: torch.Tensor

    @classmethod
    def build(cls, store: GridStore, fit_days: list[int], config: UNetDataConfig) -> "BlackoutStarts":
        crop, context = config.crop.time, config.blackout.min_context
        fit = np.zeros(len(store.dates) + 2, dtype=bool)
        fit[np.asarray(fit_days) + 1] = True
        starts, lows, highs = [], [], []
        first = torch.zeros((len(store.grids), len(KINDS)), dtype=torch.long)
        count = torch.zeros_like(first)
        for position, by_kind in enumerate(store.candidates):
            for k, kind in enumerate(KINDS):
                start = np.zeros(0, dtype=np.int64) if by_kind is None else by_kind[kind]
                day, slot = start // SLOTS_PER_DAY, start % SLOTS_PER_DAY
                low = np.where(fit[day + 2], context, np.maximum(context, slot + crop - SLOTS_PER_DAY))
                high = np.where(fit[day], crop - BLACKOUT_SLOTS - context, np.minimum(crop - BLACKOUT_SLOTS - context, slot))
                usable = fit[day + 1] & (low <= high)
                first[position, k], count[position, k] = sum(len(s) for s in starts), int(usable.sum())
                starts.append(start[usable]), lows.append(low[usable]), highs.append(high[usable])
        return cls(*(torch.as_tensor(np.concatenate(parts)) for parts in (starts, lows, highs)), first, count)


class CropSampler:
    def __init__(self, store: GridStore, fit_days: list[int], config: UNetDataConfig, seed: int, stage: str):
        if min(grid.width for grid in store.grids) < config.crop.space:
            raise ValueError(f"a panel is narrower than the crop ({config.crop.space} columns)")
        self.store, self.config, self.seed, self.stage = store, config, seed, stage
        fit = set(fit_days)
        self.days = torch.tensor(fit_days)
        last_start = [SLOTS_PER_DAY - 1 if day + 1 in fit else SLOTS_PER_DAY - config.crop.time for day in fit_days]
        self.starts_per_day = torch.tensor(last_start) + 1
        self.offsets = torch.tensor(store.offsets[:-1])
        self.starts_per_panel = torch.tensor([grid.width - config.crop.space + 1 for grid in store.grids])
        self.rates = torch.tensor(config.mask_rates, dtype=store.dtype)
        cells = SLOTS_PER_DAY * len(fit_days) / (config.crop.time * config.crop.space)
        self.panel_patches = [round(config.coverage_per_epoch * cells * grid.width) for grid in store.grids]
        self.blackouts = BlackoutStarts.build(store, fit_days, config)

    @property
    def patches(self) -> int:
        return sum(self.panel_patches)

    @property
    def steps(self) -> int:
        return -(-self.patches // self.config.batch_size)

    def _panel_days(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        n_days = len(self.days)
        panels, days = [], []
        for position, patches in enumerate(self.panel_patches):
            per_day = torch.full((n_days,), patches // n_days)
            per_day[torch.randperm(n_days, generator=generator)[: patches % n_days]] += 1
            days.append(torch.repeat_interleave(torch.arange(n_days), per_day))
            panels.append(torch.full((patches,), position))
        order = torch.randperm(self.patches, generator=generator)
        return torch.cat(panels)[order], torch.cat(days)[order]

    def epoch(self, epoch: int) -> Iterator[Crops]:
        generator = seeded(self.seed, "crops", self.stage, epoch)
        device = self.store.device
        panel, day = self._panel_days(generator)

        def uniform(counts: torch.Tensor) -> torch.Tensor:
            return (torch.rand(self.patches, generator=generator, dtype=torch.float64) * counts).long()

        start_t = self.days[day] * SLOTS_PER_DAY + uniform(self.starts_per_day[day])
        start_c = self.offsets[panel] + uniform(self.starts_per_panel[panel])
        rate = self.rates[torch.randint(len(self.rates), (self.patches,), generator=generator)]
        starts = self.blackouts
        chosen = torch.rand(self.patches, generator=generator, dtype=torch.float64) < self.config.blackout.probability
        kind = torch.randint(len(KINDS), (self.patches,), generator=generator)
        kind = torch.where(starts.count[panel, kind] > 0, kind, 1 - kind)
        chosen &= starts.count[panel, kind] > 0
        which = starts.first[panel, kind] + uniform(starts.count[panel, kind])
        which = torch.where(chosen, which, 0).clamp(max=max(len(starts.start) - 1, 0))
        if len(starts.start):
            offset = starts.low[which] + uniform(starts.high[which] - starts.low[which] + 1)
            start_t = torch.where(chosen, starts.start[which] - offset, start_t)
            blackout = torch.where(chosen, offset, -1)
        else:
            blackout = torch.full((self.patches,), -1)
        time, space = torch.arange(self.config.crop.time), torch.arange(self.config.crop.space)
        for first in range(0, self.patches, self.config.batch_size):
            batch = slice(first, first + self.config.batch_size)
            yield Crops(
                (start_t[batch, None] + time).to(device),
                (start_c[batch, None] + space).to(device),
                rate[batch].to(device),
                blackout[batch].to(device),
            )


TILE_BATCH = 128


def tile_plan(length: int, tile: int, center: int, context: int = 0) -> list[tuple[int, int, int]]:
    if length + 2 * context < tile:
        raise ValueError(f"an axis of {length} cells with {context} of context is shorter than a tile ({tile})")
    margin = (tile - center) // 2
    last = length + context - tile
    starts = list(range(-context, last, center)) + [last]
    plan, lo = [], 0
    for position, start in enumerate(starts):
        hi = length if position == len(starts) - 1 else min(start + margin + center, length)
        if hi > lo:
            plan.append((start, lo, hi))
            lo = hi
    return plan


@torch.no_grad()
def predict_days(
    model: UNet,
    store: GridStore,
    position: int,
    days: Sequence[int],
    crop: Extent,
    center: Extent,
    context: int,
    hidden: torch.Tensor | None = None,
    seen: tuple[torch.Tensor, torch.Tensor] | None = None,
    gap_context: GapContextConfig | None = None,
    tile_batch: int = TILE_BATCH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if hidden is not None and seen is not None:
        raise ValueError("pass hidden or seen, not both")
    model.eval()
    device, width, offset = store.device, store.grids[position].width, store.offsets[position]
    speed = torch.zeros((len(days), SLOTS_PER_DAY, width), dtype=store.dtype, device=device)
    flow, density = torch.zeros_like(speed), torch.zeros_like(speed)
    tiles = [
        (k, day, time_tile, space_tile)
        for k, day in enumerate(days)
        for time_tile in tile_plan(SLOTS_PER_DAY, crop.time, center.time, context)
        for space_tile in tile_plan(width, crop.space, center.space)
    ]
    time, space = torch.arange(crop.time, device=device), torch.arange(crop.space, device=device)
    for first in range(0, len(tiles), tile_batch):
        batch = tiles[first : first + tile_batch]
        order = torch.tensor([k for k, *_ in batch], device=device)
        day_start = torch.tensor([day * SLOTS_PER_DAY for _, day, *_ in batch], device=device)
        t = day_start[:, None] + torch.tensor([tile[0] for *_, tile, _ in batch], device=device)[:, None] + time
        c = offset + torch.tensor([tile[0] for *_, tile in batch], device=device)[:, None] + space

        inside = (t >= 0) & (t < store.slots)
        rows = t.clamp(0, store.slots - 1)
        absent = ~inside[:, :, None]
        seen_speed = hide(store.gather(store.masked_speed, rows, c), absent)
        seen_flow = hide(store.gather(store.masked_flow, rows, c), absent)
        pct = store.gather(store.pct, rows, c) * inside[:, :, None]
        if hidden is not None or seen is not None:
            own = (t >= day_start[:, None]) & (t < day_start[:, None] + SLOTS_PER_DAY)
            local = (order[:, None] * SLOTS_PER_DAY + (t - day_start[:, None])).clamp(0, len(days) * SLOTS_PER_DAY - 1)
            if hidden is not None:
                masked = store.gather(hidden, local, c)
                own_speed = hide(store.gather(store.truth_speed, rows, c), masked)
                own_flow = hide(store.gather(store.truth_flow, rows, c), masked)
            else:
                own_speed, own_flow = store.gather(seen[0], local, c), store.gather(seen[1], local, c)
            seen_speed = torch.where(own[:, :, None], own_speed, seen_speed)
            seen_flow = torch.where(own[:, :, None], own_flow, seen_flow)

        inputs = assemble(store, t, c, seen_speed, seen_flow, pct, gap_context)
        output = model(inputs.x, inputs.panel)
        tile_speed, tile_flow = to_state(inputs, output)
        tile_k = to_density(inputs, output)
        for b, (k, _, (start_t, lo_t, hi_t), (start_c, lo_c, hi_c)) in enumerate(batch):
            speed[k, lo_t:hi_t, lo_c:hi_c] = tile_speed[b, lo_t - start_t : hi_t - start_t, lo_c - start_c : hi_c - start_c]
            flow[k, lo_t:hi_t, lo_c:hi_c] = tile_flow[b, lo_t - start_t : hi_t - start_t, lo_c - start_c : hi_c - start_c]
            density[k, lo_t:hi_t, lo_c:hi_c] = tile_k[b, lo_t - start_t : hi_t - start_t, lo_c - start_c : hi_c - start_c]
    return speed, flow, density, len(tiles)


HEADER_WIDTH = 70
POINTS_STATE, POINTS_LWR = 0.35, 0.10


@dataclass
class StageRun:
    stage: str
    fit_days: list[int]
    holdout_days: list[int]
    family_of: dict[str, str]
    n_panels: int
    checkpoint: Path


@dataclass
class ValidationSet:
    days: list[int]
    rows: torch.Tensor
    masks: list[torch.Tensor]
    blank: torch.Tensor
    history: torch.Tensor
    ramps: torch.Tensor
    rho: list[float]
    blackouts: int

    def unseen(self, mask: torch.Tensor) -> torch.Tensor:
        return (mask & ~self.history) | self.blank


@dataclass
class Validation:
    score: float
    loss: float
    rmse_speed: float
    rmse_flow: float
    blackout_rmse_speed: float | None
    blackout_rmse_flow: float | None
    s_lwr: float
    panels: dict
    rule: RC.Rule = RC.KEEP

    @property
    def points(self) -> float:
        return POINTS_STATE * self.score + POINTS_LWR * self.s_lwr

    def row(self) -> dict:
        return {
            "val_S_state": self.score,
            "val_loss": self.loss,
            "rmse_speed": self.rmse_speed,
            "rmse_flow_per_lane": self.rmse_flow,
            "blackout_rmse_speed": self.blackout_rmse_speed,
            "blackout_rmse_flow_per_lane": self.blackout_rmse_flow,
            "val_S_LWR": self.s_lwr,
            "val_points": self.points,
            "reconcile_alpha": self.rule.alpha,
            "reconcile_beta": self.rule.beta,
        }


def regime_score(n: int, sq_speed: float, sq_flow: float) -> float:
    if n == 0:
        return 0.0
    s_speed = max(0.0, 1.0 - math.sqrt(sq_speed / n) / SPEED_NORMALIZER)
    s_flow = max(0.0, 1.0 - math.sqrt(sq_flow / n) / FLOW_NORMALIZER)
    return SPEED_WEIGHT * s_speed + FLOW_WEIGHT * s_flow


def build_model(config: UNetConfig, n_panels: int, runtime: Runtime) -> UNet:
    torch.manual_seed(runtime.seed)
    return UNet(len(INPUT_CHANNELS), n_panels, config.model).to(device=runtime.device, dtype=runtime.dtype)


def learning_rate(step: int, steps_per_epoch: int, config: UNetConfig) -> float:
    optim = config.optim
    warmup, total = optim.warmup_epochs * steps_per_epoch, optim.epochs * steps_per_epoch
    if step < warmup:
        return optim.lr * (step + 1) / warmup
    return optim.lr * 0.5 * (1.0 + math.cos(math.pi * (step - warmup + 1) / (total - warmup)))


def make_optimizer(model: UNet, config: UNetConfig) -> torch.optim.Optimizer:
    decayed = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW(
        [{"params": decayed, "weight_decay": config.optim.weight_decay}, {"params": plain, "weight_decay": 0.0}], lr=config.optim.lr
    )


@torch.no_grad()
def update_ema(ema: UNet, model: UNet, decay: float) -> None:
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.mul_(decay).add_(current, alpha=1.0 - decay)


def training_loss(
    model: UNet, store: GridStore, crops, generator: torch.Generator, data: UNetDataConfig, score_loss: ScoreLoss
) -> torch.Tensor:
    speed, flow = store.gather(store.truth_speed, crops.t, crops.c), store.gather(store.truth_flow, crops.t, crops.c)
    eligible = store.gather(store.eligible, crops.t, crops.c)
    pct = store.gather(store.pct, crops.t, crops.c)
    hidden = remask(eligible, crops.rate, generator)
    scoreable = eligible & torch.isfinite(speed) & torch.isfinite(flow)
    hidden, unseen, blank = with_blackout(hidden, crops.blackout)
    in_blank = blank & scoreable
    inputs = assemble(store, crops.t, crops.c, hide(speed, unseen), hide(flow, unseen), pct, data.gap_context)
    output = model(inputs.x, inputs.panel)
    weights = cell_weights(hidden & scoreable, in_blank, store.links[crops.c[:, 0]], crops.rate, data.blackout.weight)
    target = residual_targets(inputs, speed, flow)
    return score_loss(store, crops, inputs, output, target, speed, flow, weights, scoreable, blank)


def validation_masks(store: GridStore, days: list[int], rates: list[float], seed: int) -> list[torch.Tensor]:
    rows = torch.cat([torch.arange(day * SLOTS_PER_DAY, (day + 1) * SLOTS_PER_DAY) for day in days])
    eligible = store.eligible[rows.to(store.device)]
    masks = []
    for position, rate in enumerate(rates):
        draws = torch.rand(eligible.shape, generator=seeded(seed, "validation mask", position))
        masks.append(eligible & (draws.to(store.device) < rate))
    return masks


def validation_set(
    store: GridStore, days: list[int], config: UNetConfig, rho: dict[str, float], blackouts: list[Blackout]
) -> ValidationSet:
    rows = torch.cat([torch.arange(day * SLOTS_PER_DAY, (day + 1) * SLOTS_PER_DAY) for day in days]).to(store.device)
    masks = validation_masks(store, days, config.validation.mask_rates, config.validation.mask_seed)
    blank = torch.zeros((len(rows), store.columns), dtype=torch.bool, device=store.device)
    history = torch.zeros_like(blank)
    position_of = {grid.panel: position for position, grid in enumerate(store.grids)}
    day_of = {store.dates[day]: k for k, day in enumerate(days)}
    placed = 0
    for blackout in blackouts:
        if blackout.panel not in position_of or blackout.date not in day_of:
            continue
        columns = store.panel_columns(position_of[blackout.panel])
        start = day_of[blackout.date] * SLOTS_PER_DAY + blackout.slot
        blank[start : start + BLACKOUT_SLOTS, columns] = True
        history[start - WINDOW_HISTORY_SLOTS : start, columns] = True
        placed += 1
    shares = [rho.get(grid.panel, 0.0) for grid in store.grids]
    return ValidationSet(days, rows, masks, blank, history, store.ramp_rows(rows), shares, placed)


def _sums(cells: torch.Tensor, squared_speed: torch.Tensor, squared_flow: torch.Tensor) -> tuple[int, float, float]:
    return (int(cells.sum()), float(torch.where(cells, squared_speed, 0.0).sum()), float(torch.where(cells, squared_flow, 0.0).sum()))


def _root_mean(total: float, n: int) -> float | None:
    return math.sqrt(total / n) if n else None


def _score_rate(
    store: GridStore,
    held: ValidationSet,
    position: int,
    columns: slice,
    cells: torch.Tensor,
    speed: torch.Tensor,
    flow: torch.Tensor,
    true_speed: torch.Tensor,
    true_flow: torch.Tensor,
) -> dict:
    width, shape = speed.shape[-1], speed.shape
    squared_speed, squared_flow = (speed.double() - true_speed) ** 2, (flow.double() - true_flow) ** 2
    blank = held.blank[:, columns].view(shape)
    outside = _sums(cells & ~blank & ~held.history[:, columns].view(shape), squared_speed, squared_flow)
    inside = _sums(cells & blank, squared_speed, squared_flow)
    rho = held.rho[position] if inside[0] else 0.0
    mean_speed, mean_flow = (
        (1.0 - rho) * out / max(outside[0], 1) + rho * within / max(inside[0], 1) for out, within in zip(outside[1:], inside[1:])
    )

    scored = cells & ~held.history[:, columns].view(shape)
    length, lanes = store.length[columns], store.lanes[columns]
    n_true = accumulation(length, true_speed, true_flow * lanes).view(-1, width)
    n_answer = accumulation(
        length, torch.where(scored, speed.double(), true_speed), torch.where(scored, flow.double(), true_flow) * lanes
    ).view(-1, width)
    eligible = store.eligible[held.rows, columns]
    steps = eligible[:-1] & eligible[1:] & held.ramps[:-1, columns] & (held.rows[1:] == held.rows[:-1] + 1)[:, None]
    weight = None
    if rho > 0.0:
        share = inside[0] / (inside[0] + outside[0])
        touches = (held.blank[:-1, columns.start] | held.blank[1:, columns.start])[:, None].double()
        weight = touches * (rho / share) + (1.0 - touches) * ((1.0 - rho) / (1.0 - share))
    return {
        "score": regime_score(1 if outside[0] else 0, mean_speed, mean_flow),
        "outside": outside,
        "inside": inside,
        "loss": mean_speed / SPEED_NORMALIZER**2 + mean_flow / FLOW_NORMALIZER**2,
        "S_LWR": lwr_score(*lwr_sums(n_answer, n_true, steps, weight))["S_LWR"],
    }


def _panel_entry(rates: list[dict]) -> dict:
    def mean_rmse(where: str, channel: int) -> float | None:
        values = [_root_mean(rate[where][channel], rate[where][0]) for rate in rates]
        return float(np.mean(values)) if all(value is not None for value in values) else None

    return {
        "S_state": float(np.mean([rate["score"] for rate in rates])),
        "val_loss": float(np.mean([rate["loss"] for rate in rates])),
        "rmse_speed": mean_rmse("outside", 1),
        "rmse_flow_per_lane": mean_rmse("outside", 2),
        "cells": sum(rate["outside"][0] for rate in rates),
        "blackout_rmse_speed": mean_rmse("inside", 1),
        "blackout_rmse_flow_per_lane": mean_rmse("inside", 2),
        "blackout_cells": sum(rate["inside"][0] for rate in rates),
        "S_LWR": float(np.mean([rate["S_LWR"] for rate in rates])),
    }


@torch.no_grad()
def validate(model: UNet, store: GridStore, run: StageRun, held: ValidationSet, config: UNetConfig) -> Validation:
    days, by_rule = held.days, {rule: {} for rule in RC.GRID}
    for position, grid in enumerate(store.grids):
        columns = store.panel_columns(position)
        shape = (len(days), SLOTS_PER_DAY, grid.width)
        true_speed = store.truth_speed[held.rows, columns].view(shape).double()
        true_flow = store.truth_flow[held.rows, columns].view(shape).double()
        free = RC.free_speed(store, columns)
        rates = {rule: [] for rule in RC.GRID}
        for mask in held.masks:
            speed, flow, density, _ = predict_days(
                model,
                store,
                position,
                days,
                config.data.crop,
                config.inference.center,
                config.inference.day_context,
                hidden=held.unseen(mask),
                gap_context=config.data.gap_context,
                tile_batch=config.inference.tile_batch,
            )
            cells = mask[:, columns].view(shape)
            for rule in RC.GRID:
                answer = RC.apply(rule, speed, flow, density, free)
                rates[rule].append(_score_rate(store, held, position, columns, cells, *answer, true_speed, true_flow))
        for rule in RC.GRID:
            by_rule[rule][grid.panel] = _panel_entry(rates[rule])

    def result(rule: RC.Rule) -> Validation:
        panels = by_rule[rule]

        def over_families(name: str) -> float | None:
            values = {panel: entry[name] for panel, entry in panels.items() if entry.get(name) is not None}
            return family_mean(values, run.family_of) if values else None

        return Validation(
            over_families("S_state"),
            float(np.mean([entry["val_loss"] for entry in panels.values()])),
            over_families("rmse_speed"),
            over_families("rmse_flow_per_lane"),
            over_families("blackout_rmse_speed"),
            over_families("blackout_rmse_flow_per_lane"),
            float(np.mean([entry["S_LWR"] for entry in panels.values()])),
            panels,
            rule,
        )

    return max((result(rule) for rule in RC.GRID), key=lambda v: v.points)


def save_checkpoint(path: Path, model: UNet, epoch: int, result: Validation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "epoch": epoch, "reconcile_alpha": result.rule.alpha, "reconcile_beta": result.rule.beta}, path
    )


def load_checkpoint(path: Path, config: UNetConfig, n_panels: int, runtime: Runtime) -> tuple[UNet, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path}: run the train step for this stage first")
    saved = torch.load(path, map_location=runtime.device, weights_only=True)
    model = build_model(config, n_panels, runtime)
    model.load_state_dict(saved.pop("model"))
    return model.eval(), saved


def rule(saved: dict) -> RC.Rule:
    return RC.Rule(float(saved["reconcile_alpha"]), float(saved["reconcile_beta"]))


def header(title: str) -> str:
    return f"== [{title}] ".ljust(HEADER_WIDTH, "=")


def short_exponent(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent):+d}"


def epoch_line(
    epoch: int, epochs: int, train_loss: float, rate: float, result: Validation, best: dict, improved: bool, seconds: float
) -> str:
    parts = [
        f"ep {epoch:2d}/{epochs}",
        f"loss: tr {train_loss:.4f} / val {result.loss:.4f}",
        f"lr: {short_exponent(rate)}",
        f"S_state: {result.score:.5f}",
        f"v: {result.rmse_speed:.2f}",
        f"q: {result.rmse_flow:.1f}",
    ]
    inside = "none" if result.blackout_rmse_speed is None else f"v {result.blackout_rmse_speed:.2f} / q {result.blackout_rmse_flow:.1f}"
    parts += [f"blackout: {inside}", f"S_LWR: {result.s_lwr:.3f}", f"rule: {result.rule.alpha:g}/{result.rule.beta:g}"]
    parts.append("best! [saved]" if improved else f"best: {best['picked']:.5f} @ep{best['epoch']}")
    return " | ".join(parts) + f" ({round(seconds)}s)"


def ramp_line(store: GridStore) -> str:
    ramps = store.ramps
    panels = int(ramps.panel.unique().numel())
    return (
        f"[ramps]    {len(ramps.is_on):,} ramps ({panels} panels) | valid: {ramps.valid_share:.1%} | "
        f"channels: {len(RAMP_CHANNELS)} | standardized by month and weekday"
    )


def level_line(store: GridStore, run: StageRun) -> str:
    months = sorted({int(store.month_of_day[day]) for day in run.fit_days})
    is_link = torch.as_tensor(np.concatenate([grid.kind == LINK for grid in store.grids]), device=store.device)
    level = store.level[months][:, is_link].abs()
    return (
        f"[level]    speed baseline + monthly offset per link (median) | train months: "
        f"mean |d| {float(level.mean()):.2f} km/h, max {float(level.max()):.1f}"
    )


def loss_line(config: UNetConfig) -> str:
    return (
        f"[loss]     {config.loss.state_weight:.2f} x S_state + {config.loss.lwr_weight:.2f} x N + "
        f"{config.loss.density_weight:.2f} x |N error| of the density head"
    )


def describe_stage(
    store: GridStore, run: StageRun, config: UNetConfig, runtime: Runtime, sampler: CropSampler, held: ValidationSet, parameters: int
) -> None:
    optim, dates = config.optim, store.dates
    if runtime.device.type == "cuda":
        memory = torch.cuda.get_device_properties(runtime.device).total_memory / 2**30
        device = f"GPU: {torch.cuda.get_device_name(runtime.device)} ({memory:.0f} GB)"
    else:
        device = f"device: {runtime.device}"
    share = len(run.holdout_days) / (len(run.fit_days) + len(run.holdout_days))
    log(f"  [env]      {device} | torch: {torch.__version__} | params: {parameters:,}")
    log(
        f"  [data]     train: {len(run.fit_days)} days | holdout: {len(run.holdout_days)} days "
        f"({dates[run.holdout_days[0]]}..{dates[run.holdout_days[-1]]}, {share:.1%})"
    )
    log(
        f"  [scale]    {sampler.patches:,} patches/ep | {sampler.steps} steps/ep | "
        f"{sampler.steps * optim.epochs:,} steps total ({optim.epochs} ep, batch {config.data.batch_size})"
    )
    panels = int((sampler.blackouts.count.sum(dim=1) > 0).sum())
    rho = [value for value in held.rho if value > 0]
    spread = f" | rho: {min(rho):.1%}-{max(rho):.1%}" if rho else ""
    log(
        f"  [blackout] train: {config.data.blackout.probability:.0%} of crops ({panels} panels), weight "
        f"{config.data.blackout.weight:.3g} | holdout: {held.blackouts} blackouts{spread}"
    )
    log("  " + ramp_line(store))
    log("  " + level_line(store, run))
    log("  " + loss_line(config))


def train_stage(store: GridStore, run: StageRun, config: UNetConfig, runtime: Runtime, held: ValidationSet) -> None:
    optim, started = config.optim, time.perf_counter()
    sampler = CropSampler(store, run.fit_days, config.data, runtime.seed, run.stage)
    model = build_model(config, run.n_panels, runtime)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = make_optimizer(model, config)
    mask_generator = seeded(runtime.seed, "train mask", run.stage, device=runtime.device)
    on_gpu = runtime.device.type == "cuda"
    score_loss = ScoreLoss(store, config.loss, config.data.mask_rates, config.data.blackout.weight)

    log(header(f"train {run.stage}"))
    describe_stage(store, run, config, runtime, sampler, held, parameter_count(model))
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(runtime.device)

    best, step = {"epoch": 0, "picked": -math.inf, "result": None}, 0
    for epoch in range(1, optim.epochs + 1):
        epoch_started, losses = time.perf_counter(), []
        model.train()
        for crops in sampler.epoch(epoch):
            rate = learning_rate(step, sampler.steps, config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            loss = training_loss(model, store, crops, mask_generator, config.data, score_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{run.stage}: the training loss became non-finite at epoch {epoch}, step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip)
            optimizer.step()
            update_ema(ema, model, optim.ema_decay)
            losses.append(loss.detach())
            step += 1
        train_loss = float(torch.stack(losses).mean())
        result = validate(ema, store, run, held, config)
        improved = result.points > best["picked"]
        if improved:
            best = {"epoch": epoch, "picked": result.points, "result": result}
            save_checkpoint(run.checkpoint, ema, epoch, result)
        if epoch == 1 and on_gpu:
            reserved = torch.cuda.max_memory_reserved(runtime.device) / 2**30
            log(
                f"  [memory]   peak VRAM: {torch.cuda.max_memory_allocated(runtime.device) / 2**30:.1f} GB allocated | {reserved:.1f} GB reserved"
            )
        log(epoch_line(epoch, optim.epochs, train_loss, rate, result, best, improved, time.perf_counter() - epoch_started))
    kept = best["result"]
    log(
        f"  [done]     best: ep {best['epoch']} | points: {kept.points:.5f} | S_state: {kept.score:.5f} | S_LWR: {kept.s_lwr:.3f} | "
        f"reconcile: alpha {kept.rule.alpha:g}, beta {kept.rule.beta:g} | {duration(time.perf_counter() - started)}"
    )
