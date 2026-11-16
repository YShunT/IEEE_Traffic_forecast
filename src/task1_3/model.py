"""U-Net 本体：入力チャネル、隠し方、密度への寄せ方、モデル、点数どおりの損失。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import nn

from core.config import (
    BLACKOUT_SLOTS,
    FLOW_NORMALIZER,
    FLOW_WEIGHT,
    MIN_SPEED_KMH,
    QUEUE_SPEED_RATIO,
    SLOTS_PER_DAY,
    SPEED_NORMALIZER,
    SPEED_WEIGHT,
    WINDOW_HISTORY_SLOTS,
)
from core.runtime import accumulate
from task1_3.config import GapContextConfig, LossConfig, UNetModelConfig
from task1_3.data import STATIC_CHANNELS, STATIC_SCALES
from task1_3.store import GridStore

DENSITY_LOG_CAP = 3.0
MIN_DENSITY = 1e-3
STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Rule:
    alpha: float
    beta: float


GRID = [Rule(a, b) for a, b in itertools.product(STEPS, STEPS)]
KEEP = Rule(0.0, 0.0)


def density(base_speed: torch.Tensor, base_flow: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return base_flow.clamp(min=0.0) / base_speed.clamp(min=MIN_SPEED_KMH) * output.clamp(-DENSITY_LOG_CAP, DENSITY_LOG_CAP).exp()


def apply(
    rule: Rule, speed: torch.Tensor, flow: torch.Tensor, k: torch.Tensor, free_speed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == KEEP:
        return speed, flow
    congested = speed <= QUEUE_SPEED_RATIO * free_speed
    speed_from_k = torch.minimum(flow / k.clamp(min=MIN_DENSITY), free_speed)
    moved_speed = torch.where(congested, (1.0 - rule.alpha) * speed + rule.alpha * speed_from_k, speed)
    moved_flow = torch.where(congested, flow, (1.0 - rule.beta) * flow + rule.beta * k * speed)
    return moved_speed.clamp(min=0.0), moved_flow.clamp(min=0.0)


def free_speed(store, columns) -> torch.Tensor:
    return store.static[STATIC_CHANNELS.index("free_speed"), columns] * STATIC_SCALES["free_speed"]


DYNAMIC_CHANNELS = ("speed_residual", "flow_residual", "observed", "pct_observed")
HISTORY_CHANNELS = ("hist_mean_speed", "hist_mean_flow_per_lane", "hist_std_speed", "hist_count")
TIME_CHANNELS = ("time_sin", "time_cos", *(f"weekday_{day}" for day in range(7)))
CHANNELS = DYNAMIC_CHANNELS + HISTORY_CHANNELS + TIME_CHANNELS + STATIC_CHANNELS
GAP_CHANNELS = (
    "before_speed_residual",
    "before_flow_residual",
    "before_distance",
    "after_speed_residual",
    "after_flow_residual",
    "after_distance",
)


RAMP_CHANNELS = (
    "panel_on_ramp",
    "panel_off_ramp",
    "panel_ramp_coverage",
    "link_on_ramp",
    "link_off_ramp",
    "link_on_ramp_valid",
    "link_off_ramp_valid",
)


INPUT_CHANNELS = CHANNELS + GAP_CHANNELS + RAMP_CHANNELS

HISTORY_SCALES = {"hist_mean_speed": 100.0, "hist_mean_flow_per_lane": 2000.0, "hist_std_speed": SPEED_NORMALIZER, "hist_count": 40.0}
PCT_SCALE = 100.0


@dataclass
class Inputs:
    x: torch.Tensor
    panel: torch.Tensor
    base_speed: torch.Tensor
    base_flow: torch.Tensor


def _context_before(speed: torch.Tensor, flow: torch.Tensor, observed: torch.Tensor, config: GapContextConfig) -> list[torch.Tensor]:
    time = observed.shape[1]
    index = torch.arange(time, device=observed.device).view(1, time, 1)
    seen = observed.to(speed.dtype)

    def ending_here(values: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(values, (0, 0, config.mean_slots - 1, 0))
        return sum(padded[:, k : k + time] for k in range(config.mean_slots))

    count = ending_here(seen).clamp(min=1.0)
    means = [ending_here(speed) / count, ending_here(flow) / count]
    last = torch.cummax(torch.where(observed, index, -1), dim=1).values
    last = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], dim=1)
    found = last >= 0
    at = last.clamp(min=0)
    distance = torch.where(found, index - last, config.max_distance).clamp(max=config.max_distance)
    return [*(torch.where(found, torch.gather(mean, 1, at), 0.0) for mean in means), distance.to(speed.dtype) / BLACKOUT_SLOTS]


def gap_context(speed: torch.Tensor, flow: torch.Tensor, observed: torch.Tensor, config: GapContextConfig) -> list[torch.Tensor]:
    before = _context_before(speed, flow, observed, config)
    after = _context_before(speed.flip(1), flow.flip(1), observed.flip(1), config)
    return before + [channel.flip(1) for channel in after]


def ramp_channels(store: GridStore, t: torch.Tensor, c: torch.Tensor) -> list[torch.Tensor]:
    ramps = store.ramps
    inside = (t >= 0) & (t < store.slots)
    rows = t.clamp(0, store.slots - 1)
    space = c.shape[1]
    level = ramps.level[rows, store.position[c[:, 0]][:, None]] * inside[:, :, None]
    channels = [level[:, :, k, None].expand(-1, -1, space) for k in range(3)]
    means, flags = [], []
    for table in (ramps.column_on, ramps.column_off):
        index = table[c]
        z = ramps.z[rows[:, :, None, None], index.clamp(min=0)[:, None]]
        valid = torch.isfinite(z) & (index >= 0)[:, None] & inside[:, :, None, None]
        count = valid.sum(dim=3)
        means.append((torch.where(valid, z, 0.0).sum(dim=3) / count.clamp(min=1)).to(store.dtype))
        flags.append((count > 0).to(store.dtype))
    return channels + means + flags


def assemble(
    store: GridStore,
    t: torch.Tensor,
    c: torch.Tensor,
    speed: torch.Tensor,
    flow: torch.Tensor,
    pct: torch.Tensor,
    context: GapContextConfig,
) -> Inputs:
    mean_speed, mean_flow, std_speed, count = store.history(t, c)
    observed = torch.isfinite(speed) & torch.isfinite(flow)
    dynamic = [
        torch.where(observed, (speed - mean_speed) / SPEED_NORMALIZER, 0.0),
        torch.where(observed, (flow - mean_flow) / FLOW_NORMALIZER, 0.0),
        observed.to(speed.dtype),
        pct.to(speed.dtype) / PCT_SCALE,
    ]
    history = [
        mean_speed / HISTORY_SCALES["hist_mean_speed"],
        mean_flow / HISTORY_SCALES["hist_mean_flow_per_lane"],
        std_speed / HISTORY_SCALES["hist_std_speed"],
        count / HISTORY_SCALES["hist_count"],
    ]

    batch, time, space = speed.shape
    week_slot = store.weekslot(t)
    angle = (week_slot % SLOTS_PER_DAY).to(speed.dtype) * (2 * math.pi / SLOTS_PER_DAY)
    weekday = torch.nn.functional.one_hot(week_slot // SLOTS_PER_DAY, 7).to(speed.dtype)
    clock = torch.cat([angle.sin()[:, :, None], angle.cos()[:, :, None], weekday], dim=2)
    static = store.static[:, c].permute(1, 0, 2)
    parts = [
        torch.stack(dynamic + history, dim=1),
        clock.permute(0, 2, 1)[:, :, :, None].expand(batch, len(TIME_CHANNELS), time, space),
        static[:, :, None, :].expand(batch, len(STATIC_CHANNELS), time, space),
        torch.stack(gap_context(dynamic[0], dynamic[1], observed, context), dim=1),
        torch.stack(ramp_channels(store, t, c), dim=1),
    ]
    x = torch.cat(parts, dim=1)
    return Inputs(x, store.panel[c[:, 0]], mean_speed, mean_flow)


def residual_targets(inputs: Inputs, speed: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    target = torch.stack([(speed - inputs.base_speed) / SPEED_NORMALIZER, (flow - inputs.base_flow) / FLOW_NORMALIZER], dim=1)
    return torch.nan_to_num(target, nan=0.0)


def to_state(inputs: Inputs, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    speed = (inputs.base_speed + SPEED_NORMALIZER * output[:, 0]).clamp(min=0.0)
    flow = (inputs.base_flow + FLOW_NORMALIZER * output[:, 1]).clamp(min=0.0)
    return speed, flow


def to_density(inputs: Inputs, output: torch.Tensor) -> torch.Tensor:
    return density(inputs.base_speed, inputs.base_flow, output[:, 2])


def remask(eligible: torch.Tensor, rate: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    draws = torch.rand(eligible.shape, generator=generator, device=eligible.device)
    return eligible & (draws < rate[:, None, None])


def hide(values: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return torch.where(hidden, torch.nan, values)


def blackout_rows(start: torch.Tensor, time: int) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.arange(time, device=start.device)[None, :] - start[:, None]
    has = (start >= 0)[:, None]
    blank = has & (position >= 0) & (position < BLACKOUT_SLOTS)
    return blank, has & (position < 0) & (position >= -WINDOW_HISTORY_SLOTS)


def with_blackout(hidden: torch.Tensor, start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blank, history = (rows[:, :, None] for rows in blackout_rows(start, hidden.shape[1]))
    hidden = hidden & ~history & ~blank
    return hidden, hidden | blank, blank


def cell_weights(
    at_rate: torch.Tensor, in_blank: torch.Tensor | None, links: torch.Tensor, rate: torch.Tensor, blackout_weight: float
) -> torch.Tensor:
    at_rate = at_rate.to(links.dtype)
    per_link = 1.0 / links
    weights = at_rate * (per_link / rate)[:, None, None]
    return weights if in_blank is None else weights + in_blank * (per_link * blackout_weight)[:, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Identity() if channels_in == channels_out else nn.Conv2d(channels_in, channels_out, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class TimeBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        return x + self.conv2(self.act(self.norm2(h)))


class UNet(nn.Module):
    def __init__(self, in_channels: int, n_panels: int, config: UNetModelConfig, out_channels: int = 3):
        super().__init__()
        widths, groups = config.channels, config.norm_groups
        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.panel = nn.Embedding(n_panels, widths[0])
        self.encoders = nn.ModuleList(ResidualBlock(width, width, groups) for width in widths)
        self.downs = nn.ModuleList(nn.Conv2d(fine, coarse, 2, stride=2) for fine, coarse in zip(widths, widths[1:]))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(coarse, fine, 2, stride=2) for fine, coarse in zip(widths, widths[1:]))
        self.decoders = nn.ModuleList(ResidualBlock(2 * width, width, groups) for width in widths[:-1])
        self.head = nn.Sequential(nn.GroupNorm(groups, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], out_channels, 1))
        nn.init.normal_(self.panel.weight, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.time_blocks = nn.ModuleList(TimeBlock(widths[-1], groups) for _ in range(config.time_blocks))

    def forward(self, x: torch.Tensor, panel: torch.Tensor) -> torch.Tensor:
        h = self.stem(x) + self.panel(panel)[:, :, None, None]
        skips = []
        for level, encoder in enumerate(self.encoders):
            h = encoder(h)
            if level < len(self.downs):
                skips.append(h)
                h = self.downs[level](h)
        for block in self.time_blocks:
            h = block(h)
        for level in reversed(range(len(self.ups))):
            h = self.decoders[level](torch.cat([self.ups[level](h), skips[level]], dim=1))
        return self.head(h)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


CHUNK_DAYS = 32
MIN_RMSE = 1e-3


def accumulations(store: GridStore, c: torch.Tensor, speed: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    scale = (store.length[c] * store.lanes[c]).to(speed.dtype)[:, None, :]
    return scale * flow / speed.clamp(min=MIN_SPEED_KMH)


def panel_scales(store: GridStore) -> torch.Tensor:
    fit = store.tables.fit_day
    scales = []
    for position in range(len(store.grids)):
        columns = store.panel_columns(position)
        c = torch.arange(columns.start, columns.stop, device=store.device)[None]
        total, steps = 0.0, 0
        for first in range(0, store.n_train_days, CHUNK_DAYS):
            last = min(first + CHUNK_DAYS, store.n_train_days)
            rows = slice(first * SLOTS_PER_DAY, min(last * SLOTS_PER_DAY + 1, store.n_train_days * SLOTS_PER_DAY))
            eligible = store.eligible[rows, columns]
            n = accumulations(
                store,
                c,
                torch.where(eligible, store.truth_speed[rows, columns], 1.0).double()[None],
                torch.where(eligible, store.truth_flow[rows, columns], 0.0).double()[None],
            )[0]
            day = torch.arange(rows.start, rows.stop, device=store.device) // SLOTS_PER_DAY
            ok = eligible[:-1] & eligible[1:] & store.ramp_ok[rows, columns][:-1] & (fit[day[:-1]] & fit[day[1:]])[:, None]
            total += float(torch.where(ok, (n[1:] - n[:-1]).abs(), 0.0).sum())
            steps += int(ok.sum())
        scales.append(total / max(steps, 1))
    return torch.tensor(scales, dtype=torch.float64, device=store.device)


class ScoreLoss:
    def __init__(self, store: GridStore, config: LossConfig, rates: list[float], blackout_weight: float):
        self.config, self.blackout_weight = config, blackout_weight
        self.rates = torch.tensor(rates, dtype=store.dtype, device=store.device)
        self.groups = len(store.grids) * len(rates)
        self.mean_square = torch.full((2, self.groups), torch.nan, dtype=torch.float64, device=store.device)
        self.scale = panel_scales(store)
        self.target_weight = torch.tensor([SPEED_WEIGHT, FLOW_WEIGHT], dtype=torch.float64, device=store.device)
        self.last = (0.0, 0.0, 0.0)

    def state_term(self, output: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        squared = ((output - target) ** 2 * weights[:, None]).sum(dim=(2, 3)).double()
        weight = weights.sum(dim=(1, 2)).double()
        with torch.no_grad():
            total = torch.stack(
                [accumulate(torch.zeros(self.groups, dtype=torch.float64, device=group.device), group, squared[:, k]) for k in range(2)]
            )
            count = accumulate(torch.zeros(self.groups, dtype=torch.float64, device=group.device), group, weight)
            seen = count > 0
            now = total / count.clamp(min=torch.finfo(torch.float64).tiny)
            kept = torch.where(
                torch.isfinite(self.mean_square), self.config.decay * self.mean_square + (1.0 - self.config.decay) * now, now
            )
            self.mean_square = torch.where(seen[None], kept, self.mean_square)
            root = self.mean_square.sqrt().clamp(min=MIN_RMSE)[:, group].T
        per_crop = (self.target_weight * (squared / (2.0 * root) + weight[:, None] * root / 2.0)).sum(dim=1)
        return per_crop.sum() / weight.sum().clamp(min=torch.finfo(torch.float64).tiny)

    def lwr_term(
        self,
        store: GridStore,
        crops,
        inputs: Inputs,
        output: torch.Tensor,
        speed: torch.Tensor,
        flow: torch.Tensor,
        answered: torch.Tensor,
        scoreable: torch.Tensor,
        blank: torch.Tensor | None,
    ) -> torch.Tensor:
        answer_speed, answer_flow = to_state(inputs, output)
        true_speed, true_flow = torch.where(scoreable, speed, 1.0), torch.where(scoreable, flow, 0.0)
        n_true = accumulations(store, crops.c, true_speed, true_flow)
        n_answer = accumulations(
            store, crops.c, torch.where(answered, answer_speed, true_speed), torch.where(answered, answer_flow, true_flow)
        )
        steps = scoreable[:, :-1] & scoreable[:, 1:] & store.gather(store.ramp_ok, crops.t[:, :-1], crops.c)
        gap = (n_answer[:, 1:] - n_answer[:, :-1]) - (n_true[:, 1:] - n_true[:, :-1])
        smooth = self.config.smoothing
        residual = (gap**2 + smooth**2).sqrt() - smooth
        position = store.position[crops.c[:, 0]]
        weight = steps.to(output.dtype) / store.links[crops.c[:, 0]][:, None, None]
        if blank is not None:
            touches = blank[:, :-1] | blank[:, 1:]
            weight = weight * torch.where(touches, (crops.rate * self.blackout_weight)[:, None, None], 1.0)
        scale = self.scale[position].to(output.dtype)[:, None, None]
        return (residual * weight / scale).sum() / weight.sum().clamp(min=torch.finfo(output.dtype).tiny)

    def density_term(
        self,
        store: GridStore,
        crops,
        inputs: Inputs,
        output: torch.Tensor,
        speed: torch.Tensor,
        flow: torch.Tensor,
        weights: torch.Tensor,
        scoreable: torch.Tensor,
    ) -> torch.Tensor:
        k_true = torch.where(scoreable, flow, 0.0) / torch.where(scoreable, speed, 1.0).clamp(min=MIN_SPEED_KMH)
        scale = (store.length[crops.c] * store.lanes[crops.c]).to(output.dtype)[:, None, :]
        error = scale * (to_density(inputs, output) - k_true).abs()
        weight = weights * scoreable
        position = store.position[crops.c[:, 0]]
        return (error * weight / self.scale[position].to(output.dtype)[:, None, None]).sum() / weight.sum().clamp(
            min=torch.finfo(output.dtype).tiny
        )

    def __call__(
        self,
        store: GridStore,
        crops,
        inputs: Inputs,
        output: torch.Tensor,
        target: torch.Tensor,
        speed: torch.Tensor,
        flow: torch.Tensor,
        weights: torch.Tensor,
        scoreable: torch.Tensor,
        blank: torch.Tensor | None,
    ) -> torch.Tensor:
        rate = (crops.rate[:, None] - self.rates[None]).abs().argmin(dim=1)
        group = store.position[crops.c[:, 0]] * len(self.rates) + rate
        state = self.state_term(output[:, :2], target, weights, group)
        lwr = self.lwr_term(store, crops, inputs, output, speed, flow, weights > 0, scoreable, blank)
        density = self.density_term(store, crops, inputs, output, speed, flow, weights, scoreable)
        self.last = (float(state.detach()), float(lwr.detach()), float(density.detach()))
        return (self.config.state_weight * state + self.config.lwr_weight * lwr + self.config.density_weight * density).to(output.dtype)
