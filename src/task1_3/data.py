"""U-Net の材料：blackout、区間の履歴平均、S_LWR、本線の配列、道路に沿う空間軸。"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from core.config import (
    BLACKOUT_SLOTS,
    DEFAULT_LENGTH_KM,
    MIN_PCT_OBSERVED,
    MIN_SPEED_KMH,
    SLOTS_PER_DAY,
    SLOTS_PER_WEEK,
    SUBMISSION_SPLITS,
    TIMESTAMP_FORMAT,
    WINDOW_HISTORY_SLOTS,
    WINDOW_HORIZON_SLOTS,
    WINDOW_MIN_SEPARATION,
)
from core.runtime import Runtime, accumulate, derive_seed, to_numpy
from data.panel import lanes_by_link, week_slot, weekday_and_slot
from data.release import (
    Release,
    Road,
    date_of,
    mainline_days,
    read_day,
    read_fd_parameters,
    read_links,
    read_state_template,
    read_topology,
    read_window_index,
)

ONSET, ONGOING = "queue_onset", "queue_ongoing"
KINDS = (ONSET, ONGOING)


@dataclass(frozen=True)
class Blackout:
    panel: str
    date: str
    slot: int
    kind: str


def slot_of(timestamp: str) -> int:
    return int(timestamp[11:13]) * 12 + int(timestamp[14:16]) // 5


def window_panels(release: Release) -> list[str]:
    with_windows = {path.parent.name for split in SUBMISSION_SPLITS for path in release.task2_dirs(split)}
    return [panel for panel in release.panels() if panel in with_windows]


def released_windows(release: Release, split: str) -> list[Blackout]:
    index = read_window_index(release, split)
    return [
        Blackout(row.panel, row.forecast_start[:10], slot_of(row.forecast_start), row.condition) for row in index.itertuples(index=False)
    ]


def queued_slots(speed: np.ndarray, eligible: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (eligible & (speed <= thresholds[None, :])).any(axis=1)


def candidates(queued: np.ndarray) -> dict[str, np.ndarray]:
    by_day = queued.reshape(-1, SLOTS_PER_DAY)
    before = np.concatenate([np.zeros((len(by_day), 1), dtype=np.int64), np.cumsum(by_day, axis=1)], axis=1)
    slot = np.arange(WINDOW_HISTORY_SLOTS, SLOTS_PER_DAY - BLACKOUT_SLOTS + 1)
    in_history = before[:, slot] - before[:, slot - WINDOW_HISTORY_SLOTS]
    ahead = before[:, slot + WINDOW_HORIZON_SLOTS] - before[:, slot]
    out = {}
    for kind, allowed in ((ONSET, (in_history == 0) & (ahead > 0)), (ONGOING, by_day[:, slot - 1])):
        full = np.zeros(by_day.shape, dtype=bool)
        full[:, slot] = allowed
        out[kind] = full.reshape(-1)
    return out


def draw(
    pools: dict[str, np.ndarray], wanted: Sequence[str], rng: np.random.Generator, separation: int = WINDOW_MIN_SEPARATION
) -> list[tuple[int, str]]:
    chosen: list[tuple[int, str]] = []
    for kind in wanted:
        for candidate_kind in (kind, *(k for k in KINDS if k != kind)):
            pool = pools[candidate_kind]
            if chosen:
                taken = np.array([start for start, _ in chosen])
                pool = pool[(np.abs(pool[:, None] - taken[None, :]) >= separation).all(axis=1)]
            if len(pool):
                chosen.append((int(pool[rng.integers(len(pool))]), candidate_kind))
                break
    return sorted(chosen)


def place_on_days(
    panel: str, allowed: dict[str, np.ndarray], dates: Sequence[str], days: Sequence[int], count: int, seed: int
) -> list[Blackout]:
    placed = []
    for day in days:
        rng = np.random.default_rng(derive_seed(seed, "validation blackout", panel, dates[day]))
        rows = slice(day * SLOTS_PER_DAY, (day + 1) * SLOTS_PER_DAY)
        pools = {kind: np.flatnonzero(allowed[kind][rows]) for kind in KINDS}
        wanted = [KINDS[int(rng.integers(2))] for _ in range(count)]
        placed += [Blackout(panel, dates[day], slot, kind) for slot, kind in draw(pools, wanted, rng)]
    return placed


def blank_timestamps(blackouts: Sequence[Blackout], first: int = 0, last: int = BLACKOUT_SLOTS) -> set[str]:
    stamps = set()
    for blackout in blackouts:
        start = datetime.strptime(blackout.date, "%Y-%m-%d") + timedelta(minutes=5 * blackout.slot)
        stamps |= {(start + timedelta(minutes=5 * k)).strftime(TIMESTAMP_FORMAT) for k in range(first, last)}
    return stamps


def template_share(release: Release, panel: str) -> float:
    inside = total = 0
    for split in SUBMISSION_SPLITS:
        blank = blank_timestamps([b for b in released_windows(release, split) if b.panel == panel])
        timestamps = read_state_template(release, panel, split)["timestamp"]
        inside += int(timestamps.isin(blank).sum())
        total += len(timestamps)
    return inside / total


FIT_COLUMNS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"]
CHANNELS = ("speed_kmh", "flow_vph")


def eligible_rows(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame["pct_observed"], errors="coerce").ge(MIN_PCT_OBSERVED) & frame["speed_kmh"].notna() & frame["flow_vph"].notna()
    ).to_numpy()


def link_positions(link_ids: pd.Series, index: dict[str, int]) -> np.ndarray:
    positions = link_ids.map(index)
    if positions.isna().any():
        unknown = sorted(link_ids[positions.isna()].unique())[:5]
        raise KeyError(f"links absent from the profile: {unknown}")
    return positions.to_numpy(dtype=np.int64)


def profile(release: Release, panel: str, fit_days: Collection[str], runtime: Runtime) -> dict:
    days = mainline_days(release, panel)
    link_ids = sorted(read_day(days[0].path, ["link_id"])["link_id"].unique())
    index = {link: i for i, link in enumerate(link_ids)}
    size = len(link_ids) * SLOTS_PER_WEEK
    sums = {c: torch.zeros(size, dtype=runtime.dtype, device=runtime.device) for c in CHANNELS}
    counts = {c: torch.zeros_like(sums[c]) for c in CHANNELS}
    squares = torch.zeros_like(sums["speed_kmh"])
    chosen = set(fit_days)
    for day in days:
        if day.date not in chosen:
            continue
        frame = read_day(day.path, FIT_COLUMNS)
        cell = link_positions(frame["link_id"], index) * SLOTS_PER_WEEK + week_slot(frame["timestamp"])
        usable = eligible_rows(frame)
        for channel in CHANNELS:
            values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=np.float64)
            keep = usable & np.isfinite(values)
            accumulate(sums[channel], runtime.index(cell[keep]), runtime.floats(values[keep]))
            accumulate(counts[channel], runtime.index(cell[keep]), runtime.floats(np.ones(keep.sum())))
            if channel == "speed_kmh":
                accumulate(squares, runtime.index(cell[keep]), runtime.floats(values[keep] ** 2))
    shape = (len(link_ids), SLOTS_PER_WEEK)
    return dict(
        count=to_numpy(counts["speed_kmh"].view(shape)),
        speed_square_total=to_numpy(squares.view(shape)),
        **{f"total_{c}": to_numpy(sums[c].view(shape)) for c in CHANNELS},
    )


RAMP_COLUMNS = ["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed", "is_score_eligible"]
ON_TYPES, OFF_TYPES = ("OR", "ON"), ("FR", "OFF")


def ramp_validity(release: Release, panel: str, split: str, dates: Sequence[str], link_ids: Sequence[str]) -> np.ndarray:
    index = {link: i for i, link in enumerate(link_ids)}
    day_of = {day: i for i, day in enumerate(dates)}
    topology = read_topology(release, panel).drop_duplicates("link_id").set_index("link_id").reindex(list(link_ids))
    attached = {kind: (topology[f"{kind}_ramp_link_ids"].fillna("") != "").to_numpy() for kind in ("on", "off")}
    attachment = pd.read_csv(release.network_file(panel, "ramp_attachment_map.csv"), dtype=str)
    mainline_of = dict(zip(attachment["ramp_link_id"], attachment["nearest_mainline_link_id"]))

    valid = {kind: np.zeros((len(dates) * SLOTS_PER_DAY, len(link_ids)), dtype=bool) for kind in ("on", "off")}
    for path in release.ramp_files(panel, split):
        if date_of(path) not in day_of:
            continue
        frame = pd.read_parquet(path, columns=RAMP_COLUMNS)
        timestamp = frame["timestamp"].astype(str)
        day = timestamp.str[:10].map(day_of)
        usable = (
            day.notna()
            & frame["is_score_eligible"].astype(bool)
            & (pd.to_numeric(frame["pct_observed"], errors="coerce") >= MIN_PCT_OBSERVED)
            & pd.to_numeric(frame["flow_vph"], errors="coerce").notna()
        )
        link = frame["ramp_link_id"].astype(str).map(mainline_of).map(index)
        usable &= link.notna()
        if not usable.any():
            continue
        rows = (
            day[usable].to_numpy(dtype=np.int64) * SLOTS_PER_DAY
            + timestamp[usable].str[11:13].astype(int).to_numpy() * 12
            + timestamp[usable].str[14:16].astype(int).to_numpy() // 5
        )
        columns = link[usable].to_numpy(dtype=np.int64)
        kind = frame.loc[usable, "ramp_type"].astype(str).str.upper()
        for name, types in (("on", ON_TYPES), ("off", OFF_TYPES)):
            chosen = kind.isin(types).to_numpy()
            valid[name][rows[chosen], columns[chosen]] = True
    return (valid["on"] | ~attached["on"][None, :]) & (valid["off"] | ~attached["off"][None, :])


def accumulation(length: torch.Tensor, speed: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return length * flow / speed.clamp(min=MIN_SPEED_KMH)


def lwr_sums(
    n_submitted: torch.Tensor, n_true: torch.Tensor, scored: torch.Tensor, weight: torch.Tensor | None = None
) -> tuple[float, float]:
    true_change = n_true[1:] - n_true[:-1]
    residual = ((n_submitted[1:] - n_submitted[:-1]) - true_change).abs()
    signal = true_change.abs()
    if weight is not None:
        residual, signal = residual * weight, signal * weight
    zero = torch.zeros((), dtype=residual.dtype, device=residual.device)
    return float(torch.where(scored, residual, zero).sum()), float(torch.where(scored, signal, zero).sum())


def lwr_score(residual: float, signal: float) -> dict:
    if signal <= 0.0:
        return {"E_LWR": 1.0, "S_LWR": 0.0}
    error = residual / signal
    return {"E_LWR": error, "S_LWR": max(0.0, 1.0 - min(1.0, error))}


def to_grids(frame: pd.DataFrame, link_index: dict[str, int], columns: Sequence[str]) -> dict[str, np.ndarray]:
    if len(frame) != SLOTS_PER_DAY * len(link_index):
        raise ValueError(f"a day has {len(frame)} rows, expected {SLOTS_PER_DAY} slots x {len(link_index)} links")
    _, slot = weekday_and_slot(frame["timestamp"])
    link = link_positions(frame["link_id"], link_index)
    grids = {}
    for column in columns:
        grid = np.full((SLOTS_PER_DAY, len(link_index)), np.nan)
        grid[slot, link] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        grids[column] = grid
    return grids


LINK, PAD = 0, 1

STATIC_CHANNELS = (
    "log_length_km",
    "cumulative_distance",
    "lanes",
    "capacity_per_lane",
    "free_speed",
    "has_on_ramp",
    "has_off_ramp",
    "is_pad",
)
MIN_LENGTH_KM = 0.01
STATIC_SCALES = {"lanes": 4.0, "capacity_per_lane": 2000.0, "free_speed": 100.0}


@dataclass(frozen=True)
class PanelGrid:
    panel: str
    link_ids: list[str]
    column_ids: list[str]
    kind: np.ndarray
    link: np.ndarray
    lanes: np.ndarray
    static: np.ndarray
    length_km: np.ndarray | None = None

    @property
    def width(self) -> int:
        return len(self.column_ids)

    def column_of(self) -> dict[str, int]:
        return {self.column_ids[column]: column for column in np.flatnonzero(self.kind == LINK)}


def static_channels(
    columns: list[tuple[str, int]], links: pd.DataFrame, lanes: dict[str, float], topology: pd.DataFrame, centre: np.ndarray
) -> np.ndarray:
    kind = np.array([k for _, k in columns])
    is_link = kind == LINK
    ids = [link for link, k in columns if k == LINK]
    by_link = links.set_index("link_id")

    def of_links(values) -> np.ndarray:
        out = np.zeros(len(columns))
        out[is_link] = np.asarray(values, dtype=np.float64)
        return out

    length = of_links(by_link.loc[ids, "length_km"])
    lane_count = of_links([lanes.get(link, 1.0) for link in ids])
    capacity = of_links(by_link.loc[ids, "capacity_vph"]) / np.maximum(lane_count, 1.0)
    ramps = topology.set_index("link_id")

    channels = {
        "log_length_km": np.where(kind == PAD, 0.0, np.log(np.maximum(length, MIN_LENGTH_KM))),
        "cumulative_distance": (centre - centre.min()) / max(np.ptp(centre), 1e-9),
        "lanes": lane_count / STATIC_SCALES["lanes"],
        "capacity_per_lane": capacity / STATIC_SCALES["capacity_per_lane"],
        "free_speed": of_links(by_link.loc[ids, "free_speed_kmh"]) / STATIC_SCALES["free_speed"],
        "has_on_ramp": of_links(ramps.loc[ids, "on_ramp_link_ids"] != ""),
        "has_off_ramp": of_links(ramps.loc[ids, "off_ramp_link_ids"] != ""),
        "is_pad": kind == PAD,
    }
    return np.stack([channels[name] for name in STATIC_CHANNELS]).astype(np.float32)


def build_grid(release: Release, panel: str, min_width: int) -> PanelGrid:
    link_ids = sorted(read_day(mainline_days(release, panel)[0].path, ["link_id"])["link_id"].unique())
    links = read_links(release, panel)
    if sorted(links["link_id"]) != link_ids:
        raise ValueError(f"{panel}: links.csv and the state table list different links")
    lanes = lanes_by_link(links, read_fd_parameters(release, panel))
    along = Road(release.root, panel, link_ids)
    columns = [(link_ids[k], LINK) for k in along.order]
    centre = ((along.start + along.end) / 2)[along.order]
    if len(columns) < min_width:
        centre = np.r_[centre, np.full(min_width - len(columns), centre[-1])]
        columns += [("", PAD)] * (min_width - len(columns))

    kind = np.array([k for _, k in columns], dtype=np.int64)
    index = {link: i for i, link in enumerate(link_ids)}
    link = np.array([index[c] if k == LINK else -1 for c, k in columns], dtype=np.int64)
    length = pd.to_numeric(links.set_index("link_id")["length_km"], errors="coerce").fillna(DEFAULT_LENGTH_KM)
    return PanelGrid(
        panel=panel,
        link_ids=link_ids,
        column_ids=[c for c, _ in columns],
        kind=kind,
        link=link,
        lanes=np.array([lanes.get(c, 1.0) if k == LINK else 1.0 for c, k in columns], dtype=np.float64),
        static=static_channels(columns, links, lanes, read_topology(release, panel), centre),
        length_km=np.array([length[c] if k == LINK else DEFAULT_LENGTH_KM for c, k in columns], dtype=np.float64),
    )
