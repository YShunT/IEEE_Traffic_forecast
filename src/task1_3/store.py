"""U-Net の入力：配布データから作るパネルの配列と、それを GPU 上に置く GridStore、月ごとの速度水準。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from core.config import DEFAULT_FREE_SPEED_KMH, MIN_PCT_OBSERVED, QUEUE_SPEED_RATIO, SLOTS_PER_DAY, SLOTS_PER_WEEK, SPLITS
from core.runtime import Runtime
from data.release import Release, date_of, mainline_days, masked_days, read_day, read_links
from task1_3.config import LevelConfig, RampConfig
from task1_3.data import KINDS, ON_TYPES, RAMP_COLUMNS, PanelGrid, candidates, eligible_rows, queued_slots, ramp_validity, to_grids

MASKED_COLUMNS = ("speed_kmh", "flow_vph", "pct_observed")
TRUTH_COLUMNS = ("speed_kmh", "flow_vph", "pct_observed", "is_score_eligible")
HISTORY_CHUNK_DAYS = 32


def calendar(release: Release, panel: str) -> list[str]:
    dates = sorted(day.date for split in SPLITS for day in masked_days(release, panel, split))
    for earlier, later in zip(dates[:-1], dates[1:]):
        if date.fromisoformat(later) - date.fromisoformat(earlier) != timedelta(days=1):
            raise ValueError(f"{panel}: the calendar jumps from {earlier} to {later}")
    return dates


def read_grids(release: Release, panel: str, link_ids: list[str]) -> dict:
    index = {link: i for i, link in enumerate(link_ids)}
    masked = sorted((day for split in SPLITS for day in masked_days(release, panel, split)), key=lambda d: d.date)
    dates = calendar(release, panel)
    train = mainline_days(release, panel)
    if [day.date for day in train] != dates[: len(train)]:
        raise ValueError(f"{panel}: the train days are not the first days of the calendar")

    def empty(n_days: int, dtype, fill) -> np.ndarray:
        return np.full((n_days * SLOTS_PER_DAY, len(link_ids)), fill, dtype=dtype)

    def rows(position: int) -> slice:
        return slice(position * SLOTS_PER_DAY, (position + 1) * SLOTS_PER_DAY)

    masked_speed, masked_flow = empty(len(dates), np.float32, np.nan), empty(len(dates), np.float32, np.nan)
    pct = empty(len(dates), np.uint8, 0)
    for i, day in enumerate(masked):
        grids = to_grids(read_day(day.path, ["timestamp", "link_id", *MASKED_COLUMNS]), index, MASKED_COLUMNS)
        masked_speed[rows(i)], masked_flow[rows(i)] = grids["speed_kmh"], grids["flow_vph"]
        pct[rows(i)] = np.nan_to_num(grids["pct_observed"]).clip(0, 100)

    truth_speed, truth_flow = empty(len(train), np.float32, np.nan), empty(len(train), np.float32, np.nan)
    eligible = empty(len(train), bool, False)
    for i, day in enumerate(train):
        frame = read_day(day.path, ["timestamp", "link_id", *TRUTH_COLUMNS])
        if not np.array_equal(eligible_rows(frame), frame["is_score_eligible"].to_numpy(dtype=bool)):
            raise ValueError(f"{panel} {day.date}: is_score_eligible differs from the history's evidence rule")
        grids = to_grids(frame, index, ("speed_kmh", "flow_vph", "is_score_eligible"))
        truth_speed[rows(i)], truth_flow[rows(i)] = grids["speed_kmh"], grids["flow_vph"]
        eligible[rows(i)] = grids["is_score_eligible"] > 0.5

    return dict(
        dates=dates,
        n_train_days=len(train),
        masked_speed=masked_speed,
        masked_flow=masked_flow,
        pct_observed=pct,
        truth_speed=truth_speed,
        truth_flow=truth_flow,
        eligible=eligible,
    )


def blackout_candidates(release: Release, panel: str, link_ids: list[str], arrays: dict) -> dict[str, np.ndarray]:
    links = read_links(release, panel)
    free_speed = pd.to_numeric(links["free_speed_kmh"], errors="coerce").fillna(DEFAULT_FREE_SPEED_KMH)
    thresholds = dict(zip(links["link_id"], (QUEUE_SPEED_RATIO * free_speed).astype(float)))
    queued = queued_slots(arrays["truth_speed"], arrays["eligible"], np.array([thresholds[link] for link in link_ids]))
    return candidates(queued)


def read_ramp_validity(release: Release, panel: str, link_ids: list[str]) -> np.ndarray:
    return ramp_validity(release, panel, "train", [day.date for day in mainline_days(release, panel)], link_ids)


def read_ramp_flows(release: Release, panel: str, link_ids: list[str], dates: list[str]) -> dict:
    attachment = pd.read_csv(release.network_file(panel, "ramp_attachment_map.csv"), dtype=str)
    attachment = attachment.drop_duplicates("ramp_link_id").sort_values("ramp_link_id", ignore_index=True)
    ramp_of = {ramp: i for i, ramp in enumerate(attachment["ramp_link_id"])}
    link_of = {link: i for i, link in enumerate(link_ids)}
    day_of = {day: i for i, day in enumerate(dates)}
    flows = np.full((len(dates) * SLOTS_PER_DAY, len(ramp_of)), np.nan, dtype=np.float32)
    for split in SPLITS:
        for file in release.ramp_files(panel, split):
            frame = pd.read_parquet(file, columns=RAMP_COLUMNS)
            value = pd.to_numeric(frame["flow_vph"], errors="coerce")
            ramp = frame["ramp_link_id"].astype(str).map(ramp_of)
            usable = (
                frame["is_score_eligible"].astype(bool)
                & value.notna()
                & ramp.notna()
                & (pd.to_numeric(frame["pct_observed"], errors="coerce") >= MIN_PCT_OBSERVED)
            ).to_numpy()
            timestamp = frame["timestamp"].astype(str)[usable]
            rows = (
                day_of[date_of(file)] * SLOTS_PER_DAY
                + timestamp.str[11:13].astype(int).to_numpy() * 12
                + timestamp.str[14:16].astype(int).to_numpy() // 5
            )
            flows[rows, ramp[usable].to_numpy(dtype=np.int64)] = value[usable].to_numpy(dtype=np.float32)
    return dict(
        is_on=attachment["ramp_type"].str.upper().isin(ON_TYPES).to_numpy(),
        link=attachment["nearest_mainline_link_id"].map(link_of).fillna(-1).to_numpy(dtype=np.int64),
        flows=flows,
    )


@dataclass
class HistoryTables:
    speed: torch.Tensor
    flow: torch.Tensor
    square: torch.Tensor
    count: torch.Tensor
    link_speed: torch.Tensor
    link_flow: torch.Tensor
    link_count: torch.Tensor
    fit_day: torch.Tensor
    day_speed: torch.Tensor
    day_flow: torch.Tensor
    day_count: torch.Tensor


@dataclass
class RampTables:
    flows: np.ndarray
    is_on: torch.Tensor
    panel: torch.Tensor
    column_on: torch.Tensor
    column_off: torch.Tensor
    valid_share: float
    z: torch.Tensor | None = None
    level: torch.Tensor | None = None


class GridStore:
    def __init__(self, grids: list[PanelGrid], codes: list[int], dates: list[str], n_train_days: int, runtime: Runtime):
        self.grids, self.codes, self.dates, self.n_train_days = grids, codes, dates, n_train_days
        self.device, self.dtype = runtime.device, runtime.dtype
        self.offsets = np.concatenate([[0], np.cumsum([grid.width for grid in grids])]).astype(int).tolist()
        self.slots, self.columns = len(dates) * SLOTS_PER_DAY, self.offsets[-1]
        self.weekslot0 = date.fromisoformat(dates[0]).weekday() * SLOTS_PER_DAY

        def filled(value, dtype) -> torch.Tensor:
            return torch.full((self.slots, self.columns), value, dtype=dtype, device=self.device)

        self.truth_speed, self.truth_flow = filled(torch.nan, self.dtype), filled(torch.nan, self.dtype)
        self.masked_speed, self.masked_flow = filled(torch.nan, self.dtype), filled(torch.nan, self.dtype)
        self.pct = filled(0, torch.uint8)
        self.eligible = filled(False, torch.bool)

        def per_column(values, dtype) -> torch.Tensor:
            return torch.as_tensor(np.concatenate(values, axis=-1), dtype=dtype, device=self.device)

        self.static = per_column([grid.static for grid in grids], self.dtype)
        self.lanes = per_column([grid.lanes for grid in grids], torch.float64)
        self.length = per_column([np.ones(grid.width) if grid.length_km is None else grid.length_km for grid in grids], torch.float64)
        self.links = per_column([np.full(grid.width, int((grid.link >= 0).sum())) for grid in grids], self.dtype)
        self.panel = per_column([np.full(grid.width, code) for grid, code in zip(grids, codes)], torch.int64)
        self.position = per_column([np.full(grid.width, i) for i, grid in enumerate(grids)], torch.int64)
        self.tables: HistoryTables | None = None
        self.ramps: RampTables | None = None
        self.months = sorted({day[:7] for day in dates})
        self.month_of_day = torch.as_tensor([self.months.index(day[:7]) for day in dates], device=self.device)
        self.level: torch.Tensor | None = None
        self.ramp_ok: torch.Tensor | None = None
        self.candidates: list[dict[str, np.ndarray] | None] = [None] * len(grids)
        self.ramp_valid: list[np.ndarray] = []

    def panel_columns(self, position: int) -> slice:
        return slice(self.offsets[position], self.offsets[position + 1])

    def set_candidates(self, allowed: dict[int, dict[str, np.ndarray]]) -> None:
        for position, by_kind in allowed.items():
            self.candidates[position] = {kind: np.flatnonzero(by_kind[kind]) for kind in KINDS}

    def ramp_rows(self, rows: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((len(rows), self.columns), dtype=torch.bool, device=self.device)
        index = rows.cpu().numpy()
        for position, grid in enumerate(self.grids):
            valid = self.ramp_valid[position][index][:, grid.link[grid.link >= 0]]
            columns = torch.as_tensor(self.offsets[position] + np.flatnonzero(grid.link >= 0), device=self.device)
            out[:, columns] = torch.as_tensor(valid, device=self.device)
        return out

    def set_ramp_validity(self, valid: list[np.ndarray]) -> None:
        self.ramp_valid = valid
        self.ramp_ok = self.ramp_rows(torch.arange(self.n_train_days * SLOTS_PER_DAY, device=self.device))

    def set_ramps(self, ramp_flows: list[dict]) -> None:
        flows, is_on, panel, on_lists, off_lists = [], [], [], [[] for _ in range(self.columns)], [[] for _ in range(self.columns)]
        for position, (grid, ramps) in enumerate(zip(self.grids, ramp_flows)):
            first = sum(part.shape[1] for part in flows)
            column_of_link = self.offsets[position] + np.flatnonzero(grid.link >= 0)[np.argsort(grid.link[grid.link >= 0])]
            for ramp, (link, on) in enumerate(zip(ramps["link"], ramps["is_on"])):
                if link >= 0:
                    (on_lists if on else off_lists)[column_of_link[link]].append(first + ramp)
            flows.append(ramps["flows"]), is_on.append(ramps["is_on"])
            panel.append(np.full(len(ramps["is_on"]), position))

        def padded(lists: list[list[int]]) -> torch.Tensor:
            table = np.full((self.columns, max(1, max(len(ramps) for ramps in lists))), -1, dtype=np.int64)
            for column, ramps in enumerate(lists):
                table[column, : len(ramps)] = ramps
            return torch.as_tensor(table, device=self.device)

        flows = np.concatenate(flows, axis=1)
        train = self.n_train_days * SLOTS_PER_DAY
        self.ramps = RampTables(
            flows,
            torch.as_tensor(np.concatenate(is_on), device=self.device),
            torch.as_tensor(np.concatenate(panel), device=self.device),
            padded(on_lists),
            padded(off_lists),
            float(np.isfinite(flows[:train]).mean()),
        )

    def _ramps_by_month(self, config: RampConfig) -> torch.Tensor:
        n_days = len(self.dates)
        weekday = np.array([date.fromisoformat(day).weekday() for day in self.dates])
        month = np.array([day[:7] for day in self.dates])
        by_day = self.ramps.flows.reshape(n_days, SLOTS_PER_DAY, -1)
        n_ramps, hours = by_day.shape[2], SLOTS_PER_DAY // 12
        z = torch.full((n_days, SLOTS_PER_DAY, n_ramps), torch.nan, dtype=torch.float32, device=self.device)
        for name in sorted(set(month)):
            days = np.flatnonzero(month == name)
            flow = torch.as_tensor(by_day[days], dtype=torch.float64, device=self.device)
            seen = torch.isfinite(flow)
            value = torch.where(seen, flow, 0.0)
            others = seen.sum(dim=0) - seen.to(torch.int64)
            mean = (value.sum(dim=0) - value) / others.clamp(min=1)
            left = torch.where(seen & (others >= config.min_days), flow - mean, torch.nan)
            for day_of_week in range(7):
                same = torch.as_tensor(weekday[days] == day_of_week, device=self.device)
                if not bool(same.any()):
                    continue
                part = left[same].view(-1, hours, 12, n_ramps)
                valid = torch.isfinite(part)
                total, count = torch.where(valid, part, 0.0).sum(dim=2), valid.sum(dim=2)
                other_count = count.sum(dim=0) - count
                offset = torch.where(
                    other_count >= config.min_offset_cells, (total.sum(dim=0) - total) / other_count.clamp(min=1), torch.nan
                )
                left[same] = (part - offset[:, :, None, :]).view(-1, SLOTS_PER_DAY, n_ramps)
            by_hour = left.view(len(days), hours, 12, n_ramps)
            valid = torch.isfinite(by_hour)
            n = valid.sum(dim=(0, 2)).clamp(min=1)
            centre = torch.where(valid, by_hour, 0.0).sum(dim=(0, 2)) / n
            spread = (torch.where(valid, (by_hour - centre[None, :, None, :]) ** 2, 0.0).sum(dim=(0, 2)) / n).sqrt()
            scaled = by_hour / spread.clamp(min=config.scale_floor)[None, :, None, :]
            z[torch.as_tensor(days, device=self.device)] = (
                scaled.clamp(-config.clip, config.clip).view(len(days), SLOTS_PER_DAY, n_ramps).float()
            )
        return z

    def standardize_ramps(self, config: RampConfig) -> None:
        ramps = self.ramps
        ramps.z = self._ramps_by_month(config).view(self.slots, -1)

        valid = torch.isfinite(ramps.z)
        filled = torch.where(valid, ramps.z, 0.0)
        level = torch.zeros((self.slots, len(self.grids), 3), dtype=self.dtype, device=self.device)
        for position in range(len(self.grids)):
            mine = ramps.panel == position
            for k, kind in enumerate((mine & ramps.is_on, mine & ~ramps.is_on)):
                n = valid[:, kind].sum(dim=1)
                level[:, position, k] = (filled[:, kind].sum(dim=1) / n.clamp(min=1)).to(self.dtype)
            level[:, position, 2] = (valid[:, mine].sum(dim=1) / max(int(mine.sum()), 1)).to(self.dtype)
        ramps.level = level

    def place(self, position: int, arrays: dict) -> None:
        grid = self.grids[position]
        columns = torch.as_tensor(self.offsets[position] + np.flatnonzero(grid.link >= 0), device=self.device)
        links = grid.link[grid.link >= 0]
        lanes = torch.as_tensor(grid.lanes[grid.link >= 0], dtype=self.dtype, device=self.device)
        train = self.n_train_days * SLOTS_PER_DAY

        def on_device(name: str, dtype) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(arrays[name][:, links]), dtype=dtype, device=self.device)

        self.masked_speed[:, columns] = on_device("masked_speed", self.dtype)
        self.masked_flow[:, columns] = on_device("masked_flow", self.dtype) / lanes
        self.pct[:, columns] = on_device("pct_observed", torch.uint8)
        self.truth_speed[:train, columns] = on_device("truth_speed", self.dtype)
        self.truth_flow[:train, columns] = on_device("truth_flow", self.dtype) / lanes
        self.eligible[:train, columns] = on_device("eligible", torch.bool)

    def day_index(self, days) -> list[int]:
        position = {day: i for i, day in enumerate(self.dates)}
        return [position[day] for day in days]

    def set_history(self, profiles: list[dict], fit_days: list[str]) -> None:
        shape = (SLOTS_PER_WEEK, self.columns)
        speed, flow, square, count = (torch.zeros(shape, dtype=torch.float64, device=self.device) for _ in range(4))
        for position, (grid, profile) in enumerate(zip(self.grids, profiles)):
            columns = torch.as_tensor(self.offsets[position] + np.flatnonzero(grid.link >= 0), device=self.device)
            links = grid.link[grid.link >= 0]
            lanes = torch.as_tensor(grid.lanes[grid.link >= 0], device=self.device)

            def on_device(name: str) -> torch.Tensor:
                return torch.as_tensor(np.ascontiguousarray(profile[name][links].T), device=self.device)

            speed[:, columns] = on_device("total_speed_kmh")
            flow[:, columns] = on_device("total_flow_vph") / lanes
            square[:, columns] = on_device("speed_square_total")
            count[:, columns] = on_device("count")

        fit_day = torch.zeros(len(self.dates), dtype=torch.bool, device=self.device)
        fit_day[self.day_index(fit_days)] = True
        day_shape = (len(self.dates), self.columns)
        day_speed, day_flow, day_count = (torch.zeros(day_shape, dtype=torch.float64, device=self.device) for _ in range(3))
        for start in range(0, self.n_train_days, HISTORY_CHUNK_DAYS):
            days = slice(start, min(start + HISTORY_CHUNK_DAYS, self.n_train_days))
            rows = slice(days.start * SLOTS_PER_DAY, days.stop * SLOTS_PER_DAY)
            by_day = (days.stop - days.start, SLOTS_PER_DAY, self.columns)
            used = self.eligible[rows].view(by_day) & fit_day[days, None, None]
            day_count[days] = used.sum(dim=1)
            day_speed[days] = torch.where(used, self.truth_speed[rows].view(by_day), 0.0).double().sum(dim=1)
            day_flow[days] = torch.where(used, self.truth_flow[rows].view(by_day), 0.0).double().sum(dim=1)
        self.tables = HistoryTables(
            speed, flow, square, count, speed.sum(dim=0), flow.sum(dim=0), count.sum(dim=0), fit_day, day_speed, day_flow, day_count
        )

    def weekslot(self, t: torch.Tensor) -> torch.Tensor:
        return (self.weekslot0 + t) % SLOTS_PER_WEEK

    def gather(self, values: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return values[t[:, :, None], c[:, None, :]]

    def history(self, t: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tables = self.tables
        inside = (t >= 0) & (t < self.slots)
        rows = t.clamp(0, self.slots - 1)
        day = rows // SLOTS_PER_DAY
        leave_out = inside & tables.fit_day[day]
        used = self.gather(self.eligible, rows, c) & leave_out[:, :, None]
        own_speed = torch.where(used, self.gather(self.truth_speed, rows, c), 0.0).double()
        own_flow = torch.where(used, self.gather(self.truth_flow, rows, c), 0.0).double()

        slot, column, of_day = self.weekslot(t)[:, :, None], c[:, None, :], day[:, :, None]
        count = tables.count[slot, column] - used.double()
        speed = tables.speed[slot, column] - own_speed
        flow = tables.flow[slot, column] - own_flow
        square = tables.square[slot, column] - own_speed**2
        left = leave_out[:, :, None].double()
        link_count = (tables.link_count[column] - tables.day_count[of_day, column] * left).clamp(min=1)
        link_speed = tables.link_speed[column] - tables.day_speed[of_day, column] * left
        link_flow = tables.link_flow[column] - tables.day_flow[of_day, column] * left

        n = count.clamp(min=1)
        variance = (square - speed**2 / n) / (count - 1).clamp(min=1)
        mean_speed = torch.where(count > 0, speed / n, link_speed / link_count)
        mean_flow = torch.where(count > 0, flow / n, link_flow / link_count)
        std_speed = torch.where(count >= 2, variance.clamp(min=0).sqrt(), 0.0)
        if self.level is not None:
            offset = self.level[self.month_of_day[day][:, :, None], column]
            mean_speed = mean_speed + torch.where(inside[:, :, None], offset, 0.0)
        return tuple(value.to(self.dtype) for value in (mean_speed, mean_flow, std_speed, count))


def history_mean_speed(store: GridStore, rows: torch.Tensor) -> torch.Tensor:
    tables = store.tables
    slot = store.weekslot(rows)
    count = tables.count[slot]
    fallback = (tables.link_speed / tables.link_count.clamp(min=1))[None].expand_as(count)
    return torch.where(count > 0, tables.speed[slot] / count.clamp(min=1), fallback)


def offsets(residual: torch.Tensor, published: torch.Tensor, config: LevelConfig) -> torch.Tensor:
    values = torch.where(published, residual, torch.nan).cpu()
    level = torch.nan_to_num(torch.nanmedian(values, dim=0).values, nan=0.0).to(residual.device)
    return torch.where(published.sum(dim=0) >= config.min_cells, level, 0.0).double()


def month_rows(store: GridStore, month: int) -> torch.Tensor:
    days = torch.nonzero(store.month_of_day == month).flatten()
    return (days[:, None] * SLOTS_PER_DAY + torch.arange(SLOTS_PER_DAY, device=store.device)[None]).flatten()


def month_levels(store: GridStore, config: LevelConfig) -> torch.Tensor:
    table = torch.zeros((len(store.months), store.columns), dtype=torch.float64, device=store.device)
    for month in range(len(store.months)):
        rows = month_rows(store, month)
        speed, flow = store.masked_speed[rows], store.masked_flow[rows]
        published = torch.isfinite(speed) & torch.isfinite(flow)
        table[month] = offsets(speed.double() - history_mean_speed(store, rows), published, config)
    return table
