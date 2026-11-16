"""配布データの読み込み：フォルダ内のパス、道路網、本線の観測、タスクのファイル、パネルの道路。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from core.config import REGIMES


@dataclass(frozen=True)
class Release:
    root: Path

    def __post_init__(self) -> None:
        if not (self.root / "config" / "corridors.json").exists():
            raise FileNotFoundError(f"{self.root} is not the unpacked release (no config/corridors.json)")

    def panels(self) -> list[str]:
        return [p["corridor_id"] for p in self._manifest()["panels"]]

    def family_of(self) -> dict[str, str]:
        return {p["corridor_id"]: p["family_id"] for p in self._manifest()["panels"]}

    def _manifest(self) -> dict:
        return json.loads((self.root / "config" / "corridors.json").read_text(encoding="utf-8"))

    def network_file(self, panel: str, name: str) -> Path:
        return self.root / "corridors" / panel / "network" / name

    def mainline_files(self, panel: str) -> list[Path]:
        return sorted((self.root / "corridors" / panel / "train" / "mainline_states").glob("*/*.parquet"))

    def masked_files(self, panel: str, split: str) -> dict[str, list[Path]]:
        root = self.root / "corridors" / panel / split / "mainline_states_masked"
        return {regime: sorted((root / f"mask_regime={regime}").glob("*.parquet")) for regime in REGIMES}

    def ramp_files(self, panel: str, split: str) -> list[Path]:
        return sorted((self.root / "corridors" / panel / split / "ramp_states").glob("**/*.parquet"))

    def task1_template(self, panel: str, split: str) -> Path:
        return self.root / "task1" / panel / split / "sample_submission_state.csv"

    def task2_dirs(self, split: str) -> list[Path]:
        return sorted(p / split for p in (self.root / "task2").iterdir() if (p / split).is_dir())

    def task4_file(self, panel: str, split: str, name: str) -> Path:
        return self.root / "task4" / panel / split / name


@lru_cache(maxsize=None)
def mileposts(data_root, panel):
    path = sorted((Path(data_root) / "corridors" / panel / "train" / "mainline_states").glob("*/*.parquet"))[0]
    frame = pd.read_parquet(path, columns=["link_id", "milepost"]).drop_duplicates("link_id")
    return dict(zip(frame.link_id.astype(str), frame.milepost.astype(float)))


class Road:
    def __init__(self, data_root, panel, ids):
        net = Path(data_root) / "corridors" / panel / "network"
        mp = mileposts(str(data_root), panel)
        missing = [link for link in ids if link not in mp]
        if missing:
            raise ValueError(f"{panel}: {len(missing)} links have no milepost in mainline_states")
        length = pd.read_csv(net / "links.csv").set_index("link_id").length_km.reindex(ids).to_numpy(np.float64)
        rank = pd.read_csv(net / "lwr_mainline_topology.csv").set_index("mainline_link_id").order_index.reindex(ids).to_numpy()
        self.ids, self.start = list(ids), np.array([mp[link] for link in ids])
        self.length, self.end = length, self.start + length
        self.order = np.lexsort((rank, self.start))
        if np.any(np.diff(rank[self.order]) < 0):
            raise ValueError(f"{panel}: milepost and order_index disagree on the road order")
        on_path = {link for seq in pd.read_csv(net / "path_set.csv").link_seq for link in seq.split(";")}
        self.mainline = np.array([link in on_path for link in ids])
        self.main_order = self.order[self.mainline[self.order]]
        centre, main = (self.start + self.end) / 2, np.flatnonzero(self.mainline)
        self.anchor = np.arange(len(ids))
        for b in np.flatnonzero(~self.mainline):
            self.anchor[b] = main[np.argmin(np.abs(centre[main] - centre[b]))]

    def chain(self):
        return {self.ids[a]: {self.ids[b]} for a, b in zip(self.main_order[:-1], self.main_order[1:])}

    def feeds(self):
        return {self.ids[b]: self.ids[self.anchor[b]] for b in np.flatnonzero(~self.mainline)}


def attach(found, anchor, bypass, max_hops):
    out = list(found)
    for b in np.flatnonzero(bypass):
        a = anchor[b]
        out[b] = [(a, 1)] + [(i, h + 1) for i, h in found[a] if h + 1 <= max_hops]
    return out


def _read(release: Release, panel: str, name: str, text_columns: list[str]) -> pd.DataFrame:
    path = release.network_file(panel, name)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={c: str for c in text_columns})


def read_links(release: Release, panel: str) -> pd.DataFrame:
    return _read(release, panel, "links.csv", ["link_id"])


def read_fd_parameters(release: Release, panel: str) -> pd.DataFrame:
    return _read(release, panel, "fd_parameters.csv", ["link_id"])


def read_path_set(release: Release, panel: str) -> pd.DataFrame:
    return _read(release, panel, "path_set.csv", ["path_id", "origin_zone", "destination_zone"])


def read_path_link_incidence(release: Release, panel: str) -> pd.DataFrame:
    return _read(release, panel, "path_link_incidence.csv", ["link_id", "path_id"])


def read_topology(release: Release, panel: str) -> pd.DataFrame:
    id_columns = ["link_id", "incoming_link_ids", "outgoing_link_ids", "on_ramp_link_ids", "off_ramp_link_ids"]
    return _read(release, panel, "lwr_mainline_topology.csv", id_columns).fillna({c: "" for c in id_columns})


TEXT_COLUMNS = ("date", "timestamp", "station_id", "link_id", "ramp_link_id", "ramp_type", "mask_regime")
_DATE_IN_NAME = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


@dataclass(frozen=True)
class DayFile:
    date: str
    regime: str | None
    path: Path


def date_of(path: Path) -> str:
    match = _DATE_IN_NAME.search(path.stem)
    if match is None:
        raise ValueError(f"no date in file name {path.name}")
    return "-".join(match.groups())


def read_day(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    for column in columns:
        if column in TEXT_COLUMNS:
            frame[column] = frame[column].astype(str)
    return frame


def mainline_days(release: Release, panel: str) -> list[DayFile]:
    return [DayFile(date_of(p), None, p) for p in release.mainline_files(panel)]


def masked_days(release: Release, panel: str, split: str) -> list[DayFile]:
    days = [DayFile(date_of(p), regime, p) for regime, paths in release.masked_files(panel, split).items() for p in paths]
    return sorted(days, key=lambda d: d.date)


STATE_KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
STATE_COLUMNS = STATE_KEYS + ["speed_kmh", "flow_vph"]


def read_state_template(release: Release, panel: str, split: str) -> pd.DataFrame:
    path = release.task1_template(panel, split)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pacsv.read_csv(
        path, convert_options=pacsv.ConvertOptions(include_columns=STATE_KEYS, column_types={c: pa.string() for c in STATE_KEYS})
    )
    return table.to_pandas()


def _concat_task2(release: Release, split: str, name: str, reader) -> pd.DataFrame:
    parts = [reader(d / name) for d in release.task2_dirs(split)]
    if not parts:
        raise FileNotFoundError(f"no Task 2 {name} for split {split}")
    return pd.concat(parts, ignore_index=True)


def read_window_index(release: Release, split: str) -> pd.DataFrame:
    return _concat_task2(release, split, "window_index.csv", lambda p: pd.read_csv(p, dtype=str))


def read_link_counts(release: Release, panel: str, split: str) -> pd.Series:
    path = release.task4_file(panel, split, "synthetic_link_counts.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"link_id": str}).set_index("link_id")["count"]


def read_weak_prior(release: Release, panel: str, split: str) -> pd.DataFrame:
    path = release.task4_file(panel, split, "synthetic_weak_prior.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    prior = pd.read_csv(path, dtype={c: str for c in ("panel", "departure_time", "path_id", "origin_zone", "destination_zone")})
    if prior.empty:
        raise ValueError(f"{path} is empty; departure_time is undefined")
    return prior
