"""設定ファイルの読み込み、コンペの定数、出力先のパス。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATES = pd.date_range("2030-06-01", "2031-04-30", freq="D")
TRAIN_DAYS = 273
MONTH = np.array([d.strftime("%Y-%m") for d in DATES])
WEEKDAY = np.array([d.weekday() for d in DATES])
MONTH_START = np.array([(d.replace(day=1) - DATES[0]).days for d in DATES])
FIELDS = ["speed_kmh", "flow_vph", "occupancy"]
HISTORY_SLOTS, HORIZONS = 12, 6
STEPS = HISTORY_SLOTS + 1
SUBMISSION_ROWS = 6980503
MODELS = ["gbm", "dcrnn", "unet"]


def load(path, trial=False):
    c = json.loads(Path(path).read_text())
    c["trial"] = bool(trial)
    output = (ROOT / c["output"]).resolve() / ("trial" if trial else "")
    c["data"] = str((ROOT / c["data"]).resolve())
    c["output"] = str(output)
    c["work"] = str(output / "work")
    c["cache"] = str(output / "work" / "cache")
    if trial:
        t = c["trial_settings"]
        c["limit_windows"] = t["windows_per_panel"]
        c["gbm"]["rounds"], c["gbm"]["log_every"] = t["rounds"], t["rounds"]
        c["dcrnn"].update(epochs=t["epochs"], steps_per_epoch=t["steps_per_epoch"], teacher_epochs=1)
        c["unet"]["train"].update(epochs=t["epochs"], steps_per_epoch=t["steps_per_epoch"])
        for model in MODELS:
            c[model]["seeds"] = c[model]["seeds"][:1]
    return c


def out(c, *parts):
    path = Path(c["output"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def work_path(c, run_id, name):
    path = Path(c["work"]) / run_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


REGIMES = ("R1", "R2", "R3")
SPLITS = ("train", "validation", "private")
SUBMISSION_SPLITS = ("validation", "private")

MIN_PCT_OBSERVED = 75

SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

MIN_LANES = 1
MAX_LANES = 16

SPEED_NORMALIZER = 25.0
FLOW_NORMALIZER = 600.0
SPEED_WEIGHT = 0.54
FLOW_WEIGHT = 0.46

QUEUE_SPEED_RATIO = 0.60
DEFAULT_FREE_SPEED_KMH = 105.0

WINDOW_HISTORY_SLOTS = 13
WINDOW_HORIZON_SLOTS = 6
BLACKOUT_SLOTS = 18
WINDOW_MIN_SEPARATION = 72

MIN_SPEED_KMH = 1.0
DEFAULT_LENGTH_KM = 1.0
