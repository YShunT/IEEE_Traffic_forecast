"""工程 ongoing_ensemble：3 つのモデルの確率を平均し、渋滞の下流を 0 にする後処理をして 0.5 で切る。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import MODELS, work_path
from core.runtime import log, section
from task2.ongoing import common as SH
from task2.ongoing import common as W
from task2.ongoing.common import binarize, cell_table, cube, prepare, read_cells

NAME = "blend"


def post_process(c, p, rows, mean):
    post = c["post"]["downstream"]
    hist = W.Histories(c, p, rows, "official")
    cut = np.stack(
        [SH.downstream_of_queue(p, hist.window(k, True)[0]["speed_kmh"], post["margin_km"], post["reach_km"]) for k in range(len(rows))]
    )[..., None]
    return np.where(cut, 0.0, mean)


def run(c, run_id):
    section("ongoing ensemble")
    panels, _, official = prepare(c)
    cells = {m: read_cells(c, run_id, m) for m in MODELS}
    frames, changed = [], 0
    for n, p in panels.items():
        r = official[official.panel == n].reset_index(drop=True)
        raw = np.mean([cube(cells[m], p, r) for m in MODELS], 0)
        mean = post_process(c, p, r, raw)
        changed += int((binarize(raw) != binarize(mean)).sum())
        frames.append(cell_table(p, r, mean))
    pd.concat(frames, ignore_index=True).to_parquet(work_path(c, run_id, f"official_{NAME}.parquet"), index=False, compression="zstd")
    log(f"{NAME} | mean of {', '.join(MODELS)} | post-processing changed {changed} cells")
