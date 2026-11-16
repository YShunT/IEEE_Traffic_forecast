"""工程 onset_mix：2 つの onset モデル（place・signs）の確率を 0.75:0.25 で混ぜ、Task 2 onset の答えにする。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import work_path
from core.runtime import log, section

NAME = "onset_mix"
PLACE = ("onset_place_gbm", "onset_place_dcrnn")
SIGNS = ("onset_signs_gbm", "onset_signs_dcrnn")
WEIGHT_PLACE = 0.75
THRESHOLD = 0.5


def save_probs(c, run_id, name, windows, probs, ids):
    frames = [
        pd.DataFrame(dict(window_id=window_id, split=split, panel=panel, forecast_origin=origin, link_id=ids[panel], probability=p))
        for (window_id, split, panel, origin), p in zip(windows, probs)
    ]
    pd.concat(frames, ignore_index=True).to_parquet(work_path(c, run_id, f"official_{name}.parquet"), index=False, compression="zstd")


def read_probs(c, run_id, name):
    return pd.read_parquet(work_path(c, run_id, f"official_{name}.parquet"))


def binarize(p):
    answer = p >= THRESHOLD
    if not answer.any():
        answer[int(np.argmax(p))] = True
    return answer.astype(np.int8)


def mixed(c, run_id):
    frame = read_probs(c, run_id, PLACE[0])
    others = {m: read_probs(c, run_id, m).set_index(["window_id", "link_id"]).probability for m in PLACE[1:] + SIGNS}
    for window_id in frame.window_id.unique():
        g = frame[frame.window_id == window_id]
        keys = list(zip(g.window_id, g.link_id))
        probs = {PLACE[0]: g.probability.to_numpy(), **{m: others[m].loc[keys].to_numpy() for m in others}}
        place = 0.5 * probs[PLACE[0]] + 0.5 * probs[PLACE[1]]
        signs = 0.5 * probs[SIGNS[0]] + 0.5 * probs[SIGNS[1]]
        yield g, WEIGHT_PLACE * place + (1 - WEIGHT_PLACE) * signs


def horizon_rows(frame):
    origin = pd.to_datetime(frame.forecast_origin, utc=True)
    parts = []
    for step in range(1, 7):
        part = frame[["window_id", "split", "panel", "link_id"]].copy()
        part["step"] = step
        part["timestamp"] = (origin + pd.Timedelta(minutes=5 * step)).dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
        part["output"] = frame.output.to_numpy() if step == 6 else 0.0
        part["queue_pred"] = frame.pred.to_numpy() if step == 6 else np.int8(0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run(c, run_id):
    section("onset official")
    parts = []
    for g, p in mixed(c, run_id):
        parts.append(
            pd.DataFrame(
                dict(
                    window_id=g.window_id.to_numpy(),
                    split=g.split.to_numpy(),
                    panel=g.panel.to_numpy(),
                    link_id=g.link_id.to_numpy(),
                    forecast_origin=g.forecast_origin.to_numpy(),
                    output=p,
                    pred=binarize(p),
                )
            )
        )
    frame = horizon_rows(pd.concat(parts, ignore_index=True))
    if not c["trial"]:
        assert len(frame) == 87000 and not frame.loc[frame.step < 6, "queue_pred"].any()
    frame.to_parquet(work_path(c, run_id, f"official_{NAME}.parquet"), index=False, compression="zstd")
    answered = frame[frame.step == 6].groupby("window_id").queue_pred.sum()
    log(
        f"{NAME} | {WEIGHT_PLACE} x place + {1 - WEIGHT_PLACE:g} x signs | at least {THRESHOLD} | "
        f"{len(answered)} windows, {answered.mean():.2f} links per window"
    )
