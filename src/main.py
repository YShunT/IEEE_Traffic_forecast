"""提出 1 の入口：全工程、短い確認、1 工程だけの実行。"""

import argparse
import importlib
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import run as R
from core.config import ROOT, load
from core.runtime import log, new_run_id, open_log, peak_gb

PARTS = {
    "task1_3": "task1_3.part",
    "task4": "task4",
    "onset_place_gbm": "task2.onset_place.gbm",
    "onset_place_dcrnn": "task2.onset_place.dcrnn",
    "onset_signs_gbm": "task2.onset_signs.gbm",
    "onset_signs_dcrnn": "task2.onset_signs.dcrnn",
    "onset_mix": "task2.onset_mix",
    "ongoing_gbm": "task2.ongoing.gbm",
    "ongoing_dcrnn": "task2.ongoing.dcrnn",
    "ongoing_unet": "task2.ongoing.unet",
    "ongoing_ensemble": "task2.ongoing.ensemble",
    "submit": "submit",
}


def run_part(c, part, run_id):
    started = time.monotonic()
    open_log(R.log_dir(c, run_id) / f"{part}.log")
    importlib.import_module(PARTS[part]).run(c, run_id)
    vram = f" | VRAM peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GB" if torch.cuda.is_initialized() else ""
    log(f"{part} done in {(time.monotonic() - started) / 60:.1f} min | RSS peak {peak_gb():.1f} GB{vram}")


def main():
    ap = argparse.ArgumentParser(description="配布データから提出 1 のファイルを作る")
    ap.add_argument("--trial", action="store_true", help="窓・エポック・seed を減らした短い確認（提出ファイルなし）")
    ap.add_argument("--run-id", help="止まった run の続き（保存済みのモデルを使う）")
    ap.add_argument("--part", choices=list(PARTS), help="1 工程だけ実行する（--run-id が必要）")
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    args = ap.parse_args()
    c = load(args.config, args.trial)
    if args.part:
        if not args.run_id:
            ap.error("--part needs --run-id")
        run_part(c, args.part, args.run_id)
    else:
        R.run(c, args.run_id or new_run_id(), args.config)


if __name__ == "__main__":
    main()
