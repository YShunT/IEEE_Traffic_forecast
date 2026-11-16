"""全工程の実行：CPU と GPU の工程を並行に走らせ、最後に提出ファイルを作る。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from core.config import ROOT

CPU_PARTS = ("task4", "onset_place_gbm", "onset_signs_gbm", "ongoing_gbm")
GPU_PARTS = ("task1_3", "onset_place_dcrnn", "onset_signs_dcrnn", "ongoing_dcrnn", "ongoing_unet")
LAST_PARTS = ("onset_mix", "ongoing_ensemble", "submit")
THREADS = "4"
CHILD_ENV = dict(
    OMP_NUM_THREADS=THREADS,
    OPENBLAS_NUM_THREADS=THREADS,
    MKL_NUM_THREADS=THREADS,
    NUMEXPR_NUM_THREADS=THREADS,
    PYTHONDONTWRITEBYTECODE="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)


def say(message):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {message}", flush=True)


def log_dir(c, run_id):
    path = Path(c["work"]) / run_id / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(c, run_id, config):
    logs = log_dir(c, run_id)
    extra = ["--config", str(config)] + (["--trial"] if c["trial"] else [])
    last_parts = LAST_PARTS[:-1] if c["trial"] else LAST_PARTS
    children, cpu_status = [], {}

    def launch(part, stdout=None):
        cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--part", part, "--run-id", run_id, *extra]
        child = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env={**os.environ, **CHILD_ENV},
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout else None,
        )
        children.append(child)
        return child

    def cpu_chain():
        with open(logs / "stdout_cpu.log", "a") as f:
            for part in CPU_PARTS:
                cpu_status[part] = launch(part, f).wait()
                if cpu_status[part]:
                    return

    def stop(signum, _):
        say(f"received signal {signum}: stopping the parts")
        for child in children:
            if child.poll() is None:
                child.terminate()
        sys.exit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    say(f"run id {run_id}{' (trial)' if c['trial'] else ''} | logs: {logs}")
    cpu = threading.Thread(target=cpu_chain, daemon=True)
    cpu.start()
    failed = []
    for part in GPU_PARTS:
        say(f"{part}: started")
        status = launch(part).wait()
        if status:
            say(f"{part}: ended with status {status}")
            failed.append(part)
            break
        say(f"{part}: done")
    say("waiting for the CPU parts")
    cpu.join()
    failed += [p for p, s in cpu_status.items() if s]
    if failed or len(cpu_status) < len(CPU_PARTS):
        say(
            f"a part failed ({', '.join(failed) or 'cpu'}); continue with: python main.py --run-id {run_id}"
            + (" --trial" if c["trial"] else "")
        )
        sys.exit(1)
    for part in last_parts:
        if launch(part).wait():
            say(f"{part} failed")
            sys.exit(1)
        say(f"{part}: done")
    joined = Path(c["output"]) / "run.log"
    joined.write_text("".join((logs / f"{part}.log").read_text() for part in (*CPU_PARTS, *GPU_PARTS, *last_parts)))
    shutil.rmtree(c["work"], ignore_errors=True)
    say(f"done; the log is {joined}")
