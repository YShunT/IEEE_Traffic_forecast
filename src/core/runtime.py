"""計算環境：デバイス・dtype・seed、GPU の決定論的な設定とメモリ上限、ログ。"""

from __future__ import annotations

import hashlib
import os
import resource
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    dtype: torch.dtype
    seed: int

    def scoring(self) -> "Runtime":
        return replace(self, dtype=torch.float64)

    def floats(self, values) -> torch.Tensor:
        return torch.as_tensor(np.array(values, dtype=np.float64), dtype=self.dtype, device=self.device)

    def index(self, values) -> torch.Tensor:
        return torch.as_tensor(np.array(values, dtype=np.int64), device=self.device)


def make_runtime(device: str, dtype: str, seed: int = 0) -> Runtime:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return Runtime(torch.device(device), DTYPES[dtype], seed)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return array.astype(np.float64) if array.dtype.kind == "f" else array


_file = None


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_log(path):
    global _file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _file = open(path, "a", buffering=1)


def log(message=""):
    now = datetime.now()
    for line in str(message).splitlines() or [""]:
        text = f"[{now:%m-%d %H:%M:%S}] {line}" if line else ""
        print(text, flush=True)
        if _file:
            _file.write(text + "\n")


def section(title):
    head = f"== [{title}] "
    log(head + "=" * max(4, 70 - len(head)))


def duration(seconds):
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def set_determinism(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False


def limit_memory(c, part):
    total = torch.cuda.get_device_properties(0).total_memory
    limit = min(c["gpu"]["memory_limit_gb"][part] * 2**30, total)
    torch.cuda.set_per_process_memory_fraction(limit / total)
    log(
        f"  [env]     GPU: {torch.cuda.get_device_name(0)} ({total / 2**30:.1f} GB) | up to {limit / 2**30:.1f} GB | torch {torch.__version__}"
    )


def use_gpu(c, part):
    torch.set_num_threads(c["threads"])
    set_determinism(0)
    limit_memory(c, part)
    return torch


def derive_seed(seed: int, *names) -> int:
    text = "|".join(str(part) for part in (seed, *names))
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def accumulate(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return target.index_put_((index,), values, accumulate=True)
