"""Task 1・3 の U-Net の設定（config.json の task1_3 節）を型つきで持つ。"""

from __future__ import annotations

import typing
from dataclasses import dataclass, is_dataclass


@dataclass
class RuntimeConfig:
    device: str
    seed: int
    dtype: str


@dataclass
class PlacedBlackoutConfig:
    count: int
    seed: int


@dataclass
class CVConfig:
    start: str
    end: str


@dataclass
class Extent:
    time: int
    space: int


@dataclass
class BlackoutCropConfig:
    probability: float
    min_context: int
    weight: float


@dataclass
class GapContextConfig:
    mean_slots: int
    max_distance: int


@dataclass
class RampConfig:
    scale_floor: float
    clip: float
    min_days: int
    min_offset_cells: int


@dataclass
class LevelConfig:
    min_cells: int


@dataclass
class UNetDataConfig:
    crop: Extent
    batch_size: int
    mask_rates: list[float]
    coverage_per_epoch: float
    blackout: BlackoutCropConfig
    gap_context: GapContextConfig
    ramps: RampConfig
    level: LevelConfig


@dataclass
class UNetModelConfig:
    channels: list[int]
    norm_groups: int
    time_blocks: int


@dataclass
class UNetOptimConfig:
    lr: float
    weight_decay: float
    warmup_epochs: int
    epochs: int
    grad_clip: float
    ema_decay: float


@dataclass
class UNetValidationConfig:
    holdout_fraction: float
    holdout_seed: int
    mask_rates: list[float]
    mask_seed: int
    blackout: PlacedBlackoutConfig


@dataclass
class UNetInferenceConfig:
    center: Extent
    day_context: int
    tile_batch: int


@dataclass
class LossConfig:
    state_weight: float
    lwr_weight: float
    density_weight: float
    decay: float = 0.99
    smoothing: float = 0.01


@dataclass
class UNetConfig:
    data: UNetDataConfig
    model: UNetModelConfig
    optim: UNetOptimConfig
    validation: UNetValidationConfig
    inference: UNetInferenceConfig
    loss: LossConfig


@dataclass
class ExperimentConfig:
    runtime: RuntimeConfig
    cv: CVConfig
    unet: UNetConfig


def build(cls, data: dict):
    hints = typing.get_type_hints(cls)
    return cls(**{key: build(hints[key], value) if is_dataclass(hints[key]) else value for key, value in data.items()})
