from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class PulseSpec:
    """Описание одного импульса внутри пачки."""

    start: int
    width: int
    amplitude: float = 1.0


@dataclass(slots=True)
class BurstConfig:
    """Параметры одной пачки и всей последовательности."""

    burst_length: int
    repeats: int
    noise_std: float = 0.0
    baseline: float = 0.0
    seed: int | None = 33


DEFAULT_PULSES: tuple[PulseSpec, ...] = (
    PulseSpec(start=40, width=10, amplitude=1.0),
    PulseSpec(start=90, width=14, amplitude=0.9),
    PulseSpec(start=140, width=11, amplitude=1.2),
    PulseSpec(start=210, width=16, amplitude=0.85),
    PulseSpec(start=280, width=13, amplitude=1.1),
    PulseSpec(start=345, width=10, amplitude=1.0),
    PulseSpec(start=405, width=15, amplitude=0.95),
    PulseSpec(start=470, width=12, amplitude=1.15),
    PulseSpec(start=535, width=18, amplitude=0.8),
    PulseSpec(start=610, width=10, amplitude=1.05),
    PulseSpec(start=675, width=14, amplitude=0.92),
    PulseSpec(start=740, width=12, amplitude=1.1),
)


def build_burst(config: BurstConfig, pulses: Iterable[PulseSpec] = DEFAULT_PULSES) -> np.ndarray:
    """Строит одну пачку сигнала заданной длины."""
    burst = np.full(config.burst_length, config.baseline, dtype=float)

    for pulse in pulses:
        start = max(0, pulse.start)
        end = min(config.burst_length, pulse.start + pulse.width)
        if start >= end:
            continue
        burst[start:end] = pulse.amplitude

    return burst


def build_repeated_signal(config: BurstConfig, pulses: Iterable[PulseSpec] = DEFAULT_PULSES) -> tuple[np.ndarray, np.ndarray]:
    """Повторяет пачку repeats раз и добавляет шум.

    Returns:
        signal: итоговая последовательность
        burst: исходная одна пачка без шума
    """
    rng = np.random.default_rng(config.seed)
    burst = build_burst(config, pulses)
    signal = np.tile(burst, config.repeats)

    if config.noise_std > 0:
        signal = signal + rng.normal(0.0, config.noise_std, size=signal.shape)

    return signal, burst