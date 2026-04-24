from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import random

import numpy as np


@dataclass(slots=True)
class PulseSpec:
    """Описание одного импульса внутри пачки."""

    start: int
    width: int
    amplitude: float = 1.0


@dataclass(slots=True)
class BurstConfig:
    """Параметры генерации цифровой пачки и всей последовательности."""

    burst_length: int
    repeats: int
    noise_std: float = 0.0
    baseline: float = 0.0
    seed: int | None = 33


@dataclass(slots=True)
class ImpulseTimingConfig:
    """Параметры для генерации временных меток пачек импульсов."""

    num_impulses_in_pack: int
    min_interval_ms: int
    max_interval_ms: int
    generation_duration_s: float
    noise_sigma_ms: float = 0.0
    seed: int | None = 42


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


def build_impulse_relative_times(
    num_impulses_in_pack: int,
    min_interval_ms: int,
    max_interval_ms: int,
    rng: random.Random,
) -> list[int]:
    """Возвращает относительные времена импульсов внутри пачки."""
    if num_impulses_in_pack < 2:
        return [0]

    rel_times: list[int] = [0]
    for _ in range(num_impulses_in_pack - 1):
        delta = rng.randint(min_interval_ms, max_interval_ms)
        rel_times.append(rel_times[-1] + delta)
    return rel_times


def generate_impulse_pack_times(config: ImpulseTimingConfig) -> tuple[list[float], list[int]]:
    """Генерирует метки времени импульсов с шумом по всей длительности."""
    rng = random.Random(config.seed) if config.seed is not None else random.Random()
    impulse_rel_times = build_impulse_relative_times(
        config.num_impulses_in_pack,
        config.min_interval_ms,
        config.max_interval_ms,
        rng,
    )

    if len(impulse_rel_times) < 2:
        inter_pack_distance_ms = 0
    else:
        inter_pack_distance_ms = impulse_rel_times[1]

    all_times: list[float] = []
    current_pack_start = 0.0
    max_time = config.generation_duration_s * 1000.0

    while current_pack_start <= max_time:
        for rel_time in impulse_rel_times:
            noise = rng.gauss(0.0, config.noise_sigma_ms) if config.noise_sigma_ms > 0 else 0.0
            all_times.append(current_pack_start + rel_time + noise)
        if inter_pack_distance_ms == 0 and len(impulse_rel_times) > 1:
            inter_pack_distance_ms = impulse_rel_times[1]
        pack_end = current_pack_start + (impulse_rel_times[-1] if impulse_rel_times else 0)
        current_pack_start = pack_end + inter_pack_distance_ms

    return all_times, impulse_rel_times


def rasterize_impulse_times(
    times_ms: Sequence[float],
    resolution_ms: float = 1.0,
    max_time_ms: float | None = None,
) -> np.ndarray:
    """Конвертирует список меток времени в цифровой сигнал с заданным разрешением."""
    if resolution_ms <= 0:
        raise ValueError("resolution_ms must be positive")

    max_time = max_time_ms if max_time_ms is not None else (max(times_ms) if times_ms else 0.0)
    length = int(math.ceil(max_time / resolution_ms)) + 1
    signal = np.zeros(length, dtype=float)

    for time_ms in times_ms:
        if time_ms < 0:
            continue
        idx = int(round(time_ms / resolution_ms))
        if 0 <= idx < length:
            signal[idx] += 1.0

    return signal
