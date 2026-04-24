from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from generator import rasterize_impulse_times


@dataclass(slots=True)
class PeriodSearchResult:
    estimated_period: int
    score_curve: np.ndarray
    lag_values: np.ndarray
    best_score: float
    second_best_score: float
    confidence_ratio: float
    normalized_peak: float
    is_plausible: bool


@dataclass(slots=True)
class PeriodSearchConfig:
    min_period: int
    max_period: int
    top_exclusion_radius: int = 5
    min_confidence_ratio: float = 1.08
    min_normalized_peak: float = 0.10


@dataclass(slots=True)
class StrobeWindow:
    """Представление временного окна — «стробы» — вокруг отдельного импульса."""

    start_ms: float
    end_ms: float


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Убирает среднее и нормирует по энергии."""
    centered = signal - np.mean(signal)
    energy = np.linalg.norm(centered)
    if energy == 0:
        return centered
    return centered / energy


def shift_sum_score(signal: np.ndarray, lag: int) -> float:
    """Скалярная мера похожести сигнала и его копии со сдвигом.

    По сути это нормированное суммирование произведений со сдвигом.
    """
    if lag <= 0 or lag >= len(signal):
        return float("-inf")

    left = signal[:-lag]
    right = signal[lag:]
    return float(np.sum(left * right))


def search_period_by_shift_sum(
    raw_signal: np.ndarray,
    config: PeriodSearchConfig,
) -> PeriodSearchResult:
    """Ищет период по максимуму функции суммирования со сдвигом."""
    signal = _normalize_signal(raw_signal)

    lag_values = np.arange(config.min_period, config.max_period + 1)
    score_curve = np.array([shift_sum_score(signal, lag) for lag in lag_values], dtype=float)

    best_idx = int(np.argmax(score_curve))
    estimated_period = int(lag_values[best_idx])
    best_score = float(score_curve[best_idx])

    # Ищем второй максимум вне небольшой окрестности главного пика.
    mask = np.ones_like(score_curve, dtype=bool)
    left = max(0, best_idx - config.top_exclusion_radius)
    right = min(len(score_curve), best_idx + config.top_exclusion_radius + 1)
    mask[left:right] = False

    if np.any(mask):
        second_best_score = float(np.max(score_curve[mask]))
    else:
        second_best_score = float("-inf")

    if np.isfinite(second_best_score) and second_best_score != 0:
        confidence_ratio = best_score / second_best_score
    else:
        confidence_ratio = float("inf")

    score_mean = float(np.mean(score_curve))
    score_std = float(np.std(score_curve))
    normalized_peak = 0.0 if score_std == 0 else (best_score - score_mean) / score_std

    is_plausible = (
        confidence_ratio >= config.min_confidence_ratio
        and normalized_peak >= config.min_normalized_peak
    )

    return PeriodSearchResult(
        estimated_period=estimated_period,
        score_curve=score_curve,
        lag_values=lag_values,
        best_score=best_score,
        second_best_score=second_best_score,
        confidence_ratio=confidence_ratio,
        normalized_peak=normalized_peak,
        is_plausible=is_plausible,
    )


def build_strobe_windows(relative_offsets: Sequence[float], tolerance_ms: float) -> list[StrobeWindow]:
    """Строит стробы для одной пачки импульсов относительно её начала."""

    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")

    return [StrobeWindow(offset - tolerance_ms, offset + tolerance_ms) for offset in relative_offsets]


def tile_strobe_windows(
    base_windows: Sequence[StrobeWindow],
    period_ms: float,
    num_packs: int,
    start_time_ms: float = 0.0,
) -> list[StrobeWindow]:
    """Повторяет строб-окна для указанного числа пачек с заданным периодом."""

    tiled: list[StrobeWindow] = []
    for pack_index in range(num_packs):
        shift = start_time_ms + pack_index * period_ms
        for window in base_windows:
            tiled.append(StrobeWindow(window.start_ms + shift, window.end_ms + shift))
    return tiled


def extend_sequence_from_base(
    relative_offsets: Sequence[float],
    start_time_ms: float,
    num_packs: int,
    period_ms: float,
) -> list[float]:
    """Продолжает импульсы, повторяя базовый набор относительных меток."""

    if num_packs <= 0:
        return []

    continuation: list[float] = []
    for pack_index in range(num_packs):
        base_shift = start_time_ms + pack_index * period_ms
        for offset in relative_offsets:
            continuation.append(base_shift + offset)
    return continuation


def count_hits_in_strobes(times_ms: Sequence[float], strobes: Sequence[StrobeWindow]) -> tuple[int, int]:
    """Считает попадания импульсов в заданные стробы."""

    hits = 0
    misses = 0
    for t in times_ms:
        if any(strobe.start_ms <= t <= strobe.end_ms for strobe in strobes):
            hits += 1
        else:
            misses += 1
    return hits, misses


def analyze_impulse_times(
    times_ms: Sequence[float],
    config: PeriodSearchConfig,
    resolution_ms: float = 1.0,
) -> tuple[np.ndarray, PeriodSearchResult, dict[str, float | bool]]:
    signal = rasterize_impulse_times(times_ms, resolution_ms=resolution_ms)
    result = search_period_by_shift_sum(signal, config)
    validation = validate_period(signal, result.estimated_period)
    return signal, result, validation

def validate_period(raw_signal: np.ndarray, estimated_period: int) -> dict[str, float | bool]:
    """Набор простых проверок, что найденный период действительно разумный."""
    if estimated_period <= 0 or estimated_period >= len(raw_signal):
        return {
            "is_valid": False,
            "repeat_consistency": 0.0,
            "energy_ratio": 0.0,
        }

    signal = raw_signal - np.mean(raw_signal)
    first = signal[:-estimated_period]
    second = signal[estimated_period:]

    denom = np.linalg.norm(first) * np.linalg.norm(second)
    repeat_consistency = 0.0 if denom == 0 else float(np.dot(first, second) / denom)

    folded = signal[: len(signal) // estimated_period * estimated_period]
    if len(folded) == 0:
        return {
            "is_valid": False,
            "repeat_consistency": repeat_consistency,
            "energy_ratio": 0.0,
        }

    matrix = folded.reshape(-1, estimated_period)
    mean_template = np.mean(matrix, axis=0)
    template_energy = float(np.linalg.norm(mean_template))
    full_energy = float(np.linalg.norm(signal) / max(1, matrix.shape[0]))
    energy_ratio = 0.0 if full_energy == 0 else template_energy / full_energy

    is_valid = repeat_consistency > 0.7 and energy_ratio > 0.5
    return {
        "is_valid": is_valid,
        "repeat_consistency": repeat_consistency,
        "energy_ratio": energy_ratio,
    }


def plot_shift_sum_curve(
    result: PeriodSearchResult,
    true_period: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(result.lag_values, result.score_curve, label="Сумма со сдвигом", color="tab:blue")
    ax.axvline(
        result.estimated_period,
        linestyle=":",
        color="tab:orange",
        label=f"Найденный период = {result.estimated_period}",
    )
    if true_period is not None:
        ax.axvline(
            true_period,
            linestyle="--",
            color="tab:green",
            label=f"Истинный период = {true_period}",
        )

    ax.set_xlabel("Сдвиг")
    ax.set_ylabel("Скор")
    ax.set_title("Кривая поиска периода")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
