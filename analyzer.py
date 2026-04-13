from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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