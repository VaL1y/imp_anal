from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class ShiftSumConfig:
    """Parameters for the shift-and-sum period search."""

    min_lag: int
    max_lag: int
    tolerance: int = 0
    top_exclusion_radius: int = 3


@dataclass(slots=True)
class ShiftSumResult:
    estimated_lag: int
    score_curve: np.ndarray
    lag_values: np.ndarray
    best_score: float
    confidence_ratio: float
    tolerance: int


def _normalize(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    energy = np.linalg.norm(centered)
    return centered / energy if energy > 0 else centered


def _lag_score(signal: np.ndarray, lag: int, tolerance: int) -> float:
    if lag <= 0 or lag >= len(signal):
        return float("-inf")

    total = 0.0
    count = 0
    for offset in range(-tolerance, tolerance + 1):
        candidate_lag = lag + offset
        if candidate_lag <= 0 or candidate_lag >= len(signal):
            continue
        left = signal[:-candidate_lag]
        right = signal[candidate_lag:]
        total += float(np.dot(left, right))
        count += 1

    if count == 0:
        return float("-inf")
    return total / count


def search_period_by_shift_sum(
    signal: np.ndarray,
    config: ShiftSumConfig,
) -> ShiftSumResult:
    """Performs the shift-and-sum search across candidate lags."""

    if config.min_lag <= 0 or config.max_lag <= config.min_lag:
        raise ValueError("Invalid lag range")
    if config.tolerance < 0:
        raise ValueError("Tolerance must be non-negative")

    normalized = _normalize(signal)
    lag_values = np.arange(config.min_lag, config.max_lag + 1)
    score_curve = np.array(
        [_lag_score(normalized, lag, config.tolerance) for lag in lag_values],
        dtype=float,
    )

    best_idx = int(np.nanargmax(score_curve))
    best_score = float(score_curve[best_idx])
    estimated_lag = int(lag_values[best_idx])

    mask = np.ones_like(score_curve, dtype=bool)
    left = max(0, best_idx - config.top_exclusion_radius)
    right = min(len(score_curve), best_idx + config.top_exclusion_radius + 1)
    mask[left:right] = False

    neighbor_scores = score_curve[mask]
    second_best = float(np.nanmax(neighbor_scores)) if neighbor_scores.size else float("-inf")
    confidence_ratio = float(best_score / second_best) if second_best not in (0.0, float("-inf")) else float("inf")

    return ShiftSumResult(
        estimated_lag=estimated_lag,
        score_curve=score_curve,
        lag_values=lag_values,
        best_score=best_score,
        confidence_ratio=confidence_ratio,
        tolerance=config.tolerance,
    )
