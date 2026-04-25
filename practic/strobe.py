from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass(slots=True)
class StrobeWindow:
    start_ms: float
    end_ms: float


def build_strobe_windows(relative_offsets_ms: Sequence[float], tolerance_ms: float) -> list[StrobeWindow]:
    """Builds a series of strobes around reference offsets."""

    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")

    half = tolerance_ms / 2.0
    return [StrobeWindow(max(0.0, offset - half), offset + half) for offset in relative_offsets_ms]


def tile_strobe_windows(
    base_windows: Sequence[StrobeWindow],
    period_ms: float,
    repeats: int,
    start_time_ms: float = 0.0,
) -> list[StrobeWindow]:
    """Replicates strobes for each repeated packet in time."""

    tiled: list[StrobeWindow] = []
    for index in range(repeats):
        shift = start_time_ms + index * period_ms
        for window in base_windows:
            tiled.append(StrobeWindow(window.start_ms + shift, window.end_ms + shift))
    return tiled


def count_hits_in_strobes(times_ms: Sequence[float], strobes: Sequence[StrobeWindow]) -> Tuple[int, int]:
    """Counts how many timestamps fall inside the strobe collection."""

    hits = 0
    misses = 0
    for t in times_ms:
        if any(window.start_ms <= t <= window.end_ms for window in strobes):
            hits += 1
        else:
            misses += 1
    return hits, misses


def plot_strobes(
    times_ms: Sequence[float],
    strobes: Sequence[StrobeWindow],
    duration_ms: float | None = None,
) -> plt.Figure:
    """Illustrates the impulse timeline with strobes overlaid."""

    fig, ax = plt.subplots(figsize=(12, 4))
    max_time = duration_ms if duration_ms is not None else (times_ms[-1] if times_ms else 0.0)
    ax.vlines(times_ms, 0, 1, color="tab:blue", linewidth=1.2, label="Impulses")
    for strobe in strobes:
        if strobe.end_ms < 0:
            continue
        ax.axvspan(strobe.start_ms, strobe.end_ms, color="tab:green", alpha=0.25)

    ax.set_xlim(0, max_time)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Marker")
    ax.set_title("Strobed impulse timeline")
    ax.legend()
    return fig
