from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from practic.analyzer_shift_sum import ShiftSumConfig, search_period_by_shift_sum
from practic.generator_base import (
    JitterConfig,
    PacketGeneratorConfig,
    build_random_packet,
    jittered_times,
    pulse_offsets_in_time,
    rasterize_times,
    repeat_packet,
)
from practic.strobe import (
    build_strobe_windows,
    count_hits_in_strobes,
    plot_strobes,
    tile_strobe_windows,
)


@dataclass(slots=True)
class PracticeRunConfig:
    packet_length: int = 400
    repeats: int = 6
    impulses_per_packet: int = 18
    impulse_width: int = 6
    amplitude_range: tuple[float, float] = (1.0, 1.0)
    baseline: float = 0.1
    resolution_ms: float = 0.5
    jitter_sigma_ms: float = 1.5
    jitter_seed: int | None = 456


def describe_packet(pulses: Sequence, config: PracticeRunConfig) -> None:
    print("Stage 1 — packet generation:")
    print(f"  packet length: {config.packet_length} samples")
    print(f"  impulses per packet: {config.impulses_per_packet}")
    print("  repeating packet to build a longer signal")


def plot_signal(
    signal: np.ndarray,
    title: str,
    packet_length: int | None = None,
    repeats: int | None = None,
    resolution_ms: float = 1.0,
    jittered_times: Sequence[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, label="signal")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    if packet_length and repeats:
        for boundary in range(1, repeats):
            ax.axvline(boundary * packet_length, color="tab:gray", linestyle="--", alpha=0.55)
    if jittered_times is not None:
        sample_positions = [t / resolution_ms for t in jittered_times]
        ax.scatter(sample_positions, [1.05] * len(sample_positions), s=8, color="tab:red", alpha=0.5, label="jittered hits")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_lag_scores(lags: Sequence[int], scores: Sequence[float], title: str, highlight: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(lags, scores, label="shift-sum", color="tab:blue")
    ax.axvline(highlight, color="tab:orange", linestyle=":", label=f"estimate={highlight}")
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_shift_sum_comparison(
    first_result,
    second_result,
    title: str,
    label_first: str,
    label_second: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(first_result.lag_values, first_result.score_curve, label=label_first, alpha=0.8)
    ax.plot(second_result.lag_values, second_result.score_curve, label=label_second, alpha=0.8)
    ax.axvline(
        first_result.estimated_lag,
        color="tab:orange",
        linestyle=":",
        label=f"estimate {label_first}={first_result.estimated_lag}",
    )
    ax.axvline(
        second_result.estimated_lag,
        color="tab:green",
        linestyle=":",
        label=f"estimate {label_second}={second_result.estimated_lag}",
    )
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_jitter_histogram(base_times: Sequence[float], jittered_times: Sequence[float]) -> None:
    diffs = np.array(jittered_times) - np.array(base_times)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(diffs, bins=32, color="tab:purple", alpha=0.75)
    ax.set_title("Jitter histogram")
    ax.set_xlabel("Time offset (ms)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    mean = np.mean(diffs)
    std = np.std(diffs)
    ax.annotate(f"mean={mean:.2f} ms\nstd={std:.2f} ms", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")
    plt.tight_layout()
    plt.show()


def classify_hits(times_ms: Sequence[float], strobes) -> tuple[list[float], list[float]]:
    hits: list[float] = []
    misses: list[float] = []
    for t in times_ms:
        inside = any(window.start_ms <= t <= window.end_ms for window in strobes)
        if inside:
            hits.append(t)
        else:
            misses.append(t)
    return hits, misses


def plot_hit_miss_timeline(
    hits: Sequence[float],
    misses: Sequence[float],
    strobes,
    duration_ms: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if hits:
        ax.vlines(hits, 0.6, 1.0, color="tab:green", linewidth=1.5, label="hits")
    if misses:
        ax.vlines(misses, 0.2, 0.5, color="tab:red", linewidth=1.2, label="misses")
    for strobe in strobes:
        ax.axvspan(strobe.start_ms, strobe.end_ms, color="tab:gray", alpha=0.2)
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Strobe hits (green) vs. misses (red)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_hit_summary(hits: Sequence[float], misses: Sequence[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["hits", "misses"], [len(hits), len(misses)], color=["tab:green", "tab:red"], alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Strobe hit/miss summary")
    for i, value in enumerate([len(hits), len(misses)]):
        ax.text(i, value + max(1, value) * 0.05, str(value), ha="center")
    plt.tight_layout()
    plt.show()


def run_practice_session(config: PracticeRunConfig) -> None:
    packet_config = PacketGeneratorConfig(
        packet_length=config.packet_length,
        impulses_per_packet=config.impulses_per_packet,
        impulse_width=config.impulse_width,
        amplitude_range=config.amplitude_range,
        baseline=config.baseline,
        repeats=config.repeats,
        seed=config.jitter_seed,
    )

    packet, pulses = build_random_packet(packet_config)
    describe_packet(pulses, config)
    base_signal = repeat_packet(packet, packet_config.repeats)
    plot_signal(
        base_signal,
        "Stage 1 — repeating packet",
        packet_length=config.packet_length,
        repeats=config.repeats,
    )

    search_config = ShiftSumConfig(
        min_lag=config.packet_length - 100,
        max_lag=config.packet_length + 100,
        tolerance=0,
    )
    stage2_result = search_period_by_shift_sum(base_signal, search_config)
    print("Stage 2 — shift-and-sum (no tolerance)")
    print(f"  estimated lag: {stage2_result.estimated_lag}")
    plot_lag_scores(
        stage2_result.lag_values,
        stage2_result.score_curve,
        "Shift-and-sum scores (tight)",
        stage2_result.estimated_lag,
    )

    base_times = pulse_offsets_in_time(
        pulses,
        packet_config.repeats,
        packet_config.packet_length,
        config.resolution_ms,
    )
    jitter_config = JitterConfig(
        resolution_ms=config.resolution_ms,
        sigma_ms=config.jitter_sigma_ms,
        seed=config.jitter_seed,
    )
    jittered = jittered_times(base_times, jitter_config)
    jittered_signal = rasterize_times(
        jittered,
        config.resolution_ms,
        duration_ms=config.packet_length * config.repeats * config.resolution_ms,
    )
    plot_signal(
        jittered_signal,
        "Stage 3 — jittered timeline",
        packet_length=config.packet_length,
        repeats=config.repeats,
        resolution_ms=config.resolution_ms,
        jittered_times=jittered,
    )
    plot_jitter_histogram(base_times, jittered)

    tol_config = ShiftSumConfig(
        min_lag=config.packet_length - 100,
        max_lag=config.packet_length + 100,
        tolerance=2,
    )
    stage4_result = search_period_by_shift_sum(jittered_signal, tol_config)
    print("Stage 4 — tolerance-aware search")
    print(f"  applied tolerance: {tol_config.tolerance}")
    print(f"  estimated lag: {stage4_result.estimated_lag}")
    plot_lag_scores(
        stage4_result.lag_values,
        stage4_result.score_curve,
        "Shift-and-sum scores (tolerance)",
        stage4_result.estimated_lag,
    )
    plot_shift_sum_comparison(
        stage2_result,
        stage4_result,
        "Tolerance smooths the shift-sum peaks",
        "tight search",
        "tolerant search",
    )

    relative_offsets_ms = [pulse.start_index * config.resolution_ms for pulse in pulses]
    period_ms = packet_config.packet_length * config.resolution_ms
    strobe_windows = build_strobe_windows(relative_offsets_ms, tolerance_ms=3.0)
    strobes = tile_strobe_windows(strobe_windows, period_ms, config.repeats)
    hits, misses = classify_hits(jittered, strobes)
    total = len(hits) + len(misses)
    hit_rate = len(hits) / total if total else 0.0
    print("Stage 5 — strobed validation")
    print(f"  hits: {len(hits)}, misses: {len(misses)}, hit rate: {hit_rate:.2f}")
    print("  strobes confirm that most jittered arrivals still line up with the packet template")
    plot_hit_miss_timeline(hits, misses, strobes, duration_ms=period_ms * config.repeats)
    plot_hit_summary(hits, misses)


if __name__ == "__main__":
    cfg = PracticeRunConfig()
    run_practice_session(cfg)
