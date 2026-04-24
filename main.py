from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from analyzer import (
    PeriodSearchConfig,
    StrobeWindow,
    analyze_impulse_times,
    build_strobe_windows,
    count_hits_in_strobes,
    extend_sequence_from_base,
    plot_shift_sum_curve,
    search_period_by_shift_sum,
    tile_strobe_windows,
    validate_period,
)
from generator import (
    BurstConfig,
    ImpulseTimingConfig,
    build_repeated_signal,
    generate_impulse_pack_times,
)

def plot_results(
    signal: np.ndarray,
    burst: np.ndarray,
    result,
    validation: dict[str, float | bool],
    true_period: int,
) -> None:
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(burst)
    ax1.set_title("Одна пачка")
    ax1.set_xlabel("Отсчёт")
    ax1.set_ylabel("Амплитуда")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    preview = signal[: min(len(signal), true_period * 3)]
    ax2.plot(preview)
    ax2.set_title("Первые 3 периода последовательности")
    ax2.set_xlabel("Отсчёт")
    ax2.set_ylabel("Амплитуда")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(result.lag_values, result.score_curve, label="Сумма со сдвигом")
    ax3.axvline(true_period, linestyle="--", label=f"Истинный период = {true_period}")
    ax3.axvline(result.estimated_period, linestyle=":", label=f"Найденный период = {result.estimated_period}")
    ax3.set_title("Кривая поиска периода")
    ax3.set_xlabel("Сдвиг")
    ax3.set_ylabel("Скор")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    text = (
        f"Истинный период: {true_period}\n"
        f"Оценённый период: {result.estimated_period}\n"
        f"Пик/2-й пик: {result.confidence_ratio:.3f}\n"
        f"Нормированный пик: {result.normalized_peak:.3f}\n"
        f"Повторяемость: {validation['repeat_consistency']:.3f}\n"
        f"Энергетическое отношение: {validation['energy_ratio']:.3f}\n"
        f"Проверка результата: {'OK' if validation['is_valid'] and result.is_plausible else 'FAIL'}"
    )
    fig.text(0.73, 0.52, text, fontsize=11, va="top")

    fig.suptitle("Анализатор периода повторяющейся пачки импульсов", fontsize=16)
    plt.tight_layout()
    plt.show()

def main() -> None:
    generator_config = BurstConfig(
        burst_length=900,
        repeats=15,
        noise_std=0.08,
        baseline=0.0,
        seed=7,
    )

    signal, burst = build_repeated_signal(generator_config)
    true_period = generator_config.burst_length

    search_config = PeriodSearchConfig(
        min_period=700,
        max_period=1100,
        top_exclusion_radius=8,
        min_confidence_ratio=1.05,
        min_normalized_peak=1.0,
    )

    result = search_period_by_shift_sum(signal, search_config)
    validation = validate_period(signal, result.estimated_period)

    print("=== RESULT ===")
    print(f"True period       : {true_period}")
    print(f"Estimated period  : {result.estimated_period}")
    print(f"Best score        : {result.best_score:.6f}")
    print(f"Second best score : {result.second_best_score:.6f}")
    print(f"Confidence ratio  : {result.confidence_ratio:.6f}")
    print(f"Normalized peak   : {result.normalized_peak:.6f}")
    print(f"Plausible result  : {result.is_plausible}")
    print(f"Repeat consistency: {validation['repeat_consistency']:.6f}")
    print(f"Energy ratio      : {validation['energy_ratio']:.6f}")
    print(f"Validation passed : {validation['is_valid']}")

    plot_results(signal, burst, result, validation, true_period)

    timing_config = ImpulseTimingConfig(
        num_impulses_in_pack=15,
        min_interval_ms=40,
        max_interval_ms=80,
        generation_duration_s=12.0,
        noise_sigma_ms=2.0,
        seed=42,
    )

    times_ms, impulse_rel_times = generate_impulse_pack_times(timing_config)
    true_period_ms = int(round((impulse_rel_times[-1] if impulse_rel_times else 0) + (impulse_rel_times[1] if len(impulse_rel_times) > 1 else 0)))

    timeline_config = PeriodSearchConfig(
        min_period=max(1, int(true_period_ms * 0.5)),
        max_period=max(int(true_period_ms * 1.5), 1),
        top_exclusion_radius=8,
        min_confidence_ratio=1.05,
        min_normalized_peak=1.0,
    )

    _, timeline_result, timeline_validation = analyze_impulse_times(
        times_ms,
        config=timeline_config,
        resolution_ms=1.0,
    )

    print("=== TIMELINE SHIFT-SUM ===")
    print(f"True period (ms)   : {true_period_ms}")
    print(f"Estimated period   : {timeline_result.estimated_period}")
    print(f"Confidence ratio   : {timeline_result.confidence_ratio:.3f}")
    print(f"Normalized peak    : {timeline_result.normalized_peak:.3f}")
    print(f"Repeat consistency : {timeline_validation['repeat_consistency']:.3f}")
    print(f"Energy ratio       : {timeline_validation['energy_ratio']:.3f}")
    print(f"Validation passed  : {timeline_validation['is_valid'] and timeline_result.is_plausible}")

    # Строим стробы для одной пачки и разворачиваем их по всем наблюдаемым пачкам.
    relative_offsets = impulse_rel_times or [0.0]
    base_start_ms = times_ms[0] - relative_offsets[0] if times_ms else 0.0
    strobe_tolerance_ms = 3.0
    base_strobes = build_strobe_windows(relative_offsets, strobe_tolerance_ms)
    observed_pack_count = max(1, len(times_ms) // len(relative_offsets))
    observed_strobes = tile_strobe_windows(
        base_strobes,
        timeline_result.estimated_period,
        observed_pack_count,
        start_time_ms=base_start_ms,
    )
    obs_hits, obs_misses = count_hits_in_strobes(times_ms, observed_strobes)

    # Пытаемся продолжить ряд и проверяем попадания модельной продолжения в те же окна.
    continuation_packs = 3
    continuation_start_ms = base_start_ms + observed_pack_count * timeline_result.estimated_period
    continuation_times = extend_sequence_from_base(
        relative_offsets,
        continuation_start_ms,
        continuation_packs,
        timeline_result.estimated_period,
    )
    continuation_strobes = tile_strobe_windows(
        base_strobes,
        timeline_result.estimated_period,
        continuation_packs,
        start_time_ms=continuation_start_ms,
    )
    cont_hits, cont_misses = count_hits_in_strobes(continuation_times, continuation_strobes)

    print(f"Strobe hits (observed): {obs_hits}/{len(times_ms)} (misses {obs_misses})")
    print(
        f"Strobe hits (continuation): {cont_hits}/{len(continuation_times)} (misses {cont_misses})"
    )

    plot_impulse_timeline(
        times_ms,
        num_packs=3,
        impulses_per_pack=15,
        strobes=observed_strobes,
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_shift_sum_curve(timeline_result, true_period=true_period_ms, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_impulse_timeline(
    times_ms: Sequence[float],
    num_packs: int = 3,
    impulses_per_pack: int = 15,
    epsilon_s: float = 0.005,
    strobes: Sequence[StrobeWindow] | None = None,
) -> None:
    plot_times_s = [t / 1000.0 for t in times_ms[: num_packs * impulses_per_pack]]
    separator_times: list[float] = []
    for i in range(num_packs):
        last_idx = (i + 1) * impulses_per_pack - 1
        if last_idx < len(plot_times_s):
            separator_times.append(plot_times_s[last_idx] + epsilon_s)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.vlines(plot_times_s, 0, 1, colors="blue", linewidth=1.5, label="Импульсы")
    ax.vlines(separator_times, 0, 1, colors="red", linewidth=1.5, linestyles="--", label="Границы пачек")
    ax.set_xlabel("Время (секунды)")
    ax.set_ylabel("Амплитуда импульса")
    ax.set_title("Импульсы из временных меток")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if strobes:
        for strobe in strobes:
            ax.axvspan(
                strobe.start_ms / 1000.0,
                strobe.end_ms / 1000.0,
                color="green",
                alpha=0.5,
            )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
