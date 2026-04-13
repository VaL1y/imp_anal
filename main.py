from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analyzer import PeriodSearchConfig, search_period_by_shift_sum, validate_period
from generator import BurstConfig, build_repeated_signal

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


if __name__ == "__main__":
    main()