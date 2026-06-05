from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


PACKET_DURATION_US = 50_000.0
IMPULSES_PER_PACKET = 21
IMPULSE_WIDTH_US = 1.0
JITTER_MIN_US = 0.3
JITTER_MAX_US = 1.0
STROBE_WIDTH_US = 40.0
PACKET_COUNT = 7

MATCH_TOLERANCE_US = 5.0
SHIFT_STEP_US = 5.0
MIN_ANALYZED_SHIFT_US = 2.0 * MATCH_TOLERANCE_US
MIN_PULSE_SPACING_US = 1_000.0
SEED = 42


@dataclass(slots=True)
class ShiftSumResult:
    shifts_us: np.ndarray
    sums: np.ndarray
    scores: np.ndarray
    best_shift_us: float
    best_sum: int
    best_score: float


def gaussian_jitter(rng: np.random.Generator) -> float:
    sigma = float(rng.uniform(JITTER_MIN_US, JITTER_MAX_US))
    return float(rng.normal(0.0, sigma))


def build_packet(rng: np.random.Generator) -> np.ndarray:
    positions: list[float] = []
    while len(positions) < IMPULSES_PER_PACKET:
        candidate = float(rng.uniform(500.0, PACKET_DURATION_US - 500.0))
        if all(abs(candidate - item) >= MIN_PULSE_SPACING_US for item in positions):
            positions.append(candidate)
    return np.array(sorted(positions), dtype=float)


def build_signal(packet: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    jitters: list[float] = []
    for packet_index in range(PACKET_COUNT):
        base = packet_index * PACKET_DURATION_US
        for position in packet:
            jitter = gaussian_jitter(rng)
            times.append(base + position + jitter)
            jitters.append(jitter)
    return np.array(times, dtype=float), np.array(jitters, dtype=float)


def count_shift_hits(times: np.ndarray, shift_us: float) -> int:
    shifted = times + shift_us
    indices = np.searchsorted(times, shifted)

    left = indices > 0
    left_hits = np.zeros(len(times), dtype=bool)
    left_hits[left] = np.abs(times[indices[left] - 1] - shifted[left]) <= MATCH_TOLERANCE_US

    right = indices < len(times)
    right_hits = np.zeros(len(times), dtype=bool)
    right_hits[right] = np.abs(times[indices[right]] - shifted[right]) <= MATCH_TOLERANCE_US

    return int(np.count_nonzero(left_hits | right_hits))


def shift_sum(times: np.ndarray) -> ShiftSumResult:
    max_shift = max(MIN_ANALYZED_SHIFT_US, (times[-1] - times[0]) / 2.0)
    shifts = np.arange(MIN_ANALYZED_SHIFT_US, max_shift + SHIFT_STEP_US, SHIFT_STEP_US)
    sums = np.array([count_shift_hits(times, shift) for shift in shifts], dtype=int)

    possible = np.array([np.count_nonzero(times + shift <= times[-1] + MATCH_TOLERANCE_US) for shift in shifts])
    scores = np.divide(sums, possible, out=np.zeros_like(sums, dtype=float), where=possible > 0)

    best_index = int(np.argmax(sums))
    return ShiftSumResult(
        shifts_us=shifts,
        sums=sums,
        scores=scores,
        best_shift_us=float(shifts[best_index]),
        best_sum=int(sums[best_index]),
        best_score=float(scores[best_index]),
    )


def online_snapshots(times: np.ndarray) -> dict[int, ShiftSumResult]:
    checkpoints = [4, IMPULSES_PER_PACKET, 2 * IMPULSES_PER_PACKET, 4 * IMPULSES_PER_PACKET, len(times)]
    snapshots: dict[int, ShiftSumResult] = {}
    for count in checkpoints:
        if count <= len(times):
            snapshots[count] = shift_sum(times[:count])
    return snapshots


def reconstruct_packet(times: np.ndarray, period_us: float) -> np.ndarray:
    phases = np.sort(np.mod(times, period_us))
    clusters: list[list[float]] = []
    for phase in phases:
        if not clusters or abs(np.mean(clusters[-1]) - phase) > STROBE_WIDTH_US / 2.0:
            clusters.append([float(phase)])
        else:
            clusters[-1].append(float(phase))

    centers = np.array([np.mean(cluster) for cluster in clusters], dtype=float)
    if len(centers) > IMPULSES_PER_PACKET:
        counts = np.array([len(cluster) for cluster in clusters])
        centers = centers[np.argsort(counts)[-IMPULSES_PER_PACKET:]]
    return np.sort(centers)


def strobe_hits(predicted_times: np.ndarray, true_packet: np.ndarray) -> tuple[int, list[bool], np.ndarray]:
    true_centers = true_packet + PACKET_COUNT * PACKET_DURATION_US
    half = STROBE_WIDTH_US / 2.0
    hit_mask: list[bool] = []
    for predicted in predicted_times:
        inside = np.any(np.abs(true_centers - predicted) <= half)
        hit_mask.append(bool(inside))
    return sum(hit_mask), hit_mask, true_centers


def plot_timeline(times: np.ndarray, packet: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.vlines(times, 0.0, 1.0, color="#1f77b4", linewidth=1.0, label="принятые импульсы")
    for index in range(PACKET_COUNT + 1):
        ax.axvline(index * PACKET_DURATION_US, color="#555555", linestyle="--", alpha=0.45, linewidth=1)
    ax.set_title("Принятая псевдохаотическая последовательность импульсов с джиттером")
    ax.set_xlabel("время, мкс")
    ax.set_yticks([])
    ax.set_xlim(0, PACKET_COUNT * PACKET_DURATION_US)
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.01,
        0.92,
        f"{PACKET_COUNT} пачек x {IMPULSES_PER_PACKET} импульс, пачка = {PACKET_DURATION_US / 1000:.0f} мс",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()


def plot_jitter(jitters: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(jitters, bins=28, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_title("Гауссов временной джиттер")
    ax.set_xlabel("временной сдвиг импульса, мкс")
    ax.set_ylabel("число импульсов")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.95,
        f"среднее = {np.mean(jitters):.3f} мкс\nСКО = {np.std(jitters):.3f} мкс",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()


def plot_shift_sums(snapshots: dict[int, ShiftSumResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = ["#9ecae9", "#6baed6", "#3182bd", "#08519c", "#d62728"]

    for color, (count, result) in zip(colors, snapshots.items()):
        label = f"{count} имп."
        axes[0].plot(result.shifts_us, result.sums, color=color, linewidth=1.4, label=label)
        axes[1].plot(result.shifts_us, result.scores, color=color, linewidth=1.4, label=label)

    final_result = list(snapshots.values())[-1]
    for ax in axes:
        ax.axvline(PACKET_DURATION_US, color="#111111", linestyle="--", linewidth=1.4, label="истинный период")
        ax.axvline(final_result.best_shift_us, color="#ff7f0e", linestyle=":", linewidth=2.0, label="найденный период")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[0].set_title("Онлайн-накопление суммы совпадений со сдвигом")
    axes[0].set_ylabel("сумма совпадений")
    axes[1].set_title("Нормированная оценка совпадений")
    axes[1].set_ylabel("оценка")
    axes[1].set_xlabel("кандидат сдвига, мкс")
    axes[1].set_xlim(0, final_result.shifts_us[-1])
    fig.tight_layout()


def plot_prediction(
    predicted_times: np.ndarray,
    true_centers: np.ndarray,
    hit_mask: list[bool],
    detected_period_us: float,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    half = STROBE_WIDTH_US / 2.0

    for center in true_centers:
        ax.axvspan(center - half, center + half, color="#54a24b", alpha=0.24)
        ax.axvline(center, color="#54a24b", alpha=0.8, linewidth=1)

    hit_times = [time for time, hit in zip(predicted_times, hit_mask) if hit]
    miss_times = [time for time, hit in zip(predicted_times, hit_mask) if not hit]
    if hit_times:
        ax.vlines(hit_times, 0.1, 1.0, color="#2ca02c", linewidth=1.7, label="попадание прогноза")
    if miss_times:
        ax.vlines(miss_times, 0.1, 1.0, color="#d62728", linewidth=1.7, label="промах прогноза")

    ax.set_title("Прогноз следующей пачки и строб-окна приемника")
    ax.set_xlabel("время, мкс")
    ax.set_yticks([])
    ax.set_xlim(PACKET_COUNT * PACKET_DURATION_US - 1000.0, (PACKET_COUNT + 1) * PACKET_DURATION_US + 1000.0)
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.01,
        0.92,
        f"найденный период = {detected_period_us:.1f} мкс, ширина строба = {STROBE_WIDTH_US:.1f} мкс",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()


def plot_signal_then_strobes_and_response(
    received_times: np.ndarray,
    predicted_times: np.ndarray,
    true_centers: np.ndarray,
    hit_mask: list[bool],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    end_of_observation = PACKET_COUNT * PACKET_DURATION_US
    end_of_response = (PACKET_COUNT + 1) * PACKET_DURATION_US
    half = STROBE_WIDTH_US / 2.0

    axes[0].vlines(received_times, 0.0, 1.0, color="#1f77b4", linewidth=1.0, label="принятый сигнал")
    for index in range(PACKET_COUNT + 2):
        axes[0].axvline(index * PACKET_DURATION_US, color="#777777", linestyle="--", alpha=0.35, linewidth=1)
        axes[1].axvline(index * PACKET_DURATION_US, color="#777777", linestyle="--", alpha=0.35, linewidth=1)

    for center in true_centers:
        axes[0].axvspan(center - half, center + half, color="#54a24b", alpha=0.26)
        axes[1].axvspan(center - half, center + half, color="#54a24b", alpha=0.18)

    axes[0].axvline(end_of_observation, color="#111111", linewidth=1.4)
    axes[1].axvline(end_of_observation, color="#111111", linewidth=1.4)
    axes[0].text(end_of_observation + 700.0, 0.92, "строб-окна приемника", va="top")

    hit_times = [time for time, hit in zip(predicted_times, hit_mask) if hit]
    miss_times = [time for time, hit in zip(predicted_times, hit_mask) if not hit]
    if hit_times:
        axes[1].vlines(hit_times, 0.0, 1.0, color="#2ca02c", linewidth=1.8, label="попал в строб")
    if miss_times:
        axes[1].vlines(miss_times, 0.0, 1.0, color="#d62728", linewidth=1.8, label="не попал")

    axes[0].set_title("Наблюдение сигнала и последующие строб-окна")
    axes[1].set_title("Результат анализа: прогнозная ответная пачка")
    axes[1].set_xlabel("время, мкс")
    axes[0].set_ylabel("прием")
    axes[1].set_ylabel("ответ")
    for ax in axes:
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="upper right")
    axes[1].set_xlim(0, end_of_response)
    fig.tight_layout()


def main() -> None:
    rng = np.random.default_rng(SEED)
    packet = build_packet(rng)
    times, jitters = build_signal(packet, rng)

    analysis_started = perf_counter()
    snapshots = online_snapshots(times)
    result = snapshots[len(times)]
    reconstructed = reconstruct_packet(times, result.best_shift_us)
    predicted = reconstructed + PACKET_COUNT * result.best_shift_us
    hits, hit_mask, true_centers = strobe_hits(predicted, packet)
    analysis_time_s = perf_counter() - analysis_started
    observation_time_s = PACKET_COUNT * PACKET_DURATION_US / 1_000_000.0

    print("=== Онлайн-суммирование со сдвигом в микросекундах ===")
    print(f"Истинная длительность пачки : {PACKET_DURATION_US:.1f} мкс")
    print(f"Найденный период            : {result.best_shift_us:.1f} мкс")
    print(f"Ошибка периода              : {result.best_shift_us - PACKET_DURATION_US:.1f} мкс")
    print(f"Лучшая сумма совпадений     : {result.best_sum}")
    print(f"Лучшая нормированная оценка : {result.best_score:.3f}")
    print(f"Восстановлено импульсов     : {len(reconstructed)}")
    print(f"Попадания в стробы          : {hits}/{len(predicted)}")
    print(f"Время наблюдения сигнала    : {observation_time_s:.3f} с")
    print(f"Время расчета               : {analysis_time_s:.3f} с")
    print(f"Итого время реакции модели  : {observation_time_s + analysis_time_s:.3f} с")

    plot_timeline(times, packet)
    plot_jitter(jitters)
    plot_shift_sums(snapshots)
    plot_signal_then_strobes_and_response(times, predicted, true_centers, hit_mask)
    plot_prediction(predicted, true_centers, hit_mask, result.best_shift_us)
    plt.show()


if __name__ == "__main__":
    main()
