import numpy as np

from generator import generate_signal
from analyzer import (
    find_period_shift_sum,
    find_period_tolerant,
    find_period_fft
)

import visualization as viz


def main():

    # --------------------
    # параметры сигнала
    # --------------------

    period = np.random.randint(800, 1200)

    repeats = 15
    num_pulses = np.random.randint(10, 15)

    pulse_width = 5

    jitter_std = 0

    print("TRUE PERIOD:", period)

    # --------------------
    # генерация сигнала
    # --------------------

    signal = generate_signal(
        period,
        repeats,
        num_pulses,
        pulse_width,
        jitter_std
    )

    viz.plot_signal(signal)

    # --------------------
    # поиск периода
    # --------------------

    detected_period, scores = find_period_tolerant(
        signal,
        min_period=200,
        max_period=2000
    )

    print("DETECTED PERIOD:", detected_period)

    viz.plot_scores(scores, 200)

    # --------------------
    # FFT (опционально)
    # --------------------

    use_fft = True

    if use_fft:

        fft_period, magnitude = find_period_fft(signal)

        print("FFT PERIOD:", fft_period)

        viz.plot_fft_spectrum(magnitude)

    # --------------------
    # дополнительные графики
    # --------------------

    viz.plot_packet_alignment(signal, detected_period)

    viz.plot_period_heatmap(signal)


if __name__ == "__main__":
    main()