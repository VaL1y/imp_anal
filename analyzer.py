import numpy as np


def find_period_shift_sum(signal,
                          min_period,
                          max_period):

    scores = []

    for p in range(min_period, max_period):

        overlap = signal[:-p] * signal[p:]
        score = np.sum(overlap)

        scores.append(score)

    scores = np.array(scores)

    best_index = np.argmax(scores)
    detected_period = min_period + best_index

    return detected_period, scores


def find_period_tolerant(signal,
                         min_period,
                         max_period,
                         tolerance=0.8):

    detected, scores = find_period_shift_sum(
        signal,
        min_period,
        max_period
    )

    threshold = np.max(scores) * tolerance

    candidates = np.where(scores >= threshold)[0]

    period = min_period + candidates[0]

    return period, scores


def find_period_fft(signal):

    spectrum = np.fft.rfft(signal)

    magnitude = np.abs(spectrum)

    magnitude[0] = 0

    peak = np.argmax(magnitude)

    period = len(signal) / peak if peak != 0 else None

    return int(period), magnitude