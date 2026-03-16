import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signal):

    plt.figure(figsize=(12,3))
    plt.plot(signal)

    plt.title("Signal")
    plt.xlabel("time")
    plt.ylabel("amplitude")

    plt.show()


def plot_scores(scores, min_period):

    periods = np.arange(min_period, min_period + len(scores))

    plt.figure(figsize=(12,3))
    plt.plot(periods, scores)

    plt.title("Shift-sum period detection")
    plt.xlabel("period")
    plt.ylabel("score")

    plt.show()


def plot_fft_spectrum(magnitude):

    plt.figure(figsize=(12,3))
    plt.plot(magnitude)

    plt.title("FFT spectrum")
    plt.xlabel("frequency bin")
    plt.ylabel("magnitude")

    plt.show()


def plot_packet_alignment(signal, period):

    plt.figure(figsize=(12,4))

    segments = len(signal) // period

    for i in range(segments):

        start = i * period
        end = start + period

        plt.plot(signal[start:end], alpha=0.5)

    plt.title("Packet alignment")
    plt.xlabel("time inside period")

    plt.show()


def plot_period_heatmap(signal, max_period=500):

    heat = []

    for p in range(10, max_period):

        overlap = signal[:-p] * signal[p:]
        heat.append(overlap[:200])

    heat = np.array(heat)

    plt.figure(figsize=(6,6))

    plt.imshow(heat, aspect='auto', cmap="hot")

    plt.title("Period heatmap")
    plt.xlabel("time")
    plt.ylabel("period shift")

    plt.show()