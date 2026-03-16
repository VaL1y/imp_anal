import numpy as np
import matplotlib.pyplot as plt


def enable_interactive():
    plt.ion()


def disable_interactive():
    plt.ioff()


def show_all():
    plt.show()


def plot_signal(signal, title="Signal", show=False):
    plt.figure(figsize=(14, 3))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_signal_fragment(signal, start=0, length=1000, title="Signal fragment", show=False):
    end = min(len(signal), start + length)
    x = np.arange(start, end)

    plt.figure(figsize=(14, 3))
    plt.plot(x, signal[start:end])
    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_autocorr(corr, detected_period=None, title="Autocorrelation", show=False):
    plt.figure(figsize=(14, 4))
    plt.plot(corr, label="autocorr")

    if detected_period is not None and 0 <= detected_period < len(corr):
        plt.axvline(detected_period, linestyle="--", label=f"detected={detected_period}")

    plt.title(title)
    plt.xlabel("lag")
    plt.ylabel("correlation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def plot_fft_magnitude(magnitude, title="FFT magnitude", show=False):
    plt.figure(figsize=(14, 4))
    plt.plot(magnitude)
    plt.title(title)
    plt.xlabel("frequency bin")
    plt.ylabel("magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def plot_period_history(period_history, true_period=None, title="Online period history", show=False):
    x = np.arange(len(period_history))
    y = np.array([np.nan if v is None else v for v in period_history], dtype=float)

    plt.figure(figsize=(14, 4))
    plt.plot(x, y, label="detected period")

    if true_period is not None:
        plt.axhline(true_period, linestyle="--", label=f"true={true_period}")

    plt.title(title)
    plt.xlabel("update step")
    plt.ylabel("period")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def plot_period_overlay(signal, period, max_periods=10, title="Overlay by detected period", show=False):
    if period is None or period <= 0:
        return

    periods_count = min(len(signal) // period, max_periods)

    plt.figure(figsize=(14, 4))
    for i in range(periods_count):
        start = i * period
        end = start + period
        plt.plot(signal[start:end], alpha=0.6)

    plt.title(title)
    plt.xlabel("position inside period")
    plt.ylabel("amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def print_period_comparison(true_period, detected_period, title="Period comparison"):
    print(title)
    print(f"True period     : {true_period}")
    print(f"Detected period : {detected_period}")

    if detected_period is None:
        print("Absolute error  : None")
        print("Relative error  : None")
        return

    abs_error = detected_period - true_period
    rel_error = abs_error / true_period

    print(f"Absolute error  : {abs_error}")
    print(f"Relative error  : {rel_error:.6f}")


class LivePeriodPlot:
    """
    Живой график для online-анализа.
    Обновляется по мере поступления новых данных.
    """

    def __init__(self, true_period=None, title="Live online period detection"):
        plt.ion()

        self.true_period = true_period
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.set_title(title)
        self.ax.set_xlabel("update step")
        self.ax.set_ylabel("period")
        self.ax.grid(True, alpha=0.3)

        (self.line,) = self.ax.plot([], [], label="detected period")

        if true_period is not None:
            self.ax.axhline(true_period, linestyle="--", label=f"true={true_period}")

        self.ax.legend()
        self.fig.tight_layout()

    def update(self, period_history):
        x = np.arange(len(period_history))
        y = np.array([np.nan if p is None else p for p in period_history], dtype=float)

        self.line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)