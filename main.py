import time

from generator import PulseGenerator
from analyzer import PeriodAnalyzer
import visualization as viz


def run_offline_demo():
    true_period = 1000
    repeats = 15
    num_pulses = 12
    pulse_width = 5

    generator = PulseGenerator(
        period=true_period,
        repeats=repeats,
        num_pulses=num_pulses,
        pulse_width=pulse_width,
        seed=42,
    )

    signal = generator.generate_from_mode(
        mode="jitter",
        jitter_std=2.0,
    )

    analyzer = PeriodAnalyzer(
        method="autocorr",
        tolerant=True,
        tolerance=0.15,
        min_period_ratio=0.08,
        debug=False,
    )

    detected_period = analyzer.analyze_offline(signal)

    viz.print_period_comparison(
        true_period,
        detected_period,
        title="Offline analysis",
    )

    viz.plot_signal(signal, title="Generated signal")
    viz.plot_signal_fragment(signal, start=0, length=3000, title="Signal fragment")

    if analyzer.last_autocorr is not None:
        viz.plot_autocorr(
            analyzer.last_autocorr,
            detected_period=detected_period,
            title="Autocorrelation",
        )

    if detected_period is not None:
        viz.plot_period_overlay(
            signal,
            detected_period,
            title="Overlay by detected period",
        )

    viz.show_all()


def run_online_demo_by_samples(live_plot=True, sleep_sec=0.0):
    true_period = 900

    generator = PulseGenerator(
        period=true_period,
        repeats=15,
        num_pulses=10,
        pulse_width=4,
        seed=7,
    )

    signal = generator.generate_from_mode(
        mode="jitter",
        jitter_std=1.5,
    )

    analyzer = PeriodAnalyzer(
        method="autocorr",
        tolerant=True,
        tolerance=0.15,
        online_buffer_size=4096,
        min_samples_to_analyze=600,
        min_period_ratio=0.08,
        debug=False,
    )

    live = None
    if live_plot:
        live = viz.LivePeriodPlot(
            true_period=true_period,
            title="Live online detection by samples",
        )

    print()
    print("=== ONLINE DEMO BY SAMPLES ===")
    print(f"TRUE PERIOD: {true_period}")

    for i, sample in enumerate(generator.stream_samples(signal), start=1):
        detected = analyzer.update_online(sample)

        if i % 50 == 0:
            print(f"[sample {i:5d}] detected_period = {detected}")

            if live is not None:
                live.update(analyzer.period_history)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

    viz.print_period_comparison(
        true_period,
        analyzer.detected_period,
        title="Final online analysis",
    )

    viz.plot_period_history(
        analyzer.period_history,
        true_period=true_period,
        title="Online detection history by samples",
    )

    viz.show_all()


def run_online_demo_by_batches(live_plot=True, batch_size=128, sleep_sec=0.0):
    true_period = 1100

    generator = PulseGenerator(
        period=true_period,
        repeats=15,
        num_pulses=13,
        pulse_width=5,
        seed=11,
    )

    signal = generator.generate_from_mode(mode="regular")

    analyzer = PeriodAnalyzer(
        # method="fft",
        tolerant=False,
        online_buffer_size=8192,
        min_samples_to_analyze=1024,
        min_period_ratio=0.08,
        debug=False,
    )

    live = None
    if live_plot:
        live = viz.LivePeriodPlot(
            true_period=true_period,
            title="Live online detection by batches (FFT)",
        )

    print()
    print("=== ONLINE DEMO BY BATCHES ===")
    print(f"TRUE PERIOD: {true_period}")

    for batch_idx, batch in enumerate(generator.stream_batches(batch_size=batch_size, signal=signal), start=1):
        detected = analyzer.update_online(batch)

        print(
            f"[batch {batch_idx:4d}] "
            f"batch_len = {len(batch):4d}, "
            f"buffer_len = {len(analyzer.buffer):5d}, "
            f"detected_period = {detected}"
        )

        if live is not None:
            live.update(analyzer.period_history)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    viz.print_period_comparison(
        true_period,
        analyzer.detected_period,
        title="Final online batch analysis",
    )

    if analyzer.last_fft_magnitude is not None:
        viz.plot_fft_magnitude(
            analyzer.last_fft_magnitude,
            title="Last FFT magnitude",
        )

    viz.plot_period_history(
        analyzer.period_history,
        true_period=true_period,
        title="Online detection history by batches (FFT)",
    )

    viz.show_all()


def run_simple_random_demo():
    """
    Короткий сценарий, похожий на ваш ранний вариант:
    случайно создаём период, генерируем сигнал и проверяем,
    что нашёл анализатор.
    """
    import numpy as np

    true_period = int(np.random.randint(800, 1200))
    repeats = 15
    num_pulses = int(np.random.randint(10, 15))
    pulse_width = 5
    jitter_std = 2.0

    generator = PulseGenerator(
        period=true_period,
        repeats=repeats,
        num_pulses=num_pulses,
        pulse_width=pulse_width,
        seed=123,
    )

    signal = generator.generate_from_mode(
        mode="jitter",
        jitter_std=jitter_std,
    )

    analyzer = PeriodAnalyzer(
        method="autocorr",
        tolerant=True,
        tolerance=0.15,
        min_period_ratio=0.08,
        debug=False,
    )

    detected_period = analyzer.analyze_offline(signal)

    print()
    print("=== SIMPLE RANDOM DEMO ===")
    print(f"TRUE PERIOD:     {true_period}")
    print(f"DETECTED PERIOD: {detected_period}")

    viz.plot_signal_fragment(signal, start=0, length=min(4000, len(signal)), title="Random demo signal fragment")

    if analyzer.last_autocorr is not None:
        viz.plot_autocorr(
            analyzer.last_autocorr,
            detected_period=detected_period,
            title="Random demo autocorrelation",
        )

    viz.show_all()


if __name__ == "__main__":
    # 1. Оффлайн-анализ — все окна открываются сразу
    run_offline_demo()

    # 2. Онлайн-анализ по сэмплам
    # sleep_sec можно поставить, например, 0.02, если хочется медленнее и нагляднее
    run_online_demo_by_samples(live_plot=True, sleep_sec=0.0)

    # 3. Онлайн-анализ по батчам
    run_online_demo_by_batches(live_plot=True, batch_size=128, sleep_sec=0.0)

    # 4. Дополнительный короткий случайный сценарий
    run_simple_random_demo()