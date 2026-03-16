import numpy as np


class PeriodAnalyzer:
    def __init__(
        self,
        method: str = "autocorr",
        tolerant: bool = False,
        tolerance: float = 0.15,
        online_buffer_size: int = 8192,
        min_samples_to_analyze: int = 512,
        min_period_ratio: float = 0.08,
        debug: bool = False,
    ):
        self.method = method
        self.tolerant = tolerant
        self.tolerance = tolerance
        self.online_buffer_size = online_buffer_size
        self.min_samples_to_analyze = min_samples_to_analyze
        self.min_period_ratio = min_period_ratio
        self.debug = debug

        self.buffer = []

        self.detected_period = None
        self.period_history = []

        self.last_signal = None
        self.last_scores = None
        self.last_autocorr = None
        self.last_fft_magnitude = None
        self.last_method_used = None

    @staticmethod
    def _autocorrelation(signal: np.ndarray) -> np.ndarray:
        corr = np.correlate(signal, signal, mode="full")
        return corr[len(corr) // 2:]

    @staticmethod
    def _find_peaks_simple(x: np.ndarray) -> np.ndarray:
        if len(x) < 3:
            return np.array([], dtype=int)

        peaks = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] >= x[i + 1]:
                peaks.append(i)
        return np.array(peaks, dtype=int)

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return x.copy()

        kernel = np.ones(window, dtype=float) / window
        return np.convolve(x, kernel, mode="same")

    def _score_peak(self, values: np.ndarray, idx: int) -> float:
        if not self.tolerant:
            return float(values[idx])

        radius = max(1, int(round(self.tolerance * 10)))
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        return float(np.mean(values[left:right]))

    def _select_candidate_peaks(self, corr: np.ndarray, signal_length: int) -> np.ndarray:
        """
        Отсекаем слишком малые лаги, чтобы не ловить расстояние
        между импульсами внутри пачки.
        """
        peaks = self._find_peaks_simple(corr)
        if len(peaks) == 0:
            return peaks

        min_period = max(2, int(signal_length * self.min_period_ratio))
        peaks = peaks[peaks >= min_period]

        return peaks


    def _detect_period_autocorr(self, signal: np.ndarray) -> int | None:
        corr = self._autocorrelation(signal).astype(float)

        if len(corr) < 3:
            self.last_autocorr = corr
            self.last_scores = corr
            return None

        corr[0] = 0.0

        processed_corr = corr.copy()
        if self.tolerant:
            processed_corr = self._moving_average(processed_corr, window=5)

        peaks = self._select_candidate_peaks(processed_corr, len(signal))
        if len(peaks) == 0:
            self.last_autocorr = corr
            self.last_scores = processed_corr
            return None

        scored_peaks = [(peak, self._score_peak(processed_corr, peak)) for peak in peaks]
        scored_peaks.sort(key=lambda item: item[1], reverse=True)

        detected = int(scored_peaks[0][0])

        self.last_autocorr = corr
        self.last_scores = processed_corr

        if self.debug:
            top = scored_peaks[:10]
            print("[AUTOCORR] top peaks:", [(int(p), round(float(s), 2)) for p, s in top])

        return detected

    # def _detect_period_fft(self, signal: np.ndarray) -> int | None:
    #     n = len(signal)
    #     if n < 2:
    #         return None
    #
    #     spectrum = np.fft.rfft(signal)
    #     magnitude = np.abs(spectrum)
    #     magnitude[0] = 0.0
    #
    #     self.last_fft_magnitude = magnitude
    #     self.last_scores = magnitude
    #
    #     # пропускаем слишком высокие частоты, которые соответствуют слишком маленьким периодам
    #     max_bin_to_ignore = max(1, int(len(magnitude) * 0.05))
    #     magnitude_search = magnitude.copy()
    #     magnitude_search[max_bin_to_ignore:] = magnitude[max_bin_to_ignore:]
    #
    #     peak_bin = np.argmax(magnitude_search[1:]) + 1
    #     if peak_bin <= 0:
    #         return None
    #
    #     period = int(round(n / peak_bin))
    #
    #     if self.debug:
    #         print(f"[FFT] peak_bin={peak_bin}, estimated_period={period}")
    #
    #     return period

    def _run_detection(self, signal: np.ndarray) -> int | None:
        self.last_signal = np.asarray(signal, dtype=float)
        self.last_method_used = self.method

        if self.method == "autocorr":
            return self._detect_period_autocorr(self.last_signal)

        # if self.method == "fft":
        #     return self._detect_period_fft(self.last_signal)

        raise ValueError(f"Unsupported method: {self.method}")

    def analyze_offline(self, signal: np.ndarray) -> int | None:
        self.detected_period = self._run_detection(np.asarray(signal, dtype=float))
        return self.detected_period

    def update_online(self, samples) -> int | None:
        """
        Можно подавать:
        - один sample
        - список
        - numpy array (батч)
        """
        if np.isscalar(samples):
            samples = [samples]

        for sample in samples:
            self.buffer.append(float(sample))

        if len(self.buffer) > self.online_buffer_size:
            overflow = len(self.buffer) - self.online_buffer_size
            self.buffer = self.buffer[overflow:]

        if len(self.buffer) < self.min_samples_to_analyze:
            self.period_history.append(None)
            return None

        signal = np.array(self.buffer, dtype=float)
        detected = self._run_detection(signal)

        self.detected_period = detected
        self.period_history.append(detected)
        return detected

    def reset_online(self):
        self.buffer = []
        self.detected_period = None
        self.period_history = []
        self.last_signal = None
        self.last_scores = None
        self.last_autocorr = None
        self.last_fft_magnitude = None
        self.last_method_used = None