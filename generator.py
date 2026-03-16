import numpy as np


class PulseGenerator:
    """
    Генератор пачек импульсов.
    Может генерировать:
    - идеальный периодический сигнал
    - сигнал с джиттером (смещение времени прихода импульсов)
    - в будущем можно расширить другими режимами
    """

    def __init__(
        self,
        period: int,
        repeats: int,
        num_pulses: int,
        pulse_width: int,
        amplitude: float = 1.0,
        seed: int | None = None,
    ):
        self.period = period
        self.repeats = repeats
        self.num_pulses = num_pulses
        self.pulse_width = pulse_width
        self.amplitude = amplitude
        self.rng = np.random.default_rng(seed)

        self.last_packet = None
        self.last_signal = None
        self.last_mode = None
        self.last_params = {}

    def _base_positions(self) -> np.ndarray:
        """
        Базовые позиции импульсов внутри одного периода.
        """
        margin = max(5, self.pulse_width + 2)
        if self.num_pulses == 1:
            return np.array([self.period // 2], dtype=int)

        positions = np.linspace(
            margin,
            self.period - margin - self.pulse_width,
            self.num_pulses,
            dtype=int,
        )
        return positions

    def _build_packet(self, positions: np.ndarray) -> np.ndarray:
        packet = np.zeros(self.period, dtype=float)

        for pos in positions:
            start = max(0, int(pos))
            end = min(self.period, start + self.pulse_width)
            packet[start:end] = self.amplitude

        return packet

    def generate_regular(self) -> np.ndarray:
        """
        Идеальный периодический сигнал без смещений.
        """
        positions = self._base_positions()
        packet = self._build_packet(positions)
        signal = np.tile(packet, self.repeats)

        self.last_packet = packet
        self.last_signal = signal
        self.last_mode = "regular"
        self.last_params = {}

        return signal

    def generate_with_jitter(self, jitter_std: float) -> np.ndarray:
        """
        Периодический сигнал с гауссовым смещением импульсов
        внутри КАЖДОГО периода.
        """
        signal = np.zeros(self.period * self.repeats, dtype=float)
        base_positions = self._base_positions()

        for repeat_idx in range(self.repeats):
            packet_start = repeat_idx * self.period
            noise = self.rng.normal(0.0, jitter_std, size=self.num_pulses)

            positions = np.clip(
                np.round(base_positions + noise).astype(int),
                0,
                self.period - self.pulse_width,
            )

            for pos in positions:
                start = packet_start + pos
                end = start + self.pulse_width
                signal[start:end] = self.amplitude

        self.last_packet = None
        self.last_signal = signal
        self.last_mode = "jitter"
        self.last_params = {"jitter_std": jitter_std}

        return signal

    def generate_from_mode(self, mode: str = "regular", **kwargs) -> np.ndarray:
        """
        Унифицированный интерфейс.
        """
        if mode == "regular":
            return self.generate_regular()

        if mode == "jitter":
            jitter_std = kwargs.get("jitter_std", 1.0)
            return self.generate_with_jitter(jitter_std=jitter_std)

        raise ValueError(f"Unsupported mode: {mode}")

    def stream_samples(self, signal: np.ndarray | None = None):
        """
        Поэлементная выдача сигнала.
        """
        if signal is None:
            if self.last_signal is None:
                raise ValueError("No generated signal available.")
            signal = self.last_signal

        for sample in signal:
            yield sample

    def stream_batches(self, batch_size: int, signal: np.ndarray | None = None):
        """
        Батчевая выдача сигнала.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if signal is None:
            if self.last_signal is None:
                raise ValueError("No generated signal available.")
            signal = self.last_signal

        for i in range(0, len(signal), batch_size):
            yield signal[i:i + batch_size]