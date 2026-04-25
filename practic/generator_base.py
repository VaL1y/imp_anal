from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class PacketGeneratorConfig:
    """Configures construction of a repeatable impulse packet."""

    packet_length: int
    impulses_per_packet: int
    impulse_width: int
    amplitude_range: tuple[float, float]
    baseline: float
    repeats: int
    seed: int | None = None


@dataclass(slots=True)
class PacketPulse:
    """A single impulse inside the generated packet."""

    start_index: int
    width: int
    amplitude: float


@dataclass(slots=True)
class JitterConfig:
    """Defines how much jitter to inject into a generated timeline."""

    resolution_ms: float
    sigma_ms: float
    seed: int | None = None


def build_random_packet(config: PacketGeneratorConfig) -> tuple[np.ndarray, list[PacketPulse]]:
    """Builds one random pulse packet and returns the waveform and pulse descriptions."""

    if config.packet_length <= 0:
        raise ValueError("packet_length must be positive")
    if config.impulses_per_packet <= 0:
        raise ValueError("impulses_per_packet must be positive")
    if config.impulse_width <= 0:
        raise ValueError("impulse_width must be positive")

    rng = np.random.default_rng(config.seed)
    packet = np.full(config.packet_length, config.baseline, dtype=float)

    available_slots = max(1, config.packet_length - config.impulse_width + 1)
    picks = min(config.impulses_per_packet, available_slots)
    starts = rng.choice(available_slots, size=picks, replace=False)
    starts.sort()

    pulses: list[PacketPulse] = []
    for start in starts:
        amplitude = float(rng.uniform(*config.amplitude_range))
        end = min(config.packet_length, start + config.impulse_width)
        packet[start:end] = amplitude
        pulses.append(PacketPulse(start, end - start, amplitude))

    return packet, pulses


def repeat_packet(packet: np.ndarray, repeats: int) -> np.ndarray:
    """Stacks identical packets in time to create a quasi-periodic signal."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return np.tile(packet, repeats)


def pulse_offsets_in_time(
    pulses: Sequence[PacketPulse],
    repeats: int,
    packet_length: int,
    resolution_ms: float,
) -> list[float]:
    """Converts pulse starts into time stamps (ms) for each repetition."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if packet_length <= 0:
        raise ValueError("packet_length must be positive")

    timeline: list[float] = []
    period_ms = packet_length * resolution_ms
    for packet_index in range(repeats):
        base_ms = packet_index * period_ms
        for pulse in pulses:
            timeline.append(base_ms + pulse.start_index * resolution_ms)
    return timeline


def jittered_times(
    base_times_ms: Sequence[float],
    config: JitterConfig,
) -> list[float]:
    """Applies Gaussian timing jitter to a sequence of timestamps."""

    rng = np.random.default_rng(config.seed)
    return [float(t + rng.normal(0.0, config.sigma_ms)) for t in base_times_ms]


def rasterize_times(
    times_ms: Sequence[float],
    resolution_ms: float,
    duration_ms: float | None = None,
) -> np.ndarray:
    """Converts time stamps into a digital waveform by binning them into a fixed grid."""

    if resolution_ms <= 0:
        raise ValueError("resolution_ms must be positive")
    if not times_ms and duration_ms is None:
        return np.zeros(0, dtype=float)

    max_time = duration_ms if duration_ms is not None else (max(times_ms) if times_ms else 0.0)
    length = int(np.ceil(max_time / resolution_ms)) + 1
    signal = np.zeros(length, dtype=float)

    for time_ms in times_ms:
        if time_ms < 0:
            continue
        idx = int(round(time_ms / resolution_ms))
        if 0 <= idx < length:
            signal[idx] += 1.0

    return signal
