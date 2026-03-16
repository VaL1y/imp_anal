import numpy as np


def generate_packet(packet_length,
                    num_pulses,
                    pulse_width,
                    jitter_std=0):
    """
    Генерация одной пачки импульсов
    """

    signal = np.zeros(packet_length)

    base_positions = np.linspace(
        50,
        packet_length - 50,
        num_pulses,
        dtype=int
    )

    for pos in base_positions:

        jitter = int(np.random.normal(0, jitter_std))

        p = pos + jitter
        p = max(0, min(packet_length - pulse_width, p))

        signal[p:p + pulse_width] = 1

    return signal


def generate_signal(period,
                    repeats,
                    num_pulses,
                    pulse_width,
                    jitter_std=0):
    """
    Генерирует полный сигнал
    """

    packet = generate_packet(
        packet_length=period,
        num_pulses=num_pulses,
        pulse_width=pulse_width,
        jitter_std=jitter_std
    )

    signal = np.tile(packet, repeats)

    return signal