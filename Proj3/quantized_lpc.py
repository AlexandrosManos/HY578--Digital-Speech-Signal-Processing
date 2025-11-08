import os
import sys

import numpy as np
from scipy.signal import lfilter

from scalar_quantizer import uniform_scalar_quantize

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROJ1_DIR = os.path.join(ROOT_DIR, "Proj1")
if PROJ1_DIR not in sys.path:
    sys.path.append(PROJ1_DIR)

from my_levinson import my_levinson  # noqa: E402


# Extract LPC gains frame-by-frame without quantisation.
def extract_lpc_gains(sig, Fs, order=24, frame_ms=30, window_fn=np.hanning):
    horizon = int(frame_ms * Fs / 1000)
    shift = horizon // 2
    window = window_fn(horizon)
    total_frames = int(np.floor((len(sig) - horizon) / shift) + 1)
    gains = []

    slice_start = 0
    for _ in range(total_frames):
        slice_end = slice_start + horizon
        sig_frame = sig[slice_start:slice_end] * window
        r = np.correlate(sig_frame, sig_frame, mode="full")
        r = r[len(sig_frame) - 1:]
        a = my_levinson(r, order)
        gain = np.sqrt(np.sum(a * r[:len(a)]))
        gains.append(gain)
        slice_start += shift

    return np.asarray(gains, dtype=np.float64)


# LPC analysis/synthesis using a quantised gain value per frame.
def lpc_synthesis_with_quantized_gain(
    sig,
    Fs,
    bits,
    gain_min,
    gain_max,
    order=24,
    frame_ms=30,
    window_fn=np.hanning,
):
    horizon = int(frame_ms * Fs / 1000)
    shift = horizon // 2
    window = window_fn(horizon)

    out = np.zeros_like(sig)
    buffer = np.zeros(horizon - shift, dtype=np.float64)
    quantised_gains = []

    slice_start = 0
    tosave_start = 0
    total_frames = int(np.floor((len(sig) - horizon) / shift) + 1)

    for frame_id in range(total_frames):
        slice_end = slice_start + horizon
        tosave_end = tosave_start + shift

        sig_frame = sig[slice_start:slice_end] * window
        autocorr = np.correlate(sig_frame, sig_frame, mode="full")
        autocorr = autocorr[len(sig_frame) - 1:]
        a = my_levinson(autocorr, order)
        gain = np.sqrt(np.sum(a * autocorr[:len(a)]))
        q_gain, q_index, step = uniform_scalar_quantize(gain, bits, gain_min, gain_max)
        quantised_gains.append((frame_id, gain, q_gain, q_index))

        excitation = lfilter(a, [1], sig_frame)
        synthesis = lfilter([q_gain], a, excitation)

        # ens = np.sum(synthesis ** 2)
        # g = np.sqrt(np.sum(sig_frame ** 2) / (ens + 1e-12))
        # synthesis = synthesis * g

        synthesis[:shift] += buffer
        out[tosave_start:tosave_end] = synthesis[:shift]
        buffer = synthesis[shift:horizon]

        slice_start += shift
        tosave_start += shift

    return out, quantised_gains, step
