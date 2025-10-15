import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import os


def lpc_formant_modification(sig, Fs, shift_percent):

    Horizon = 30  # 30ms - window length
    OrderLPC = 24  # LPC order
    Buffer = 0  # initialization
    out = np.zeros_like(sig)  # output signal

    Horizon = int(Horizon * Fs / 1000)
    Shift = int(Horizon / 2)  # frame size - step size
    Win = np.hanning(Horizon)  # analysis window

    Lsig = len(sig)
    slice_start = 0
    tosave_start = 0
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1)  # number of frames

    # analysis frame-by-frame
    for l in range(Nfr):
        slice_end = slice_start + Horizon
        tosave_end = tosave_start + Shift

        sigLPC = Win * sig[slice_start:slice_end]
        en = np.sum(sigLPC ** 2)

        # --- LPC analysis ---
        r = np.correlate(sigLPC, sigLPC, mode='full')
        r = r[len(sigLPC) - 1:]  # keep positive lags
        a = my_levinson(r, OrderLPC)  # LPC coefficients - this is YOUR function
        G = np.sqrt(np.sum(a * r[:len(a)]))  # gain
        ex = lfilter(a, [1], sigLPC) # inverse filter - use lfilter

        # --- Pole analysis ---
        poles = np.roots(a)
        complex_poles = [p for p in poles if np.imag(p) > 0 and np.abs(p) < 1.0]
        complex_poles = sorted(complex_poles, key=lambda p: np.abs(p), reverse=True)
        selected = complex_poles[:3]  # 3 strongest poles

        # Modify their angles
        modified = []
        for p in selected:
            mag = np.abs(p)
            ang = np.angle(p)
            new_ang = ang * (1 + shift_percent / 100.0)
            new_ang = np.clip(new_ang, 0, np.pi)
            modified.append(mag * np.exp(1j * new_ang))

        # Replace original formant poles
        new_poles = poles.copy()
        tol = 1e-6
        for orig, mod in zip(selected, modified):
            for j, p in enumerate(new_poles):
                if np.abs(p - orig) < tol:
                    new_poles[j] = mod
                elif np.abs(p - np.conj(orig)) < tol:
                    new_poles[j] = np.conj(mod)

        # --- Rebuild LPC ---
        a_new = np.poly(new_poles)
        a_new = np.real(a_new / a_new[0])  # normalize

        # --- Synthesis ---
        s = lfilter([G], a_new, ex)  # synthesis with modified formants
        ens = np.sum(s ** 2)
        g = np.sqrt(en / (ens + 1e-12))  # normalization factor
        s = s * g  # energy compensation

        s[:Shift] = s[:Shift] + Buffer  # overlap-add
        out[tosave_start:tosave_end] = s[:Shift]  # save the first part of the frame
        Buffer = s[Shift:Horizon]  # buffer the rest of the frame

        slice_start += Shift  # Move the frame
        tosave_start += Shift

    return out


if __name__ == "__main__":
    # Example usage
    Fs, sig = wavfile.read("speechsample.wav")
    sig = sig / np.max(np.abs(sig)) # normalize the signal

    os.makedirs("formant_output", exist_ok=True)

    transforms = [
        (-20, "elderly_minus20.wav"),
        (-10, "elderly_minus10.wav"),
        (+10, "younger_plus10.wav"),
        (+20, "younger_plus20.wav")
    ]

    for shift, filename in transforms:
        modified = lpc_formant_modification(sig, Fs, shift)
        wavfile.write(f"formant_output/{filename}", Fs, np.int16(modified * 32767))

