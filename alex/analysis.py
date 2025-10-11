import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.io import wavfile
from my_levinson import my_levinson


def calculate_zcr(frame):
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))


def freq_exp(sig, Fs, OrderLPC):

    Horizon = 30  # 30 ms

    Horizon = int(Horizon * Fs / 1000)
    Shift = int(Horizon / 2)
    Win = np.hanning(Horizon)

    Lsig = len(sig)
    slice_start = 0
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1)

    # Fast Fourier Transform
    SFFT = 1024 # 1024, 2048, 4096
    freq = np.arange(0, Fs/2, Fs/SFFT)

    """""
    freq[0] = 0 Hz
    freq[1] = Fs / NFFT
    freq[2] = 2 * Fs / NFFT
    ...
    freq[NFFT // 2 - 1] ≈ Fs / 2 - Fs / NFFT
    """""

    has_plotted_voiced = False
    has_plotted_unvoiced = False
    comp = 0.15

    for l in range(Nfr):
        slice_end = slice_start + Horizon

        sigLPC = Win * sig[slice_start:slice_end]

        # Autocorrelation
        r = np.correlate(sigLPC, sigLPC, mode='full')
        r = r[len(sigLPC) - 1:]  # positive lags only
        a = my_levinson(r, OrderLPC)
        G = np.sqrt(np.sum(a * r[:len(a)]))

        zcr = calculate_zcr(sigLPC)
        is_voiced = zcr < comp

        # Voiced frame example sounds with clear pitch and harmonic structure
        # if l == 166: pre-determined frame numbers
        if is_voiced and not has_plotted_voiced:
            has_plotted_voiced = True
            # Compute the frequency response of the LPC filter
            w, h = freqz(G, a, worN=SFFT, whole=True)
            X = np.fft.fft(sigLPC, SFFT)

            plt.figure()
            plt.plot(freq, 20 * np.log10(np.abs(h[:SFFT // 2])), linewidth=2)
            plt.plot(freq, 20 * np.log10(np.abs(X[:SFFT // 2])), linewidth=1)
            plt.grid(True)
            plt.title(f'Voiced Frame in Frequency Domain with orderLPC: {OrderLPC}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.legend(['Magnitude of LP filter', 'Magnitude of FFT of frame'])
            plt.show()

        # Unvoiced frame example sounds like "s", "sh", "f" with noise-like characteristics
        # if l == 0: pre-determined frame numbers
        if not is_voiced and not has_plotted_unvoiced:
            has_plotted_unvoiced = True
            w, h = freqz(G, a, worN=SFFT, whole=True)
            X = np.fft.fft(sigLPC, SFFT)

            plt.figure()
            plt.plot(freq, 20 * np.log10(np.abs(h[:SFFT // 2])), linewidth=2)
            plt.plot(freq, 20 * np.log10(np.abs(X[:SFFT // 2])), linewidth=1)
            plt.grid(True)
            plt.title(f'Unvoiced Frame in Frequency Domain with orderLPC: {OrderLPC}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.legend(['Magnitude of LP filter', 'Magnitude of FFT of frame'])
            plt.show()

        slice_start += Shift

if __name__ == "__main__":
    Fs, sig = wavfile.read('speechsample.wav')
    # Fs = 16000, we should use OrderLPC = 16 - 24
    sig = sig / np.max(np.abs(sig))
    OrderLPC = 68
    freq_exp(sig, Fs, OrderLPC)