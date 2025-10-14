import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import soundfile as sf
def lpc_whisper(sig, Fs, save_path=None):
    np.random.seed(42)

    Horizon = 30  # 30 ms
    OrderLPC = 24  # order of LPC 12, 24, 48
    Buffer = 0  # initialization
    out = np.zeros_like(sig)  # initialization

    Horizon = int(Horizon * Fs / 1000)
    Shift = int(Horizon / 2)
    Win = np.hanning(Horizon)
    # print(Horizon) 480
    Lsig = len(sig)
    slice_start = 0
    tosave_start = 0
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1)

    i = 0

    for l in range(Nfr):
        slice_end = slice_start + Horizon
        tosave_end = tosave_start + Shift

        sigLPC = Win * sig[slice_start:slice_end]

        en = np.sum(sigLPC ** 2)  # get the short-term energy of the input

        # --- LPC analysis ---
        r = np.correlate(sigLPC, sigLPC, mode='full')  # correlation mode {‘valid’, ‘same’, ‘full’} # line 50  # without mode = 'full' we get [0.00328293] # lags = np.arange(-len(sigLPC) + 1, len(sigLPC))  # r = r[lags >= 0]
        r = r[len(sigLPC) - 1:]  # keep only the positive lags     # line 51
        a = my_levinson(r, OrderLPC)  # LPC coefficients - this is YOUR function # line 52
        G = np.sqrt(np.sum(a * r[:len(a)]))  # gain    # line 53
        ex = np.random.randn(Horizon)  # inverse filter - use lfilter # line 54

        # --- synthesis ---
        s = lfilter([G], a, ex)
        ens = np.sum(s ** 2)  # short-time energy of output
        g = np.sqrt(en / (ens + 1e-12))  # normalization factor
        s = s * g  # energy compensation

        s[:Shift] = s[:Shift] + Buffer  # overlap-add
        out[tosave_start:tosave_end] = s[:Shift]  # save the first part of the frame
        Buffer = s[Shift:Horizon]  # buffer the rest of the frame

        slice_start += Shift  # Move the frame
        tosave_start += Shift

    sf.write(save_path, out, Fs)
    return out

if __name__ == "__main__":
    Fs, sig = wavfile.read('speechsample.wav')
    sig = sig / np.max(np.abs(sig))  # normalize the signal

    out = lpc_whisper(sig, Fs, "whisper.wav")

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(sig)
    plt.title('Original Signal')
    plt.subplot(2, 1, 2)
    plt.plot(out)
    plt.title('LPC Synthesized Signal')
    plt.tight_layout()
    plt.show()
    # plt.savefig("plot.png")

    sd.play(out, Fs)
    sd.wait()
    sd.play(sig, Fs)
    sd.wait()
    sd.play(np.concatenate((out, np.zeros(2000), sig[:-2000])), Fs)  # create echo
    sd.wait()
