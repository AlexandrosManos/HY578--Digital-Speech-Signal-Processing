import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import sounddevice as sd
import matplotlib.pyplot as plt
import os

def lpc_whisper(sig, Fs, OrderLPC=24, seed=40):
    """LPC whisper synthesis with white-noise excitation."""
    np.random.seed(seed)

    frame_len = int(30 * Fs / 1000)
    shift = frame_len // 2
    window = np.hanning(frame_len)
    Nframes = (len(sig) - frame_len) // shift
    out = np.zeros(len(sig))

    for i in range(Nframes):
        start = i * shift
        frame = sig[start:start + frame_len] * window

        # --- LPC analysis ---
        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame)-1:]
        a = my_levinson(r, OrderLPC)

        # LPC gain (prediction error energy)
        G = np.sqrt(np.sum(a * r[:len(a)]))

        # --- Whisper synthesis ---
        noise = np.random.randn(frame_len)
        whisper_frame = G*lfilter([1], a, noise)

        # Overlap-add
        out[start:start + frame_len] += whisper_frame * window

    # Normalize amplitude
    out /= np.max(np.abs(out) + 1e-12)
    return out

def plot_spectrogram(sig, Fs, title, filename):
    """Helper to show and save spectrograms."""
    plt.figure(figsize=(10, 5))
    plt.specgram(sig, NFFT=512, Fs=Fs, noverlap=256, cmap='magma')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    os.makedirs("whisper_output", exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()

if __name__ == "__main__":
    Fs, sig = wavfile.read("speechsample.wav")
    sig = sig / np.max(np.abs(sig))

    os.makedirs("whisper_output", exist_ok=True)
    for order in [12, 24, 36]:
        whisper = lpc_whisper(sig, Fs, order)
        wavfile.write(f"whisper_output/whisper_order{order}.wav",
                      Fs, (whisper * 32767).astype(np.int16))
        print(f"Saved whisper_order{order}.wav")
        filename_png = f"whisper_output/whisper_order{order}_spectrogram.png"
        plot_spectrogram(whisper, Fs, f"LPC Whisper (Order {order})", filename_png)
        print(f"   → Saved spectrogram: {filename_png}")

    # Plot the waveform of each whisper file for orders 12, 24, 36

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    for idx, order in enumerate([12, 24, 36]):
        wav_path = f"whisper_output/whisper_order{order}.wav"
        Fs, whisper_sig = wavfile.read(wav_path)
        whisper_sig = whisper_sig / (np.max(np.abs(whisper_sig)) + 1e-12)
        t = np.linspace(0, len(whisper_sig) / Fs, len(whisper_sig))
        axs[idx].plot(t, whisper_sig)
        axs[idx].set_title(f"Whispered Signal (Order {order})")
        axs[idx].set_xlabel("Time (s)")
        axs[idx].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()