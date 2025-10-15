import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import sounddevice as sd
import matplotlib.pyplot as plt
import os

def lpc_whisper(sig, Fs, OrderLPC=24, seed=42):
    """LPC whisper synthesis with white-noise excitation."""
    np.random.seed(seed)

    # Frame parameters
    Horizon = int(30 * Fs / 1000)  # 30 ms frame
    Shift = Horizon // 2  # 50% overlap
    Win = np.hanning(Horizon)
    
    # Initialize
    Nframes = (len(sig) - Horizon) // Shift + 1
    out = np.zeros_like(sig)
    Buffer = np.zeros(Shift)
    
    for i in range(Nframes):
        start = i * Shift
        frame = sig[start:start + Horizon] * Win
        
        # LPC analysis
        r = np.correlate(frame, frame, mode='full')[len(frame)-1:] # keep only the positive lags    
        a = my_levinson(r, OrderLPC)
        G = np.sqrt(np.sum(a * r[:len(a)])) # gain   
        
        # Whisper synthesis with noise
        noise = np.random.randn(Horizon)
        whisper = G * lfilter([1], a, noise)
        
        # Energy compensation
        en = np.sum(frame ** 2)
        ens = np.sum(whisper ** 2)
        whisper *= np.sqrt(en / (ens))
        
        # Overlap-add
        whisper[:Shift] += Buffer
        out[start:start + Shift] = whisper[:Shift]
        Buffer = whisper[Shift:]
    
    return out / (np.max(np.abs(out)))

if __name__ == "__main__":
    Fs, sig = wavfile.read("speechsample.wav")
    sig = sig / np.max(np.abs(sig))
    
    # Generate whisper
    whisper = lpc_whisper(sig, Fs, OrderLPC=24)
    
    # Save output
    os.makedirs("whisper_output", exist_ok=True)
    wavfile.write("whisper_output/whisper.wav", Fs, (whisper * 32767).astype(np.int16))
    print("Saved whisper.wav")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sig)
    plt.title('Original Signal')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.plot(whisper)
    plt.title('Whispered Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('whisper_output/comparison.png', dpi=150)
    plt.show()
    
    # Play audio
    print("Playing original...")
    sd.play(sig, Fs)
    sd.wait()
    
    print("Playing whisper...")
    sd.play(whisper, Fs)
    sd.wait()
