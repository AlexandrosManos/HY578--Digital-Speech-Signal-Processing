import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import os


def lpc_formant_modification(sig, Fs, shift_percent):
    """Modify formants by shifting pole angles."""
    
    # Frame parameters
    Horizon = int(30 * Fs / 1000)  # 30ms frame
    Shift = Horizon // 2
    Win = np.hanning(Horizon)
    OrderLPC = 24
    
    # Initialize
    Nframes = (len(sig) - Horizon) // Shift + 1
    out = np.zeros_like(sig)
    Buffer = np.zeros(Shift)
    
    for i in range(Nframes):
        start = i * Shift
        frame = sig[start:start + Horizon] * Win
        en = np.sum(frame ** 2)
        
        # LPC analysis
        r = np.correlate(frame, frame, mode='full')[len(frame)-1:]
        a = my_levinson(r, OrderLPC)
        G = np.sqrt(np.sum(a * r[:len(a)]))
        ex = lfilter(a, [1], frame)
        
        # Get 3 strongest formant poles
        poles = np.roots(a)
        formants = sorted([p for p in poles if 0 < np.imag(p) and np.abs(p) < 1],
                         key=lambda p: np.abs(p), reverse=True)[:3]
        
        # Shift formant frequencies
        new_poles = poles.copy()
        for orig in formants:
            new_ang = np.clip(np.angle(orig) * (1 + shift_percent/100), 0, np.pi)
            modified = np.abs(orig) * np.exp(1j * new_ang)
            # Replace original and conjugate
            for j, p in enumerate(new_poles):
                if np.abs(p - orig) < 1e-6:
                    new_poles[j] = modified
                elif np.abs(p - np.conj(orig)) < 1e-6:
                    new_poles[j] = np.conj(modified)
        
        # Rebuild LPC and synthesize
        a_new = np.real(np.poly(new_poles))
        a_new /= a_new[0]
        s = lfilter([G], a_new, ex)
        s *= np.sqrt(en / (np.sum(s**2) + 1e-12))
        
        # Overlap-add
        s[:Shift] += Buffer
        out[start:start + Shift] = s[:Shift]
        Buffer = s[Shift:]
    
    return out


if __name__ == "__main__":
    Fs, sig = wavfile.read("goofy.wav")
    sig = sig / np.max(np.abs(sig))
    
    os.makedirs("formant_output", exist_ok=True)
    
    shifts = [(-20, "elderly_minus20"), (-10, "elderly_minus10"),
              (+10, "younger_plus10"), (+20, "younger_plus20")]
    
    for shift, name in shifts:
        modified = lpc_formant_modification(sig, Fs, shift)
        wavfile.write(f"formant_output/{name}.wav", Fs, (modified * 32767).astype(np.int16))
        print(f"Saved {name}.wav (shift: {shift:+d}%)")
