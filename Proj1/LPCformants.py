"""
Formant Modification for Voice Transformation
Modifies formant frequencies by changing pole angles to create younger/elderly voices
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import os

def select_formant_poles(poles, num_formants=3):
    """Select the 3 most significant poles (by magnitude) with positive imaginary part."""
    # Keep only poles inside unit circle with positive imaginary part
    valid_poles = [p for p in poles if np.abs(p) < 1.0 and np.imag(p) > 0]
    # Sort by magnitude (descending) and select top 3
    valid_poles = sorted(valid_poles, key=lambda p: np.abs(p), reverse=True)
    return np.array(valid_poles[:num_formants])


def modify_formant_poles(poles, shift_percent):
    """Modify pole angles by shift_percent to change formant frequencies."""
    modified = []
    for pole in poles:
        magnitude = np.abs(pole)
        angle = np.angle(pole)
        new_angle = angle * (1 + shift_percent / 100.0)
        new_angle = np.clip(new_angle, 0, np.pi)
        modified.append(magnitude * np.exp(1j * new_angle))
    return np.array(modified)


def reconstruct_filter(original_poles, modified_formant_poles, original_formant_poles):
    """Reconstruct LPC filter with modified formant poles."""
    new_poles = original_poles.copy()
    tolerance = 1e-6
    
    # Replace formant poles and their conjugates
    for orig_pole, mod_pole in zip(original_formant_poles, modified_formant_poles):
        for idx, pole in enumerate(new_poles):
            if np.abs(pole - orig_pole) < tolerance:
                new_poles[idx] = mod_pole
            elif np.abs(pole - np.conj(orig_pole)) < tolerance:
                new_poles[idx] = np.conj(mod_pole)
    
    # Reconstruct polynomial using np.convolve
    a_new = np.array([1.0])
    for pole in new_poles:
        a_new = np.convolve(a_new, [1, -pole])
    
    return np.real(a_new)


def lpc_formant_modification(sig, Fs, shift_percent):
    """Apply formant modification to speech signal."""
    # Frame parameters
    frame_len = int(30 * Fs / 1000)  # 30ms
    shift = frame_len // 2
    window = np.hanning(frame_len)
    order = 24
    
    out = np.zeros_like(sig)
    buffer = 0
    
    # Process frame by frame
    for i in range(0, len(sig) - frame_len, shift):
        # LPC analysis
        frame = window * sig[i:i+frame_len]
        energy = np.sum(frame ** 2)
        
        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame)-1:]
        a = my_levinson(r, order)
        G = np.sqrt(np.sum(a * r[:len(a)]))
        
        # Pole extraction and modification
        poles = np.roots(a)
        formant_poles = select_formant_poles(poles)
        modified_poles = modify_formant_poles(formant_poles, shift_percent)
        a_new = reconstruct_filter(poles, modified_poles, formant_poles)
        
        # Ensure same length
        if len(a_new) > len(a):
            a_new = a_new[:len(a)]
        elif len(a_new) < len(a):
            a_new = np.pad(a_new, (0, len(a) - len(a_new)))
        
        # Synthesis
        excitation = lfilter(a, [1], frame)
        synthesized = lfilter([G], a_new, excitation)
        
        # Energy normalization
        synth_energy = np.sum(synthesized ** 2)
        if synth_energy > 0:
            synthesized *= np.sqrt(energy / synth_energy)
        
        # Overlap-add
        synthesized[:shift] += buffer
        out[i:i+shift] = synthesized[:shift]
        buffer = synthesized[shift:]
    
    return out


if __name__ == "__main__":
    # Load speech
    Fs, sig = wavfile.read('speechsample.wav')
    sig = sig / np.max(np.abs(sig))
    
    print("Formant Modification for Voice Transformation")
    print("=" * 60)
    print(f"Input: speechsample.wav ({len(sig)/Fs:.2f}s, {Fs}Hz)")
    
    # Define transformations
    transforms = [
        (-20, "elderly_minus20.wav", "Elderly (-20%)"),
        (-10, "elderly_minus10.wav", "Elderly (-10%)"),
        (+10, "younger_plus10.wav", "Younger (+10%)"),
        (+20, "younger_plus20.wav", "Younger (+20%)")
    ]
    os.makedirs("formant_output", exist_ok=True)
    # Process each transformation
    for shift, filename, description in transforms:
        print(f"\nProcessing {description}...")
        modified = lpc_formant_modification(sig, Fs, shift)
        wavfile.write(f"formant_output/{filename}", Fs, np.int16(modified * 32767))
        print(f"  Saved: {filename}")
    
    print("\n" + "=" * 60)
    print("Complete! Generated 4 audio files.")
    print("\nObservations:")
    print("- Negative shifts (-10%, -20%): Lower formants → elderly voice")
    print("- Positive shifts (+10%, +20%): Higher formants → younger voice")
    print("\nDesign choices:")
    print("- Selected 3 most significant poles (F1, F2, F3) by magnitude")
    print("- Modified pole angles (frequency) while preserving magnitude (bandwidth)")
    print("- Used np.convolve to reconstruct filter from modified poles")
    print("- Maintained conjugate pairs for real-valued coefficients")
