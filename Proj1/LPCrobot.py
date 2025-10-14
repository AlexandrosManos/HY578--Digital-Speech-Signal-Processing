"""
Robotic Voice Generation using LPC with Artificial Periodic Excitation
Replaces natural excitation with constant pitch period pulse train
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
import matplotlib.pyplot as plt


def create_periodic_excitation(frame_length, pitch_period):
    """
    Create artificial excitation with periodic impulses.
    
    Args:
        frame_length: length of the frame (N)
        pitch_period: distance between impulses (controls pitch)
    
    Returns:
        excitation signal with impulses spaced by pitch_period
    """
    excitation = np.zeros(frame_length)
    # Place impulses at regular intervals
    excitation[::pitch_period] = 1.0
    return excitation


def lpc_robot_voice(sig, Fs, lpc_order, pitch_period):
    """
    LPC synthesis with artificial periodic excitation for robotic voice.
    
    Args:
        sig: input speech signal
        Fs: sampling frequency
        lpc_order: order of LPC filter
        pitch_period: period of artificial pitch (in samples)
    
    Returns:
        synthesized robotic speech
    """
    # Frame parameters
    frame_len = int(30 * Fs / 1000)  # 30ms window
    shift = frame_len // 2
    window = np.hanning(frame_len)
    
    out = np.zeros_like(sig)
    buffer = 0
    
    # Process frame by frame
    for i in range(0, len(sig) - frame_len, shift):
        # Windowed frame
        frame = window * sig[i:i + frame_len]
        energy = np.sum(frame ** 2)
        
        # LPC analysis
        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame) - 1:]  # keep positive lags
        a = my_levinson(r, lpc_order)
        G = np.sqrt(np.sum(a * r[:len(a)]))
        
        # --- MODIFIED: Use artificial periodic excitation instead of natural residual ---
        # Original line 58: ex = lfilter(a, [1], frame)  # natural excitation
        # New: Create periodic pulse train
        ex = create_periodic_excitation(frame_len, pitch_period)
        
        # Synthesis with artificial excitation
        s = lfilter([G], a, ex)
        
        # Energy normalization
        synth_energy = np.sum(s ** 2)
        if synth_energy > 0:
            s = s * np.sqrt(energy / synth_energy)
        
        # Overlap-add
        s[:shift] = s[:shift] + buffer
        out[i:i + shift] = s[:shift]
        buffer = s[shift:]
    
    return out


def create_spectrograms(original, robot_signals, Fs):
    """Create spectrograms comparing original and robotic voices."""
    
    # 1. Original vs Robot (baseline)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].specgram(original, Fs=Fs, NFFT=512, noverlap=256, cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Original Speech')
    axes[0].set_ylim([0, 4000])
    
    axes[1].specgram(robot_signals['baseline'], Fs=Fs, NFFT=512, noverlap=256, cmap='viridis')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Robot Voice (Order 24, Pitch 200 Hz)')
    axes[1].set_ylim([0, 4000])
    
    plt.tight_layout()
    plt.savefig('robot_output/spectrogram_original_vs_robot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: robot_output/spectrogram_original_vs_robot.png")
    
    # 2. LPC Order Comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    titles = ['Robot Voice - Order 12', 'Robot Voice - Order 24', 'Robot Voice - Order 36']
    signals = [robot_signals['order12'], robot_signals['order24'], robot_signals['order36']]
    
    for ax, sig, title in zip(axes, signals, titles):
        ax.specgram(sig, Fs=Fs, NFFT=512, noverlap=256, cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        ax.set_ylim([0, 4000])
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('robot_output/spectrogram_order_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: robot_output/spectrogram_order_comparison.png")
    
    # 3. Pitch Period Comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    titles = ['Robot Voice - High Pitch (320 Hz)', 
              'Robot Voice - Medium Pitch (200 Hz)', 
              'Robot Voice - Low Pitch (133 Hz)']
    signals = [robot_signals['pitch50'], robot_signals['pitch80'], robot_signals['pitch120']]
    
    for ax, sig, title in zip(axes, signals, titles):
        ax.specgram(sig, Fs=Fs, NFFT=512, noverlap=256, cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        ax.set_ylim([0, 4000])
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('robot_output/spectrogram_pitch_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: robot_output/spectrogram_pitch_comparison.png")


if __name__ == "__main__":
    import os
    
    # Load speech
    Fs, sig = wavfile.read('speechsample.wav')
    sig = sig / np.max(np.abs(sig))
    
    print("Robotic Voice Generation using LPC")
    print("=" * 70)
    print(f"Input: speechsample.wav ({len(sig)/Fs:.2f}s, {Fs}Hz)")
    print()
    
    # Create output directory
    os.makedirs("robot_output", exist_ok=True)
    
    # Define configurations: (lpc_order, pitch_period, filename, description)
    # Pitch period: controls the robot voice pitch
    # - Smaller period (40-60 samples) = higher pitch
    # - Larger period (80-120 samples) = lower pitch
    
    configs = [
        (12, 80, "robot_output/robot_order12_pitch80.wav", "Order 12, Pitch period 80 samples", 'order12'),
        (24, 80, "robot_output/robot_order24_pitch80.wav", "Order 24, Pitch period 80 samples (baseline)", 'baseline'),
        (36, 80, "robot_output/robot_order36_pitch80.wav", "Order 36, Pitch period 80 samples", 'order36'),
        (24, 50, "robot_output/robot_order24_pitch50.wav", "Order 24, Pitch period 50 samples (higher pitch)", 'pitch50'),
        (24, 120, "robot_output/robot_order24_pitch120.wav", "Order 24, Pitch period 120 samples (lower pitch)", 'pitch120'),
    ]
    
    # Store robot signals for spectrogram generation
    robot_signals = {}
    
    # Process each configuration
    for order, pitch, filename, description, key in configs:
        print(f"Processing: {description}")
        robot_voice = lpc_robot_voice(sig, Fs, order, pitch)
        robot_signals[key] = robot_voice
        if key == 'baseline':
            robot_signals['order24'] = robot_voice
            robot_signals['pitch80'] = robot_voice
        wavfile.write(filename, Fs, np.int16(robot_voice * 32767))
        print(f"  → Saved: {filename}")
        print(f"     LPC order: {order}, Pitch: {pitch} samples ({Fs/pitch:.1f} Hz)")
        print()
    
    print("=" * 70)
    print("Complete! Generated 5 robotic voice files.")
    print()
    print("OBSERVATIONS:")
    print("-" * 70)
    print()
    print("Effect of LPC Order:")
    print("  • Order 12 (lower): Less spectral detail, more 'buzzy' robotic sound")
    print("  • Order 24 (baseline): Good balance of formant preservation")
    print("  • Order 36 (higher): More natural formants, less robotic effect")
    print()
    print("Effect of Pitch Period:")
    print("  • 50 samples (~320 Hz): Higher pitched robot voice")
    print("  • 80 samples (~200 Hz): Medium pitched robot voice")
    print("  • 120 samples (~133 Hz): Lower pitched robot voice")
    print()
    print("DESIGN CHOICES:")
    print("-" * 70)
    print()
    print("1. Pitch Period Selection:")
    print("   - Used 80 samples as baseline (~200 Hz at 16kHz)")
    print("   - This gives a neutral robotic pitch similar to classic robot voices")
    print("   - Tested 50 (high) and 120 (low) for comparison")
    print()
    print("2. Pulse Train Construction:")
    print("   - Used single-sample impulses spaced by pitch_period")
    print("   - Alternative: could use k=2-4 consecutive ones, but single impulses")
    print("     produce clearer robotic effect with sharper harmonics")
    print()
    print("3. LPC Order Comparison:")
    print("   - Order 12: Insufficient for capturing all formants → more robotic")
    print("   - Order 24: Standard order (Fs/1000 + ~16) → balanced")
    print("   - Order 36: Higher order → captures more detail, less robotic")
    print()
    print("4. Energy Normalization:")
    print("   - Maintained to ensure consistent loudness across frames")
    print("   - Important because artificial excitation has different energy than natural")
    print()
    print("Key Insight:")
    print("  The robotic effect comes from replacing the natural pitch variation")
    print("  with a constant periodic excitation. The LPC filter preserves formant")
    print("  structure (vowel quality) while the periodic excitation removes natural")
    print("  prosody and pitch variation, creating the characteristic robot sound.")
    print()
    print("=" * 70)
    print("GENERATING SPECTROGRAMS")
    print("-" * 70)
    create_spectrograms(sig, robot_signals, Fs)
    print()
    print("=" * 70)
    print("All done! Generated 5 audio files + 3 spectrograms in robot_output/")

