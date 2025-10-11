import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, freqz
import sounddevice as sd
import matplotlib.pyplot as plt
import os

def lpc_as_toyou(sig, Fs):
    """
    INPUT:
        sig: input signal
        Fs: sampling frequency
    OUTPUT:
        out: a vector containing the output signal
    Example:
        Fs, sig = wavfile.read('speechsample.wav')
        sig = sig / np.max(np.abs(sig))  # normalize the signal
        out = lpc_as(sig, Fs)
        sounddevice.play(out, Fs)
        sounddevice.play(sig, Fs)
        sounddevice.play(np.concatenate((out, np.zeros(2000), sig[:-2000])), Fs) # create echo
    
    Yannis Stylianou (Python version by Alex Angelakis, 2025)
    CSD - CS 578
    """
    
    Horizon = 30   # 30ms - window length
    OrderLPC = 24  # order of LPC
    Buffer = 0     # initialization
    out = np.zeros_like(sig)  # initialization

    Horizon = int(Horizon * Fs / 1000)
    Shift = int(Horizon / 2)    # frame size - step size
    Win = np.hanning(Horizon)   # analysis window

    Lsig = len(sig)
    slice_start = 0
    tosave_start = 0
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1)  # number of frames

    # analysis frame-by-frame
    for l in range(Nfr):
        slice_end = slice_start + Horizon
        tosave_end = tosave_start + Shift

        sigLPC = Win * sig[slice_start:slice_end]
        en = np.sum(sigLPC ** 2)  # get the short-term energy of the input

        # --- LPC analysis ---
        r = np.correlate(sigLPC, sigLPC, mode='full')  #autocorrelation
        r = r[len(sigLPC)-1:] # keep only the positive lags, future samples and not the past ones
        a = my_levinson(r, OrderLPC)    # LPC coefficients - this is YOUR function
        G = np.sqrt(en / (np.sum(a ** 2))) # gain
        ex = lfilter([1], a, sigLPC) # inverse filter - use lfilter

        # --- synthesis ---
        s = lfilter([G], a, ex)
        ens = np.sum(s ** 2)            # short-time energy of output
        g = np.sqrt(en / (ens + 1e-12)) # normalization factor
        s = s * g                       # energy compensation
        
        s[:Shift] = s[:Shift] + Buffer            # overlap-add
        out[tosave_start:tosave_end] = s[:Shift]  # save the first part of the frame
        Buffer = s[Shift:Horizon]                 # buffer the rest of the frame

        slice_start += Shift    # Move the frame
        tosave_start += Shift

    return out

def my_levinson(r, order):
    """
    Lecture 4, Slide 33

    INPUT:
        r: autocorrelation sequence (numpy array), length >= order+1
           r[0] is the zero-lag autocorrelation
        order: order of the LPC filter (integer)
    
    OUTPUT:
        a: LPC coefficients (numpy array of length order+1)
           a[0] = 1.0 (by convention for the all-pole filter)
           a[1:] are the predictor coefficients
    """
    # Initial step: l₀⁰ = 0, E⁰ = r[0]
    # l[i, j] represents l_j^i (j-th coefficient at iteration i)
    l = np.zeros((order + 1, order + 1))
    
    # E[i] represents E^i (minimum squared prediction error at iteration i)
    E = np.zeros(order + 1)
    E[0] = r[0]
    
    # Iterate through orders i = 1, 2, ..., p
    for i in range(1, order + 1):
        # Step 1: Compute the partial correlation coefficient (reflection coefficient)
        # kᵢ = (r[i] - Σⱼ₌₁ⁱ⁻¹ lⱼⁱ⁻¹ r[i-j]) / Eⁱ⁻¹
        numerator = r[i]
        for j in range(1, i):
            numerator -= l[i-1, j] * r[i - j]
        k_i = numerator / E[i-1]
        
        # Step 2: Update prediction coefficients
        # lᵢⁱ = kᵢ
        l[i, i] = k_i
        
        # lⱼⁱ = lⱼⁱ⁻¹ - kᵢ lᵢ₋ⱼⁱ⁻¹, for 1 ≤ j ≤ i-1
        for j in range(1, i):
            l[i, j] = l[i-1, j] - k_i * l[i-1, i-j]
        
        # Step 3: Update the minimum squared prediction error
        # Eⁱ = (1 - kᵢ²) Eⁱ⁻¹
        E[i] = (1 - k_i**2) * E[i-1]
    
    # Final Step: Return optimal predictor coefficients as [1, -l₁*, -l₂*, ..., -lₚ*]
    # The negative sign is for the IIR filter representation H(z) = 1/A(z)
    # lⱼ* = lⱼᵖ for 1 ≤ j ≤ p
    a = np.zeros(order + 1)
    a[0] = 1.0
    a[1:] = -l[order, 1:order + 1]
    
    return a

def analyze_lpc_frame(sigLPC, a, Fs, frame_num, order, frame_type="unknown", save_plot=True):
    """
    Analyze and compare LPC filter frequency response with FFT of speech frame.
    
    INPUT:
        sigLPC: windowed speech frame
        a: LPC coefficients
        Fs: sampling frequency
        frame_num: frame number for labeling
        order: LPC order used
        frame_type: "voiced", "unvoiced", or "unknown"
        save_plot: whether to save the plot
    """
    # Compute FFT of speech frame
    fft_frame = np.fft.fft(sigLPC, n=2048)
    fft_freq = np.fft.fftfreq(2048, 1/Fs)
    fft_mag = np.abs(fft_frame)
    
    # Compute frequency response of LPC filter H(z) = 1/A(z)
    w, h = freqz([1], a, worN=2048, fs=Fs)
    lpc_mag = np.abs(h)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Magnitude spectrum comparison (linear scale)
    axes[0].plot(fft_freq[:1024], fft_mag[:1024], 'b-', alpha=0.7, linewidth=1, label='FFT of Speech Frame')
    axes[0].plot(w[:1024], lpc_mag[:1024], 'r-', linewidth=2, label=f'LPC Filter Response (order={order})')
    axes[0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[0].set_ylabel('Magnitude', fontsize=11)
    axes[0].set_title(f'Frame {frame_num} - {frame_type.upper()} - Magnitude Spectrum Comparison', 
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_xlim([0, Fs/2])
    
    # Plot 2: Magnitude spectrum in dB (log scale)
    fft_mag_db = 20 * np.log10(fft_mag[:1024] + 1e-10)
    lpc_mag_db = 20 * np.log10(lpc_mag[:1024] + 1e-10)
    axes[1].plot(fft_freq[:1024], fft_mag_db, 'b-', alpha=0.7, linewidth=1, label='FFT of Speech Frame')
    axes[1].plot(w[:1024], lpc_mag_db, 'r-', linewidth=2, label=f'LPC Filter Response (order={order})')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1].set_ylabel('Magnitude (dB)', fontsize=11)
    axes[1].set_title(f'Frame {frame_num} - {frame_type.upper()} - Magnitude Spectrum (dB)', 
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_xlim([0, Fs/2])
    axes[1].set_ylim([np.max(fft_mag_db) - 60, np.max(fft_mag_db) + 5])
    
    plt.tight_layout()
    
    if save_plot:
        # Create plots directory if it doesn't exist
        os.makedirs('lpc_analysis_plots', exist_ok=True)
        filename = f'lpc_analysis_plots/frame_{frame_num:04d}_{frame_type}_order_{order}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    return fig


    

def frame_by_frame_analysis(sig, Fs, lpc_orders=[8, 16, 24, 32, 40]):
    """
    Perform detailed frame-by-frame LPC analysis with different orders.
    Compare LPC filter response with FFT spectrum.
    """
    print("\n" + "="*70)
    print("FRAME-BY-FRAME LPC ANALYSIS")
    print("="*70)
    
    Horizon = 30   # 30ms - window length
    Horizon_samples = int(Horizon * Fs / 1000)
    Shift = int(Horizon_samples / 2)
    Win = np.hanning(Horizon_samples)
    
    Lsig = len(sig)
    slice_start = 0
    Nfr = int(np.floor((Lsig - Horizon_samples) / Shift) + 1)
    
    # Store frame energies and zero-crossing rates for classification
    frame_energies = []
    frame_zcr = []
    frame_data = []
    
    print(f"\nAnalyzing {Nfr} frames...")
    print(f"Frame length: {Horizon_samples} samples ({Horizon} ms)")
    print(f"Frame shift: {Shift} samples ({Shift*1000/Fs:.1f} ms)")
    
    # First pass: compute features for all frames
    for l in range(Nfr):
        slice_end = slice_start + Horizon_samples
        sigLPC = Win * sig[slice_start:slice_end]
        
        # Energy
        energy = np.sum(sigLPC ** 2)
        frame_energies.append(energy)
        
        # Zero-crossing rate (indicator of voicing)
        zcr = np.sum(np.abs(np.diff(np.sign(sigLPC)))) / (2 * len(sigLPC))
        frame_zcr.append(zcr)
        
        frame_data.append({
            'frame_num': l,
            'slice_start': slice_start,
            'slice_end': slice_end,
            'sigLPC': sigLPC,
            'energy': energy,
            'zcr': zcr
        })
        
        slice_start += Shift
    
    frame_energies = np.array(frame_energies)
    frame_zcr = np.array(frame_zcr)
    
    # Classify frames as voiced/unvoiced
    # Voiced: high energy, low ZCR
    # Unvoiced: lower energy, high ZCR
    energy_threshold = np.median(frame_energies)
    zcr_threshold = np.median(frame_zcr)
    
    voiced_frames = []
    unvoiced_frames = []
    
    for i, frame in enumerate(frame_data):
        if frame['energy'] > energy_threshold and frame['zcr'] < zcr_threshold:
            frame['type'] = 'voiced'
            voiced_frames.append(i)
        elif frame['energy'] > energy_threshold * 0.3 and frame['zcr'] > zcr_threshold:
            frame['type'] = 'unvoiced'
            unvoiced_frames.append(i)
        else:
            frame['type'] = 'silence'
    
    print(f"\nFrame classification:")
    print(f"  Voiced frames: {len(voiced_frames)}")
    print(f"  Unvoiced frames: {len(unvoiced_frames)}")
    print(f"  Silence/transition frames: {Nfr - len(voiced_frames) - len(unvoiced_frames)}")
    
    # Select interesting frames to analyze
    selected_frames = []
    
    # Select 2-3 voiced frames from different parts
    if len(voiced_frames) >= 3:
        selected_frames.append((voiced_frames[len(voiced_frames)//4], 'voiced'))
        selected_frames.append((voiced_frames[len(voiced_frames)//2], 'voiced'))
        selected_frames.append((voiced_frames[3*len(voiced_frames)//4], 'voiced'))
    elif len(voiced_frames) > 0:
        selected_frames.append((voiced_frames[0], 'voiced'))
    
    # Select 2-3 unvoiced frames
    if len(unvoiced_frames) >= 2:
        selected_frames.append((unvoiced_frames[len(unvoiced_frames)//3], 'unvoiced'))
        selected_frames.append((unvoiced_frames[2*len(unvoiced_frames)//3], 'unvoiced'))
    elif len(unvoiced_frames) > 0:
        selected_frames.append((unvoiced_frames[0], 'unvoiced'))
    
    print(f"\nSelected {len(selected_frames)} frames for detailed analysis:")
    for frame_idx, frame_type in selected_frames:
        print(f"  Frame {frame_idx}: {frame_type} (energy={frame_data[frame_idx]['energy']:.2f}, ZCR={frame_data[frame_idx]['zcr']:.3f})")
    
    # Analyze selected frames with different LPC orders
    print(f"\nAnalyzing with LPC orders: {lpc_orders}")
    print(f"\nGenerating comparison plots...")
    
    for frame_idx, frame_type in selected_frames:
        frame = frame_data[frame_idx]
        sigLPC = frame['sigLPC']
        
        # Compute autocorrelation
        r = np.correlate(sigLPC, sigLPC, mode='full')
        r = r[len(sigLPC)-1:]
        
        for order in lpc_orders:
            print(f"\nFrame {frame_idx} ({frame_type}) - Order {order}:")
            
            # Compute LPC coefficients
            a = my_levinson(r, order)
            
            # Create analysis plot
            fig = analyze_lpc_frame(sigLPC, a, Fs, frame_idx, order, frame_type, save_plot=True)
            plt.close(fig)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Plots saved in: ./lpc_analysis_plots/")
    print("="*70)

if __name__ == "__main__":
    import sys
    
    # Load audio file
    Fs, sig = wavfile.read('Speech Sample.wav')
    sig = sig / np.max(np.abs(sig))  # normalize the signal
    
    print("\n" + "="*70)
    print("LPC ANALYSIS TOOL")
    print("="*70)
    print("\nModes:")
    print("  1. Audio Demo (listen to original, synthesized, and echo)")
    print("  2. Frame-by-Frame Analysis (compare LPC filter with FFT spectrum)")
    print("  3. Both")
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        print(f"\nMode selected: {choice}")
    else:
        try:
            choice = input("\nEnter choice (1, 2, or 3) [default: 2]: ").strip()
        except EOFError:
            choice = '2'
            print("2 (default)")
        if not choice:
            choice = '2'
    
    if choice in ['1', '3']:
        print("\n" + "-"*70)
        print("RUNNING AUDIO DEMO")
        print("-"*70)
        
        print("Processing LPC analysis-synthesis...")
        out = lpc_as_toyou(sig, Fs)
        
        # Create echo signal
        echo_sig = np.concatenate((out, np.zeros(2000), sig[:-2000]))

        # Enable interactive mode for matplotlib
        plt.ion()
        
        # Create figure with all signals
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Plot 1: Original Signal
        time_sig = np.arange(len(sig)) / Fs
        axes[0].plot(time_sig, sig)
        axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: LPC Synthesized Signal
        time_out = np.arange(len(out)) / Fs
        axes[1].plot(time_out, out, color='orange')
        axes[1].set_title('LPC Synthesized Signal', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Echo Effect (Synthesized + Original)
        time_echo = np.arange(len(echo_sig)) / Fs
        axes[2].plot(time_echo, echo_sig, color='green')
        axes[2].set_title('Echo Effect (Synthesized + Silence + Original)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()  # Draw the figure
        plt.pause(1)  # Give more time for window to fully render
        
        # Now play audio while figure is visible (in same order as plots)
        print("\n1. Playing ORIGINAL signal (top plot)...")
        sd.play(sig, Fs)
        sd.wait()
        
        print("2. Playing LPC SYNTHESIZED signal (middle plot)...")
        sd.play(out, Fs)
        sd.wait()
        
        print("3. Playing ECHO effect (bottom plot)...")
        sd.play(echo_sig, Fs)
        sd.wait()
        
        print("\nPlayback complete!")
        if choice == '1':
            print("Close the figure window to exit.")
            plt.show()
        else:
            plt.close('all')
    
    if choice in ['2', '3']:
        if choice == '3':
            print("\n" + "-"*70)
        
        # Ask for LPC orders to test
        print("\nEnter LPC orders to test (comma-separated)")
        print("Example: 8,16,24,32,40")
        
        # Check for command-line argument for orders
        if len(sys.argv) > 2:
            orders_input = sys.argv[2]
            print(f"Orders: {orders_input}")
        else:
            try:
                orders_input = input("Orders [default: 8,16,24,32,40]: ").strip()
            except EOFError:
                orders_input = ""
                print("8,16,24,32,40 (default)")
        
        if orders_input:
            lpc_orders = [int(x.strip()) for x in orders_input.split(',')]
        else:
            lpc_orders = [8, 16, 24, 32, 40]
        
        # Run frame-by-frame analysis
        frame_by_frame_analysis(sig, Fs, lpc_orders)
        
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)