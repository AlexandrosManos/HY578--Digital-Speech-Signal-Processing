import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, freqz
from my_levinson import my_levinson
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
        r = np.correlate(sigLPC, sigLPC, mode='full')  # autocorrelation
        r = r[len(sigLPC)-1:]  # keep only the positive lags
        a = my_levinson(r, OrderLPC)  # LPC coefficients
        
        # Calculate LPC gain (prediction error energy)
        G = np.sqrt(np.sum(a * r[:len(a)]))
        
        # Get excitation signal (residual)
        ex = lfilter(a, [1], sigLPC)  # inverse filter A(z)
        
        # --- synthesis ---
        s = lfilter([G], a, ex)  # H(z) = G/A(z) applied to excitation
        ens = np.sum(s ** 2) # short-time energy of output
        g = np.sqrt(en / (ens)) # normalization factor
        s = s * g                       # energy compensation
        
        s[:Shift] = s[:Shift] + Buffer            # overlap-add
        out[tosave_start:tosave_end] = s[:Shift]  # save the first part of the frame
        Buffer = s[Shift:Horizon]                 # buffer the rest of the frame

        slice_start += Shift    # Move the frame
        tosave_start += Shift

    return out

def analyze_frame(sig_frame, Fs, order, frame_num, frame_type):
    """Compare LPC filter response with FFT of the given frame."""
    NFFT = 2048
    
    # LPC Analysis
    r = np.correlate(sig_frame, sig_frame, mode='full')
    r = r[len(sig_frame)-1:]  # keep positive lags
    
    a = my_levinson(r, order)  # LPC coefficients
    G = np.sqrt(np.sum(a * r[:len(a)]))  # LPC gain (prediction error energy)

    # LPC frequency response: H(z) = G / A(z)
    w, h = freqz([G], a, worN=NFFT//2, fs=Fs)
    lpc_db = 20 * np.log10(np.abs(h) + 1e-10)

    # FFT of the frame
    X = np.fft.fft(sig_frame, NFFT)
    X_db = 20 * np.log10(np.abs(X[:NFFT//2]) + 1e-10)
    freqs = np.linspace(0, Fs/2, NFFT//2)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, X_db, 'b-', alpha=0.5, linewidth=1.5, label='FFT of Speech Frame')
    plt.plot(w, lpc_db, 'r-', linewidth=2.5, label=f'LPC Filter Response (order={order})')
    plt.title(f'{frame_type.capitalize()} Frame {frame_num} - LPC Order {order}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    plt.xlim([0, Fs/2])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    filename = f"plots/frame{frame_num}_{frame_type}_order{order}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()



    

def frame_by_frame_analysis(sig, Fs, lpc_orders=[8, 16, 24]):
    """
    Frame-by-frame LPC analysis comparing FFT with LPC filter response.
    Analyzes one voiced and one unvoiced frame with different LPC orders.
    """
    print("="*70)
    print("FRAME-BY-FRAME LPC ANALYSIS")
    print("="*70)
    
    # Frame parameters
    frame_len_ms = 30
    frame_len = int(frame_len_ms * Fs / 1000)
    shift = frame_len // 2
    window = np.hanning(frame_len)
    
    
    # Framing
    frames = []
    for start in range(0, len(sig) - frame_len, shift):
        frame = sig[start:start + frame_len] * window
        frames.append(frame)
    
    # Compute energy and zero-crossing for voiced/unvoiced classification
    energies = [np.sum(f ** 2) for f in frames]
    zcrs = [np.mean(np.abs(np.diff(np.sign(f)))) for f in frames]
    
    energy_th = np.percentile(energies, 60)
    zcr_th = np.median(zcrs)
    
    # Pick one voiced and one unvoiced frame
    voiced_idx = next(i for i, (e, z) in enumerate(zip(energies, zcrs)) if e > energy_th and z < zcr_th)
    unvoiced_idx = next(i for i, (e, z) in enumerate(zip(energies, zcrs)) if e > 0.1 * energy_th and z > zcr_th)
    
    voiced_frame = frames[voiced_idx]
    unvoiced_frame = frames[unvoiced_idx]
    
    print("-" * 70)
    
    for order in lpc_orders:
        analyze_frame(voiced_frame, Fs, order, voiced_idx, 'voiced')
        analyze_frame(unvoiced_frame, Fs, order, unvoiced_idx, 'unvoiced')
    

if __name__ == "__main__":
    import sys
    
    # Load audio file
    Fs, sig = wavfile.read('speechsample.wav')
    sig = sig / np.max(np.abs(sig))  # normalize the signal
    
    print("\n" + "="*70)
    print("LPC ANALYSIS TOOL")
    print("="*70)
    print("\nModes:")
    print("  1. Audio Demo (listen to original and synthesized)")
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
        
        out = lpc_as_toyou(sig, Fs)

        # Enable interactive mode for matplotlib
        plt.ion()
        
        # Create figure with both signals
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
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
        
        plt.tight_layout()
        plt.draw()  # Draw the figure
        plt.pause(1)  # Give more time for window to fully render
        
        print("\n1. Playing ORIGINAL signal (top plot)...")
        sd.play(sig, Fs)
        sd.wait()
        
        print("2. Playing LPC SYNTHESIZED signal (bottom plot)...")
        sd.play(out, Fs)
        sd.wait()
        
        print("\nPlayback complete!")
        if choice == '1':
            plt.show()
        else:
            plt.close('all')
    
    if choice in ['2', '3']:
        if choice == '3':
            print("\n" + "-"*70)
        
        print("\nEnter LPC orders to test (comma-separated)")
        print("Example: 8,16,24,32,40")
        
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
            print(f"Using LPC orders: {lpc_orders}")
        else:
            lpc_orders = [8, 16, 24, 32, 40]
            print(f"Using default LPC orders: {lpc_orders}")
        
        # Run frame-by-frame analysis
        frame_by_frame_analysis(sig, Fs, lpc_orders)
        
