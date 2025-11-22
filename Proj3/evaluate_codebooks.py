import os
import sys
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import sounddevice as sd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Resolve directories
PROJ3_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJ3_DIR)

# Add Proj3 directory to import path for local modules
if PROJ3_DIR not in sys.path:
    sys.path.append(PROJ3_DIR)

# Import local modules
from my_levinson import my_levinson
from lpc_transforms import lpc_to_companded, companded_to_lpc
from lbg_vq import vq_encode, vq_decode

def analyze_frame(sig_frame, order=24):
    # Apply window
    window = np.hanning(len(sig_frame))
    frame = sig_frame * window
    
    # Skip silent frames
    if np.sum(np.abs(frame)) < 1e-5:
        return None, None, None
    
    try:
        # LPC analysis
        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame)-1:]  # Take non-negative lags
        a = my_levinson(r, order)
        
        # Calculate excitation
        e = lfilter(a, [1], frame)
        
        # Convert to companded reflection coefficients
        g = lpc_to_companded(a)
        
        return g, e, a
        
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error in LPC analysis: {e}")
        return None, None, None

def analyze_quantize_synthesize(input_wav, output_wav, codebook_path, order=24, frame_ms=30):
    # Load the codebook
    codebook = np.load(codebook_path)
    
    # Read input audio
    Fs, sig = wavfile.read(input_wav)
    if sig.dtype != np.float64:
        sig = sig.astype(np.float64) / (2 ** (8 * sig.itemsize - 1) - 1)
    
    # Frame parameters
    frame_size = int(Fs * frame_ms / 1000)
    hop_size = frame_size // 2
    num_frames = (len(sig) - frame_size) // hop_size + 1
    
    # Initialize output signal
    out_sig = np.zeros_like(sig)
    
    # Process each frame
    for i in tqdm(range(num_frames), desc=f"Processing {os.path.basename(codebook_path)}"):
        start = i * hop_size
        end = start + frame_size
        frame = sig[start:end]
        
        # Analyze frame
        g_companded, e, a = analyze_frame(frame, order)
        if g_companded is None:
            continue
        
        # Vector quantization
        try:
            labels = vq_encode(g_companded.reshape(1, -1), codebook)[0]
            g_quantized = vq_decode(labels, codebook)[0]
            
            # Convert back to LPC coefficients
            a_quantized = companded_to_lpc(g_quantized)
            
            # Synthesis
            if e is not None and a_quantized is not None:
                synth_frame = lfilter([1], a_quantized, e)
                
                # Apply window and overlap-add
                window = np.hanning(frame_size)
                out_sig[start:end] += synth_frame[:frame_size] * window
                
        except Exception as e:
            print(f"Error in quantization/synthesis: {e}")
            continue
    
    # Normalize and save output
    original_peak = np.max(np.abs(sig))
    if np.max(np.abs(out_sig)) > 0:
        scale = original_peak if original_peak > 0 else 1.0
        out_sig = out_sig / np.max(np.abs(out_sig)) * scale
    wavfile.write(output_wav, Fs, out_sig.astype(np.float32))
    
    return out_sig

def _to_float(sig):
    if sig.dtype == np.float64:
        return sig
    max_val = (2 ** (8 * sig.itemsize - 1)) - 1
    return sig.astype(np.float64) / max_val

def save_waveform_plot(original_sig, synth_sig, Fs, size, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    min_len = min(len(original_sig), len(synth_sig))
    if min_len == 0:
        return

    time = np.arange(min_len) / Fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, original_sig[:min_len], label="Original", linewidth=1)
    plt.plot(time, synth_sig[:min_len], label=f"Synthesized {size}", linewidth=1, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform Comparison - {size}-codeword")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"waveform_{size}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved waveform plot: {plot_path}")

def process_test_files(codebook_path, output_dir="test_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = []
    test_files.extend(glob.glob(os.path.join(PROJ3_DIR, "TIMIT", "test", "m*", "*.wav"))[:1])  # First male
    test_files.extend(glob.glob(os.path.join(PROJ3_DIR, "TIMIT", "test", "f*", "*.wav"))[:1])  # First female
    
    for test_file in test_files:
        print(f"\nProcessing {os.path.basename(test_file)}...")
        
        # Set up output path
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        codebook_size = os.path.basename(codebook_path).split('_')[-1].split('.')[0]
        output_wav = os.path.join(output_dir, f"test_{codebook_size}_{base_name}.wav")
        
        # Process the file
        analyze_quantize_synthesize(test_file, output_wav, codebook_path)
        
        # Play original and synthesized
        Fs, original = wavfile.read(test_file)
        if original.dtype != np.float64:
            original = original.astype(np.float64) / (2 ** (8 * original.itemsize - 1) - 1)
        
        print("Playing original...")
        sd.play(original, Fs)
        sd.wait()
        
        print("Playing synthesized...")
        synth = wavfile.read(output_wav)[1].astype(np.float64)
        sd.play(synth, Fs)
        sd.wait()

def main():
    # Input file (using available speech sample)
    input_file = os.path.join(ROOT_DIR, "Proj1", "speechsample.wav")

    # Output files
    output_64 = os.path.join(PROJ3_DIR, "synthesized_64.wav")
    output_512 = os.path.join(PROJ3_DIR, "synthesized_512.wav")
    output_2056 = os.path.join(PROJ3_DIR, "synthesized_2056.wav")

    # Codebook paths
    codebook_64 = os.path.join(PROJ3_DIR, "lpc_codebook_64.npy")
    codebook_512 = os.path.join(PROJ3_DIR, "lpc_codebook_512.npy")
    codebook_2056 = os.path.join(PROJ3_DIR, "lpc_codebook_2056.npy")

    configs = [
        (64, codebook_64, output_64),
        (512, codebook_512, output_512),
        (2056, codebook_2056, output_2056),
    ]

    synthesized = {}

    for size, codebook_path, output_path in configs:
        if not os.path.exists(codebook_path):
            print(f"Codebook not found for size {size}: {codebook_path}")
            continue

        print(f"\nProcessing with {size}-codeword codebook...")
        synthesized[size] = analyze_quantize_synthesize(input_file, output_path, codebook_path)
        print(f"{size}-codeword output saved to: {output_path}")

    if not synthesized:
        print("No codebooks were processed.")
        return

    print("\nProcessing complete!")

    # Play the results for comparison
    print("\nPlaying original...")
    Fs, original = wavfile.read(input_file)
    original_float = _to_float(original)
    sd.play(original, Fs)
    sd.wait()

    for size in [64, 512, 2056]:
        if size in synthesized:
            print(f"\nPlaying {size}-codeword synthesis...")
            sd.play(synthesized[size], Fs)
            sd.wait()
            save_waveform_plot(original_float, synthesized[size], Fs, size, output_dir=os.path.join(PROJ3_DIR, "plots"))

if __name__ == "__main__":
    main()