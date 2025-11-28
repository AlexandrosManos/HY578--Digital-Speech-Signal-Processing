import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import sounddevice as sd
from my_levinson import my_levinson
from lpc_transforms import lpc_to_companded, companded_to_lpc
from lbg_vq import vq_encode, vq_decode

Dir = os.path.dirname(os.path.abspath(__file__))

def lpc_vq_synthesis(Fs, sig, codebook):
    """LPC analysis with vector quantization and synthesis."""
    OrderLPC = 24
    Horizon = int(30 * Fs / 1000)  # 30ms frames
    Shift = Horizon // 2
    Win = np.hanning(Horizon)
    
    Nfr = int(np.floor((len(sig) - Horizon) / Shift) + 1)
    out = np.zeros_like(sig)
    buffer = np.zeros(Horizon - Shift)
    
    for l in range(Nfr):
        start = l * Shift
        sigLPC = Win * sig[start:start + Horizon]
        
        if np.sum(np.abs(sigLPC)) < 1e-5:
            out[start:start + Shift] = buffer
            buffer = np.zeros(Shift)
            continue
        
        try:
            # LPC analysis with ORIGINAL coefficients
            r = np.correlate(sigLPC, sigLPC, mode='full')[len(sigLPC)-1:]
            a = my_levinson(r, OrderLPC)
            
            # Gain from ORIGINAL LPC
            gain = np.sqrt(np.sum(a * r[:len(a)]))
            
            # Excitation from ORIGINAL LPC
            ex = lfilter(a, [1], sigLPC)
            
            # Vector quantize reflection coefficients
            g = lpc_to_companded(a)
            labels, _ = vq_encode(g.reshape(1, -1), codebook)
            g_quant = vq_decode(labels, codebook)[0]
            a_quant = companded_to_lpc(g_quant)
            
            # Synthesis with ORIGINAL excitation and gain
            s = lfilter([gain], a_quant, ex)
            s[:Shift] += buffer
            out[start:start + Shift] = s[:Shift]
            buffer = s[Shift:Horizon]
            
        except (ValueError, np.linalg.LinAlgError):
            out[start:start + Shift] = buffer
            buffer = np.zeros(Shift)
    
    return out


if __name__ == "__main__":
    # Configuration
    codebook_sizes = [64, 512, 2056]  # Test different codebook sizes
    
    # Get first male and female test files from TIMIT
    test_dir = os.path.join(Dir, "TIMIT", "test")
    male_files = [f for f in os.listdir(test_dir) if f.startswith('m')][:1]
    female_files = [f for f in os.listdir(test_dir) if f.startswith('f')][:1]
    
    test_files = []
    for speaker in male_files + female_files:
        speaker_dir = os.path.join(test_dir, speaker)
        wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')][:1]
        if wav_files:
            test_files.append(os.path.join(speaker_dir, wav_files[0]))
    
    if not test_files:
        print("No test files found in TIMIT/test")
        exit(1)
    
    # Process each test file
    for test_file in test_files:
        print(f"\n{'='*50}")
        print(f"File: {os.path.basename(test_file)}")
        print('='*50)
        
        # Load audio
        Fs, sig = wavfile.read(test_file)
        sig = sig.astype(float) / np.max(np.abs(sig))
        
        # Play original
        print("Playing ORIGINAL...")
        sd.play(sig, Fs)
        sd.wait()
        
        # Test each codebook size
        for size in codebook_sizes:
            codebook_path = os.path.join(Dir, f"lpc_codebook_{size}.npy")
            if not os.path.exists(codebook_path):
                print(f"Codebook {size} not found")
                continue
            
            print(f"\nCodebook size: {size}")
            codebook = np.load(codebook_path)
            
            # Synthesize
            synth = lpc_vq_synthesis(Fs, sig, codebook)
            synth = synth / (np.max(np.abs(synth)) + 1e-9)
            
            # Save
            output_path = os.path.join(Dir, f"output_{size}_{os.path.basename(test_file)}")
            wavfile.write(output_path, Fs, (synth * 32767).astype(np.int16))
            print(f"Saved: {output_path}")
            
            # Play synthesized
            print(f"Playing SYNTHESIZED ({size})...")
            sd.play(synth, Fs)
            sd.wait()
