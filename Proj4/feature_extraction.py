import librosa
import numpy as np
import h5py
import os

def add_white_noise(signal, snr_db):
    """
    Adds white Gaussian noise to a signal at a specified SNR.
    """
    # Calculate signal power
    sig_power = np.mean(signal ** 2)
    
    # Calculate noise power required for the target SNR
    # SNR_db = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR_db / 10)
    noise_power = sig_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    return signal + noise

def extract_and_save_mfcc(input_wav, output_h5, n_mfcc=13, n_fft_ms=20, hop_length_ms=5, n_mels=40, snr_db=None):
    """
    Extracts MFCC features from a speech signal and saves them to an .h5 file.

    Args:
        input_wav (str): Path to the input .wav file.
        output_h5 (str): Path to the output .h5 file.
        n_mfcc (int): Number of MFCCs to extract. Default is 13.
        n_fft_ms (int): Frame size in milliseconds. Default is 20 ms.
        hop_length_ms (int): Time step (hop length) in milliseconds. Default is 5 ms.
        n_mels (int): Number of Mel bands to generate. Default is 40.
        snr_db (float): Signal-to-Noise Ratio in dB. If None, no noise is added.
    """
    try:
        # Load the audio file
        # sr=None preserves the native sampling rate
        y, sr = librosa.load(input_wav, sr=None)
        
        # Add noise if requested
        if snr_db is not None:
            y = add_white_noise(y, snr_db)
        
        # Calculate n_fft and hop_length in samples
        n_fft = int(sr * n_fft_ms / 1000)
        hop_length = int(sr * hop_length_ms / 1000)
        
        # Extract MFCCs
        # We use n_fft as the window length as well
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, n_mels=n_mels)
        
        # Save to .h5 file
        with h5py.File(output_h5, 'w') as hf:
            hf.create_dataset('mfccs', data=mfccs)
            hf.attrs['sr'] = sr
            hf.attrs['n_fft'] = n_fft
            hf.attrs['hop_length'] = hop_length
            if snr_db is not None:
                hf.attrs['snr_db'] = snr_db
            
        # print(f"Successfully saved MFCCs to {output_h5}")
        # print(f"MFCC shape: {mfccs.shape}")
        
    except Exception as e:
        print(f"Error processing {input_wav}: {e}")

if __name__ == "__main__":
    # Example usage
    # Define a sample file path from the dataset we explored
    sample_wav = "/Users/pswmi64/Desktop/my-projects/hy578/HY578--Digital-Speech-Signal-Processing/Proj4/TIMIT/train/falk0/sa1.wav"
    output_h5 = "sa1_features.h5"
    
    if os.path.exists(sample_wav):
        extract_and_save_mfcc(sample_wav, output_h5)
    else:
        print(f"Sample file not found: {sample_wav}")
