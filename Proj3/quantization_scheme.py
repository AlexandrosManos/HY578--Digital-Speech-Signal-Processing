import os
import sys
import glob
import numpy as np
from scipy.io import wavfile
from my_levinson import my_levinson
from lpc_transforms import lpc_to_companded
from lbg_vq import lbg_vq

try:
    Dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    Dir = os.path.abspath('.')

tempRoot = os.path.join(Dir, "TIMIT")
dataSet = os.path.join(tempRoot, "train")


Size = 512 # 64
# The proper format to save the codebook??????
OUTPUT_FILE = f"lpc_codebook_{Size}.npy"



def features(filepath):
    OrderLPC = 24
    Horizon = 30
    Fs, sig = wavfile.read(filepath)
    sig = sig.astype(float)
    sig /= np.max(np.abs(sig))

    Horizon = int(Horizon * Fs / 1000)
    Shift = Horizon // 2
    Win = np.hanning(Horizon)
    Lsig = len(sig)
    slice_start = 0
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1)

    features = []

    for l in range(Nfr):
        slice_end = slice_start + Horizon
        sigLPC = Win * sig[slice_start:slice_end]

        if np.sum(np.abs(sigLPC)) < 1e-5:
            slice_start += Shift
            continue

        try:
            r = np.correlate(sigLPC, sigLPC, mode='full')
            r = r[len(sigLPC) - 1:]
            a = my_levinson(r, OrderLPC)
            g = lpc_to_companded(a)
            features.append(g)

        except (np.linalg.LinAlgError, ValueError):
            pass

        slice_start += Shift

    return np.asarray(features, dtype=np.float64)



if __name__ == "__main__":

    if not os.path.exists(dataSet):
        print(f"Directory not found: {dataSet}")
        sys.exit(1)

    wav_files = glob.glob(os.path.join(dataSet, "**/*.wav"), recursive=True)
    # it will take an hour for all the files, so I limit to 30 for testing -- to remove
    wav_files = glob.glob(os.path.join(dataSet, "**/*.wav"), recursive=True)[:30]

    if not wav_files:
        print(f"No .wav files found in {dataSet}.")
        sys.exit(1)

    all_features = []

    for i, filepath in enumerate(wav_files):
        if i % 5 == 0:
            print(f"  Processing file {i + 1}/{len(wav_files)}: {os.path.basename(filepath)}")

        file_features = features(filepath)

        if file_features.ndim == 2 and file_features.shape[0] > 0:
            all_features.append(file_features)

    if not all_features:
        print("Empty array all_features.")
        sys.exit(1)

    training_data = np.vstack(all_features)

    try:
        codebook = lbg_vq(
            training_data,
            codebook_size=Size,
            epsilon=0.01,
            max_iter=50,
            tol=1e-4
        )
    except Exception as e:
        print(f"Error during LBG-VQ training: {e}")
        sys.exit(1)

    output_path = os.path.join(Dir, OUTPUT_FILE)
    np.save(output_path, codebook)
    print(f"Codebook successfully saved to: {output_path}")
