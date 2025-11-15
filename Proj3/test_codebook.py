import os
import glob
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from my_levinson import my_levinson
from lpc_transforms import lpc_to_companded, companded_to_lpc
from lbg_vq import vq_encode, vq_decode


try:
    Dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    Dir = os.path.abspath('.')

tempRoot = os.path.join(Dir, "TIMIT")
dataSet = os.path.join(tempRoot, "test")
output_dir = os.path.join(Dir, "my_test")

def lpc_vq_synthesis(Fs, sig, codebook):
    OrderLPC = 24
    Horizon = 30
    Horizon = int(Horizon * Fs / 1000)
    Shift = Horizon // 2
    Win = np.hanning(Horizon)

    Lsig = len(sig)
    slice_start = 0
    tosave_start = 0
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1)

    out = np.zeros_like(sig)
    buffer = np.zeros(Horizon - Shift)

    for l in range(Nfr):
        slice_end = slice_start + Horizon
        tosave_end = slice_start + Shift

        sigLPC = Win * sig[slice_start:slice_end]

        if np.sum(np.abs(sigLPC)) < 1e-5:
            out[tosave_start:tosave_end] = buffer
            buffer = np.zeros(Shift)
            slice_start += Shift
            tosave_start += Shift
            continue

        try:
            r = np.correlate(sigLPC, sigLPC, mode='full')
            r = r[len(sigLPC) - 1:]
            a = my_levinson(r, OrderLPC)

            orignal = np.sqrt(np.sum(a * r[:len(a)]))

            ex_original = lfilter(a, [1], sigLPC)

            g = lpc_to_companded(a)
            labels, _ = vq_encode(g.reshape(1, -1), codebook)
            g_quantized = vq_decode(labels, codebook)[0]

            a_quantized = companded_to_lpc(g_quantized)

            s = lfilter([orignal], a_quantized, ex_original)

            s[:Shift] += buffer
            out[tosave_start:tosave_end] = s[:Shift]
            buffer = s[Shift:Horizon]

        except (ValueError, np.linalg.LinAlgError):
            out[tosave_start:tosave_end] = buffer
            buffer = np.zeros(Shift)

        slice_start += Shift
        tosave_start += Shift

    return out


if __name__ == "__main__":

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(dataSet):
        print(f"Directory not found: {dataSet}")
        sys.exit(1)

    test_files = glob.glob(os.path.join(dataSet, "**/*.wav"), recursive=True)

    if not test_files:
        print(f"No .wav files found in {dataSet}.")
        sys.exit(1)

    codebook_sizes = [64, 512]

    for size in codebook_sizes:

        size_output_dir = os.path.join(output_dir, f"size_{size}")
        if not os.path.exists(size_output_dir):
            os.makedirs(size_output_dir)
            print(f"Created output directory: {size_output_dir}")


        codebook_file = f"lpc_codebook_{size}.npy"
        if not os.path.exists(codebook_file):
            print(f"Codebook file {codebook_file} not found")
            continue

        codebook = np.load(codebook_file)

        for i, filepath in enumerate(test_files):
            if i % 5 == 0:
                print(f" Processing file {i + 1}/{len(test_files)} with size {size}: {os.path.basename(filepath)}")

            Fs, sig = wavfile.read(filepath)
            sig = sig.astype(float)
            sig /= np.max(np.abs(sig))
            synthesized = lpc_vq_synthesis(Fs, sig, codebook)

            out_norm = synthesized / (np.max(np.abs(synthesized)) + 1e-9)
            out_int16 = (out_norm * 32767).astype(np.int16)

            output_file = os.path.join(size_output_dir, f"synth_size{size}_{os.path.basename(filepath)}")
            # wavfile.write(output_file, Fs, synthesized.astype(np.float32))
            wavfile.write(output_file, Fs, out_int16)
