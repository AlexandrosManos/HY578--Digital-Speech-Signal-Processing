# Miltos Vasilakis, 2005
# Multimedia Informatics Lab.
# Computer Science Dept.
# University of Crete
# mvasilak@csd.uoc.gr
#
# Python version: Alex Angelakis, 2025
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import numpy.typing as npt
from dataclasses import dataclass
from typing import List, Tuple
from scipy.signal import find_peaks


def SinM_test_hy578(X, Fs, N=None, S=None, L=None, W=None):
    if Fs is None:
        raise ValueError('usage SinMtest(X, Fs, N, S, L, W)')
    if N is None:
        N = 0.030  # 30 ms frame size
    if S is None:
        S = 0.015  # 15 ms frame step
    if L is None:
        L = 80     # 80 frequencies

    N = int(np.floor(N * Fs))
    N = int(np.floor(N / 2) * 2 + 1)  # make it odd
    S = int(np.floor(S * Fs))

    if W is None or len(W) != N:
        W = np.hanning(N)  # hanning window

    SinM = SinM_analysis(X, Fs, N, S, L, W)
    Y, SNR_by_frame = SinM_synthesis_PI(SinM, N, S, Fs, X)
    return Y, SNR_by_frame


# -----------------------------
# Data structure
# -----------------------------
@dataclass
class SinMFrame:
    Tc: float
    AMP: np.ndarray
    PH: np.ndarray
    F: np.ndarray
    SNR: float


# -----------------------------
# Analysis
# -----------------------------
def SinM_analysis(X, Fs, N, S, L=80, W=None) -> List[SinMFrame]:
    """
    SinM = SinM_analysis(X, Fs, N, S, L, W)
    """
    if W is None:
        W = np.hanning(N)
    LN = len(X)
    Nfr = int(np.floor((LN - N) / S) + 1)  # number of frames
    # analyze each frame
    frame_start = 0
    frame_end = N
    SinM: List[SinMFrame] = []
    for fr in range(Nfr):
        Tc = (frame_start + frame_end) / 2.0
        Xf = X[frame_start:frame_end]
        AMP, PH, F = SinM_analysis_frame(Xf, Fs, int(np.floor((N - 1) / 2)), L, W)
        SinM.append(SinMFrame(Tc=Tc,
                              AMP=AMP,
                              PH=PH,
                              F=F,
                              SNR=0.0))
        frame_start += S
        frame_end += S
    return SinM


def SinM_analysis_frame(Xf, Fs, N, L=80, W=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [ AMP, PH, F ] = SinM_analysis_frame(Xf, N, Fs, L, W)
    Computes the SinM parameters of a single frame
    """
    if W is None:
        W = np.hanning(2 * N + 1)
    Wn = W / np.sum(W)  # normalize window to 1
    NFFT = 2048  # 2^(ceil(log2(2*N+1)));
    Wf = Xf * Wn
    Sw = np.zeros(NFFT, dtype=complex)
    Sw[0:N + 1] = Wf[N:2 * N + 1]
    Sw[NFFT - N:NFFT] = Wf[0:N]
    S_all = np.fft.fft(Sw, n=NFFT)
    S = S_all[:NFFT // 2 + 1]
    FBins, AMP, PH = SinM_peakPicking(S, L)
    F = (FBins.astype(float)) * Fs / NFFT  # FBins already zero-based in Python
    return AMP, PH, F


def SinM_peakPicking(S, L) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [ F AMP PH ] = SinM_peakPicking(S, L)
    Returns the L highest amplitude peaks of the spectrum S, along with frequency and unwrapped phase
    """
    F = np.array([], dtype=int)
    AMP = np.array([], dtype=float)
    PH = np.array([], dtype=float)

    mag = np.abs(S)
    peaks = find_peaks(mag)
    if len(peaks) == 0:
        return F, AMP, PH

    # Find the indices of the L largest-amplitude peaks.
    # 1. mag[peaks] gives the amplitudes at the detected peak frequencies.
    # 2. np.argsort sorts these amplitudes in ascending order.
    # 3. [-L:] takes the indices of the L largest values (from the sorted list).
    # 4. [::-1] reverses to descending order (largest to smallest).
    idx = np.argsort(mag[peaks])[-L:][::-1]

    F = peaks[idx]
    AMP = mag[F]
    PH = np.angle(S[F]) #unwrap maybe

    return F, AMP, PH


# -----------------------------
# Synthesis with parameter interpolation
# -----------------------------
def SinM_synthesis_PI(SinM: List[SinMFrame], N, S, Fs, X):
    """
    [ Y SNR_by_frame ] = SinM_synthesis_PI(SinM, N, S, Fs, X)
    """
    Nfr = len(SinM)
    LN = (Nfr - 1) * S + N
    Y = np.zeros(LN)
    # start -> SinM(1)
    L_start = len(SinM[0].F)
    S_start = int(np.floor((N - 1) / 2))
    if L_start > 0:
        Yf = SinM_synthesis_sin_PI(
            S_start, Fs,
            np.zeros(L_start), SinM[0].AMP,
            SinM[0].PH - (2 * np.pi * SinM[0].F / Fs) * S_start,
            SinM[0].PH,
            SinM[0].F, SinM[0].F
        )
        Y[:S_start] = Yf
    #
    step_start = S_start
    step_end = S_start + S
    for fr in range(1, Nfr):
        SinM_prev_new, SinM_curr_new = SinM_FrameToFramePeakMatching(SinM[fr - 1], SinM[fr], S, Fs)
        Yf = SinM_synthesis_sin_PI(
            S, Fs,
            SinM_prev_new.AMP, SinM_curr_new.AMP,
            SinM_prev_new.PH, SinM_curr_new.PH,
            SinM_prev_new.F, SinM_curr_new.F
        )
        Y[step_start:step_end] = Yf
        xf = X[step_start - S:step_end]
        yf = Y[step_start - S:step_end]
        SinM[fr - 1].SNR = 20 * np.log10(np.std(xf) / (np.std(xf - yf) + 1e-12))
        step_start += S
        step_end += S
    # SinM(Nfr) -> end
    L_end = len(SinM[-1].F)
    if L_end > 0:
        S_end = LN - step_start
        Yf = SinM_synthesis_sin_PI(
            S_end, Fs,
            SinM[-1].AMP, np.zeros(L_end),
            SinM[-1].PH,
            SinM[-1].PH + (2 * np.pi * SinM[-1].F / Fs) * S_end,
            SinM[-1].F, SinM[-1].F
        )
        Y[step_start:LN] = Yf
    xf = X[step_start - S:LN]
    yf = Y[step_start - S:LN]
    SinM[-1].SNR = 20 * np.log10(np.std(xf) / (np.std(xf - yf) + 1e-12))

    SNR_by_frame = np.array([fr.SNR for fr in SinM])
    return Y, SNR_by_frame


def SinM_synthesis_sin_PI(N, Fs, AMP_1, AMP_2, PH_1, PH_2, F_1, F_2):
    """
    Yf = SinM_synthesis_sin_PI(N, Fs, AMP_1, AMP_2, PH_1, PH_2, F_1, F_2)
    SinM synthesis by sinusoidal Parameter Interploation.
    Synthesizes N samples of the speech signal from the SinM parameters.
    """
    Yf = np.zeros(N)
    L = len(AMP_1)
    if L == 0 or N <= 0:
        return Yf

    t = np.arange(N, dtype=float)
    for l in range(L):
        # Linear amplitude interpolation, Lecture 5  slide 36
        a_t = AMP_1[l] + (AMP_2[l] - AMP_1[l]) * t / N
        
        # Cubic phase interpolation (McAulay-Quatieri)
        # φ(t) = a + bt + ct² + dt³
        # Boundary conditions: φ(0)=θ₁, φ(N)=θ₂, φ'(0)=ω₁, φ'(N)=ω₂
        w1 = 2 * np.pi * F_1[l] / Fs
        w2 = 2 * np.pi * F_2[l] / Fs
        z = PH_1[l] # initial phase
        c = w1
        a = (3*(PH_2[l] - PH_1[l]) - N*(2*w1 + w2)) / (N**2)
        b = (2*(PH_1[l] - PH_2[l]) + N*(w1 + w2)) / (N**3) 
        phase_t = z + c*t + a*t**2 + b*t**3
        
        # Synthesize
        Yf += a_t * np.cos(phase_t)
			
    return Yf


def SinM_FrameToFramePeakMatching(SinM_prev: SinMFrame, SinM_curr: SinMFrame, S, Fs, Delta=10):
    """
    [SinM_prev_new, SinM_curr_new] = SinM_FrameToFramePeakMatching(SinM_prev, SinM_curr, S, Fs, Delta)
    Returns the frame to frame matched peaks, according to the algorithm in
    "Speech Analysis/Synthesis Based on a Sinusoidal Representation"
    Note: Zeros are inserted in birth and death matchings.
    """
    L_1 = len(SinM_prev.F)
    L_2 = len(SinM_curr.F)
    AMP_1 = SinM_prev.AMP
    AMP_2 = SinM_curr.AMP
    PH_1 = SinM_prev.PH
    PH_2 = SinM_curr.PH
    F_1 = SinM_prev.F
    F_2 = SinM_curr.F

    # I_1, I_2: matching indices (Python: -1 means unmatched)
    I_1 = -np.ones(L_1, dtype=int)
    I_2 = -np.ones(L_2, dtype=int)

    # --------- Inserted code translated -----------
    for n in range(L_1):
        # finding candidate frequencies (if any)
		# #########################
		#    INSERT CODE HERE    #
		# #########################

    # Check for no matched frequencies and make births/deaths
    L_new = int(L_2 + np.sum(I_1 == -1))

    AMP_1_new = np.zeros(L_new)
    AMP_2_new = np.zeros(L_new)
    PH_1_new = np.zeros(L_new)
    PH_2_new = np.zeros(L_new)
    F_1_new = np.zeros(L_new)
    F_2_new = np.zeros(L_new)

    # #########################
	#    INSERT CODE HERE    #
	# #########################

    SinM_1_new = SinMFrame(
        Tc=SinM_prev.Tc,
        AMP=AMP_1_new,
        PH=PH_1_new,
        F=F_1_new,
        SNR=SinM_prev.SNR
    )
    SinM_2_new = SinMFrame(
        Tc=SinM_curr.Tc,
        AMP=AMP_2_new,
        PH=PH_2_new,
        F=F_2_new,
        SNR=SinM_curr.SNR
    )
    return SinM_1_new, SinM_2_new


if __name__ == "__main__":
    fs, s = wavfile.read('arctic_bdl1_snd_norm.wav')

    # normalize if integer PCM
    if s.dtype.kind in 'iu':
        s = s.astype(np.float32) / np.max(np.abs(s))

    # ensure mono
    if s.ndim > 1:
        s = s[:, 0]

    print("Playing original signal...")
    sd.play(s, fs)
    sd.wait()
    sd.stop()

    Y, SNR = SinM_test_hy578(s, fs)

    print("Playing synthesized signal...")
    sd.play(Y, fs)
    sd.wait()
    sd.stop()

    pad_len = len(s) - len(Y)
    if pad_len > 0:
        y_padded = np.concatenate((Y, np.zeros(pad_len)))
    else:
        y_padded = Y[:len(s)]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(s)
    plt.title('Original signal')

    plt.subplot(2, 1, 2)
    plt.plot(y_padded)
    plt.title('Synthesized signal')
    plt.tight_layout()
    plt.show()

    MSE = np.mean((s - y_padded) ** 2)
    print(f"MSE: {MSE:.6f}")

    plt.figure()
    plt.plot(SNR)
    plt.title('SNR per frame')
    plt.xlabel('Frame number')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    plt.show()