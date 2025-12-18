# pip install soundfile sounddevice numpy
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd   # optional (for playback)
    HAVE_SD = True
except Exception:
    HAVE_SD = False

def rms_energy(x):
    """Total signal energy (sum of squares) across all channels."""
    return np.sum(x.astype(np.float64)**2)

def add_noise_at_snr(signal, snr_db, rng=None):
    """
    Add zero-mean unit-variance Gaussian noise scaled to achieve the desired (global) SNR in dB.
    SNR = 10*log10(E_signal / E_noise).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(size=signal.shape)

    Es = rms_energy(signal)
    En = rms_energy(noise)

    # Scale factor for noise amplitude to hit target SNR:
    a = np.sqrt(Es / En) * 10 ** (-snr_db / 20.0)

    noisy = signal + a * noise
    return noisy

def normalize_for_wav(x, margin=1.1):
    """Normalize to keep headroom before saving to 16-bit PCM."""
    peak = np.max(np.abs(x))
    if peak == 0:
        return x
    return x / (margin * peak)


def spectra_power(sig, fs):
    # y[n] = x[n] + b[n] --> Y(w) = X(w) + B(w)
    Horizon = 30   # 30ms - window length
    Horizon = int(Horizon * fs / 1000)

    Shift = int(Horizon / 2) # frame size - step size
    Win = np.hanning(Horizon) # analysis window

    Lsig = len(sig)
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1) # number of frames

    all_spec = []
    slice_start = 0

    for l in range(Nfr):
        slice_end = slice_start + Horizon

        # check bounds
        if slice_end > Lsig:
            break

        s = Win * sig[slice_start:slice_end]

        sigFFT = np.abs(np.fft.fft(s)) ** 2 # |B(w)|^2
        all_spec.append(sigFFT)

        slice_start += Shift

    if not all_spec:
        return np.zeros(Horizon)

    out = np.mean(all_spec, axis=0) # Expecation E[]
    return out


def spectral_subtraction(sig, fs):

    noise = spectra_power(sig[:1000], fs)

    Horizon = 30  # 30ms - window length
    Horizon = int(Horizon * fs / 1000)

    Shift = int(Horizon / 2) # frame size - step size
    Win = np.hanning(Horizon) # analysis window

    Lsig = len(sig)
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1) # number of frames

    out = np.zeros_like(sig)
    Buffer = np.zeros(Shift)

    slice_start = 0
    tosave_start = 0

    for l in range(Nfr):
        slice_end = slice_start + Horizon
        tosave_end = tosave_start + Shift

        if slice_end > Lsig:
            break

        # Windowing
        s = Win * sig[slice_start:slice_end]

        # FFT Analysis
        fft = np.fft.fft(s) # flag --> check the fft.fft implementation
        sig_abs = np.abs(fft)
        sig_angle = np.angle(fft)

        subtr = sig_abs ** 2 - noise # |X(w)|^2 = |Y(w)|^2 - Sb (Sb = E[|B(w)|^2])
        subtr = np.maximum(subtr, 0) # 0 otherwise

        Csig = np.sqrt(subtr) * np.exp(1j * sig_angle) # |X| * exp(j*angle(Y))
        Csig = np.real(np.fft.ifft(Csig))

        Csig[:Shift] = Csig[:Shift] + Buffer            # overlap-add
        out[tosave_start:tosave_end] = Csig[:Shift]  # save the first part of the frame
        Buffer = Csig[Shift:Horizon]                 # buffer the rest of the frame

        slice_start += Shift
        tosave_start += Shift

    return out


# -------- Wiener filter ---------
def wiener_filter_enhancement(sig, fs, t=0.9): # flag --> gia t (?)

    noise = spectra_power(sig[:1000], fs)

    Horizon = 30  # 30ms - window length
    Horizon = int(Horizon * fs / 1000)

    Shift = int(Horizon / 2) # frame size - step size
    Win = np.hanning(Horizon) # analysis window

    Lsig = len(sig)
    Nfr = int(np.floor((Lsig - Horizon) / Shift) + 1) # number of frames

    out = np.zeros_like(sig)
    Buffer = np.zeros(Shift)

    slice_start = 0
    tosave_start = 0

    s_init = Win * sig[0:Horizon]
    fft = np.fft.fft(s_init)
    Sx = np.maximum(np.abs(fft) ** 2 - noise, 0)

    H = Sx / (Sx + noise)

    for l in range(Nfr):
        slice_end = slice_start + Horizon
        tosave_end = tosave_start + Shift

        if slice_end > Lsig:
            break

        # Windowing
        s = Win * sig[slice_start:slice_end]

        # FFT Analysis
        fft = np.fft.fft(s) # flag --> check the fft.fft implementation
        sig_abs = np.abs(fft)
        sig_angle = np.angle(fft)

        # X(pL,w) = Y(pL,w) * Hw((p-1)L,w)
        X_freq = fft * H

        subtr = sig_abs ** 2 - noise # |X(w)|^2 = |Y(w)|^2 - Sb (Sb = E[|B(w)|^2])
        subtr = np.maximum(subtr, 0) # 0 otherwise

        # Smooth power spectrum
        # Sx(pL,w) = τSx((p-1)L,w) + (1-τ)Sx(pL,w)
        Sx = t * Sx + (1 - t) * subtr

        # Hw(pL,w) = Sx(pL,w) / (Sx(pL,w) + Sb(w))
        H = Sx / (Sx + noise)

        Csig = np.abs(X_freq) * np.exp(1j * sig_angle)
        Csig = np.real(np.fft.ifft(Csig))


        Csig[:Shift] = Csig[:Shift] + Buffer            # overlap-add
        out[tosave_start:tosave_end] = Csig[:Shift]  # save the first part of the frame
        Buffer = Csig[Shift:Horizon]                 # buffer the rest of the frame

        slice_start += Shift
        tosave_start += Shift

    return out




# --- Parameters ---
in_wav  = "furelise-1000z.wav"
out_wav = "furelise-1000z-noise.wav"
SNR_DB  = 10  # target global SNR in dB

# --- Read (soundfile returns float in [-1, 1] when possible) ---
s, fs = sf.read(in_wav, always_2d=False)  # shape: (N,) or (N, C)

# --- (Optional) Listen to the clean audio ---
if HAVE_SD:
    print("Playing clean audio...")
    sd.play(s, fs)
    sd.wait()

# --- Add white Gaussian noise at target SNR ---
sn = add_noise_at_snr(s, SNR_DB)

# --- (Optional) Listen to the noisy audio ---
if HAVE_SD:
    print("Playing noisy audio...")
    sd.play(sn, fs)
    sd.wait()

# --- Normalize and save as 16-bit PCM ---
sn_norm = normalize_for_wav(sn, margin=1.1)
sf.write(out_wav, sn_norm, fs, subtype="PCM_16")

print(f"Saved noisy file to: {out_wav}")

# --- Next steps :
# 1) Spectral subtraction
enhanced_sig = spectral_subtraction(sn_norm, fs)

enhanced_norm = normalize_for_wav(enhanced_sig)
out_ss_filename = "enhanced_spectral_sub.wav"
sf.write(out_ss_filename, enhanced_norm, fs, subtype="PCM_16")

print(f"Saved enhanced file to: {out_ss_filename}")
print()
if HAVE_SD:
    print("Playing enhanced spectral subtraction audio...")
    sd.play(enhanced_norm, fs)
    sd.wait()

# 2) Wiener filtering

enhanced_wiener = wiener_filter_enhancement(sn_norm, fs)
enhanced_wiener_norm = normalize_for_wav(enhanced_wiener)
out_wiener_filename = "enhanced_wiener.wav"

sf.write(out_wiener_filename, enhanced_wiener_norm, fs, subtype="PCM_16")
print(f"Saved enhanced file to: {out_wiener_filename}")

# --- (Optional) Listen to the noisy audio ---
if HAVE_SD:
    print("Playing enhanced Wiener audio...")
    sd.play(enhanced_wiener_norm, fs)
    sd.wait()

# You can process `sn_norm` with those methods and save additional outputs.
