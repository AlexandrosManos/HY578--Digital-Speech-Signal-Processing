# LPC Whisper Synthesis

## 📖 Overview

This module generates **whispered speech** by replacing the periodic glottal pulse excitation with **white noise** while keeping the vocal tract filter (LPC coefficients) unchanged.

## 🚀 Quick Start

```bash
python LPCwhisper.py
```

## 📂 Output Files

All outputs are saved in `whisper_output/`:

| File | Description |
|------|-------------|
| `whisper_order12.wav` | Lower-order LPC whisper (smoother, less detail) |
| `whisper_order24.wav` | Standard-order LPC whisper (balanced quality) |
| `whisper_order36.wav` | Higher-order LPC whisper (more detail) |
| `whisper_order12_spectrogram.png` | Spectrogram visualization (order 12) |
| `whisper_order24_spectrogram.png` | Spectrogram visualization (order 24) |
| `whisper_order36_spectrogram.png` | Spectrogram visualization (order 36) |

## 🎯 How It Works

### Normal Speech:
```
Periodic Glottal Pulses → LPC Filter H(z) → Speech
     (has pitch)            (vocal tract)
```

### Whispered Speech:
```
White Noise → LPC Filter H(z) → Whispered Speech
  (no pitch)     (same vocal tract)
```

**Key Concept:** Only the excitation changes; the vocal tract filter remains the same!

## 🔬 Technical Details

### Algorithm:
1. **Frame the signal** (30ms windows, 50% overlap)
2. **LPC Analysis:** Extract vocal tract filter coefficients
   ```python
   a = my_levinson(r, OrderLPC)
   G = sqrt(sum(a * r))
   ```
3. **Generate noise excitation:** `noise = randn(frame_len)`
4. **Synthesize whisper:** Filter noise through LPC filter
   ```python
   whisper_frame = G * lfilter([1], a, noise)
   ```
5. **Overlap-add** to reconstruct full signal

### Parameters:
- **Frame length:** 30 ms
- **Frame shift:** 15 ms (50% overlap)
- **Window:** Hanning
- **LPC orders tested:** 12, 24, 36

## 📊 Expected Results

### Order Comparison:

| Order | Formants Captured | Quality | Notes |
|-------|------------------|---------|-------|
| **12** | F1, F2 clearly | Smooth, slightly muffled | Minimal representation |
| **24** | F1-F4 well | Most natural | Standard for 16 kHz |
| **36** | All + fine detail | Detailed but slightly harsh | Overparameterized |

### Spectrograms:
- ✅ **No pitch harmonics** (noise-like texture throughout)
- ✅ **Formants preserved** (resonant bands visible)
- ✅ **Temporal structure maintained** (speech timing intact)

## 🎧 Listening Guide

### What to Notice:

1. **Pitch Removal:**
   - Original: Has melody/intonation
   - Whisper: Flat, no pitch variation

2. **Intelligibility:**
   - Words should still be understandable
   - Order 24 typically clearest

3. **Naturalness:**
   - Does it sound like natural whispering?
   - Or artificial/robotic?

4. **Order Differences:**
   - Order 12: Smoother, softer
   - Order 36: More detail, potentially harsher

## 📝 Key Observations

### What the Results Show:

1. **Successful Pitch Removal:**
   - Spectrograms show no harmonic structure
   - Confirms noise excitation working correctly

2. **Formant Preservation:**
   - Vocal tract characteristics maintained
   - Speech remains intelligible

3. **Order Effects:**
   - Order 12: Adequate for basic whisper
   - Order 24: Optimal balance (follows rule: Order ≈ Fs(kHz) + 4-6)
   - Order 36: Diminishing returns, minimal improvement

4. **Source-Filter Model Validated:**
   - Changing excitation alone transforms speech character
   - Proves separation of source and vocal tract

## 🧪 Experiment Modifications

To test different parameters, edit `LPCwhisper.py`:

```python
# Change LPC orders
orders_to_test = [8, 16, 24, 32]

# Change random seed (different noise patterns)
whisper = lpc_whisper(sig, Fs, OrderLPC=24, seed=42)

# Change frame parameters
frame_len = int(20 * Fs / 1000)  # 20ms instead of 30ms
```

## 🔗 Related Files

- `LPC.py` - Main LPC analysis/synthesis
- `my_levinson.py` - Levinson-Durbin recursion algorithm
- `speechsample.wav` - Input audio file


---

**Created:** 2025  
**Course:** HY578 - Digital Speech Signal Processing  
**Algorithm:** LPC Analysis/Synthesis with Noise Excitation

