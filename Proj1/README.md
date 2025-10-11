# LPC Analysis Project - HY578

## Project Overview

This project implements **Linear Predictive Coding (LPC)** analysis and synthesis of speech using the **Levinson-Durbin recursion algorithm**.

## Files

### Code Files
- **`Proj1_csd5111.py`** - Main implementation file containing:
  - `my_levinson()` - Levinson-Durbin recursion implementation
  - `lpc_as_toyou()` - LPC analysis-synthesis function
  - `analyze_lpc_frame()` - Frame-by-frame spectrum comparison
  - `frame_by_frame_analysis()` - Automated analysis routine

### Data Files
- **`Speech Sample.wav`** - Test audio file for analysis
- **`HY578 Project LPC.pdf`** - Project tutorial and theoretical background

### Analysis Results
- **`lpc_analysis_plots/`** - Directory containing 25 comparison plots
  - Format: `frame_XXXX_{voiced|unvoiced}_order_YY.png`
  - Each plot shows FFT vs LPC frequency response comparison
  
### Documentation
- **`LPC_Analysis_Observations.md`** - Detailed observations and comments on analysis results
- **`.cursor/rules/lpc-levinson-recursion.mdc`** - Cursor AI rule for project context

## How to Run

### Option 1: Interactive Mode
```bash
python Proj1_csd5111.py
```
Then follow the prompts to choose:
1. Audio Demo (listen to synthesis results)
2. Frame-by-Frame Analysis (generate comparison plots)
3. Both

### Option 2: Command Line Mode
```bash
# Audio demo only
python Proj1_csd5111.py 1

# Frame-by-frame analysis with default orders (8,16,24,32,40)
python Proj1_csd5111.py 2

# Frame-by-frame analysis with custom orders
python Proj1_csd5111.py 2 "10,20,30"

# Both modes
python Proj1_csd5111.py 3
```

## Analysis Results Summary

### Frames Analyzed
- **Total frames:** 265
- **Voiced frames:** 32 (vowel sounds, high energy, low zero-crossing)
- **Unvoiced frames:** 103 (fricatives/consonants, noise-like)
- **Silence/transition:** 130

### Selected Representative Frames
- **Voiced:** Frames 110, 125, 171
- **Unvoiced:** Frames 68, 127

### LPC Orders Tested
8, 16, 24, 32, 40

### Key Findings

✅ **Order 24 is optimal** for general-purpose speech processing
- Captures 5-6 formants in voiced speech
- Provides good spectral envelope for unvoiced speech
- Balances accuracy and computational cost

✅ **LPC successfully models vocal tract resonances**
- All-pole filter captures formant structure
- Smooth envelope matches FFT peaks
- Validates source-filter model of speech production

✅ **Voiced vs. Unvoiced differences**
- Voiced speech: Order 16-24 sufficient
- Unvoiced speech: Benefits from order 24-32

## Viewing Results

### Recommended Plots to Examine

1. **Evolution of order (voiced):**
   - `frame_0110_voiced_order_8.png`
   - `frame_0110_voiced_order_16.png`
   - `frame_0110_voiced_order_24.png`
   - `frame_0110_voiced_order_32.png`

2. **Best examples:**
   - `frame_0171_voiced_order_24.png` - Excellent formant tracking
   - `frame_0125_voiced_order_24.png` - Clear F1-F5 formants

3. **Unvoiced comparison:**
   - `frame_0127_unvoiced_order_8.png` - Insufficient detail
   - `frame_0127_unvoiced_order_32.png` - Better spectral match

4. **Overfitting example:**
   - `frame_0125_voiced_order_40.png` - Too high, models harmonics

### Plot Interpretation

Each plot contains:
- **Top subplot:** Linear magnitude spectrum
  - Blue line: FFT of windowed speech frame
  - Red line: LPC filter frequency response H(z) = 1/A(z)
  
- **Bottom subplot:** dB magnitude spectrum
  - Same comparison in logarithmic scale
  - Better for viewing dynamic range and formants

**What to look for:**
- LPC envelope should follow FFT peaks (formants)
- LPC should be smooth, not matching individual harmonics
- Higher orders give more detail but risk overfitting

## Dependencies

```bash
pip install numpy scipy matplotlib sounddevice
```

## Algorithm Implementation

The `my_levinson()` function implements the Levinson-Durbin recursion:

**Input:** 
- `r` - autocorrelation sequence
- `order` - LPC order (typically 24)

**Output:**
- `a` - LPC filter coefficients [1, -l₁, -l₂, ..., -lₚ]

**Algorithm steps:**
1. Initialize: E⁰ = r[0]
2. For i = 1 to p:
   - Compute reflection coefficient kᵢ
   - Update prediction coefficients lⱼⁱ
   - Update prediction error Eⁱ
3. Return a = [1, -l₁ᵖ, -l₂ᵖ, ..., -lₚᵖ]

See `.cursor/rules/lpc-levinson-recursion.mdc` for detailed algorithm documentation.

## Project Structure

```
Proj1/
├── Proj1_csd5111.py              # Main implementation
├── Speech Sample.wav              # Test audio
├── HY578 Project LPC.pdf         # Tutorial
├── README.md                      # This file
├── LPC_Analysis_Observations.md  # Detailed analysis results
├── lpc_analysis_plots/           # 25 comparison plots
│   ├── frame_0068_unvoiced_order_*.png
│   ├── frame_0110_voiced_order_*.png
│   ├── frame_0125_voiced_order_*.png
│   ├── frame_0127_unvoiced_order_*.png
│   └── frame_0171_voiced_order_*.png
└── .cursor/
    └── rules/
        └── lpc-levinson-recursion.mdc
```

## Credits

- **Course:** HY578 - Speech & Audio Signal Processing
- **Original MATLAB version:** Yannis Stylianou
- **Python version:** Alex Angelakis, 2025
- **Student implementation:** CSD-5111

## References

- HY578 Project LPC.pdf - Theoretical background and algorithm specification
- LPC_Analysis_Observations.md - Detailed analysis and observations

