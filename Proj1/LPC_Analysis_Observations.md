# LPC Frame-by-Frame Analysis - Observations and Comments

## Analysis Summary

**Date:** October 10, 2025  
**Audio File:** Speech Sample.wav  
**Frames Analyzed:** 265 total (30ms windows, 15ms shift)  
**Frame Classification:**
- Voiced frames: 32
- Unvoiced frames: 103
- Silence/transition frames: 130

**Selected Frames for Analysis:**
- **Voiced:** Frames 110, 125, 171
- **Unvoiced:** Frames 68, 127

**LPC Orders Tested:** 8, 16, 24, 32, 40

---

## Key Observations

### 1. **Voiced Frames Analysis**

#### Frame 110, 125, 171 (Voiced Speech - Vowel Sounds)

**Characteristics:**
- High energy, low zero-crossing rate
- Clear harmonic structure visible in FFT
- Multiple sharp peaks representing pitch harmonics

**LPC Performance by Order:**

- **Order 8:** 
  - Captures only the broad spectral envelope
  - Identifies 3-4 main formants (resonances)
  - Smooth envelope but misses fine spectral details
  - Good for rough vocal tract shape estimation

- **Order 16:**
  - Better formant resolution
  - Captures 5-6 formants clearly
  - Envelope follows FFT peaks more closely
  - Good balance between smoothness and detail

- **Order 24 (Standard):**
  - Excellent formant tracking
  - Captures most important spectral features
  - Still maintains smooth envelope characteristic
  - **BEST PERFORMANCE for voiced speech**
  - Follows the spectral envelope without overfitting to harmonics

- **Order 32:**
  - Very detailed envelope
  - Starts to model some harmonic structure (not ideal)
  - May be capturing more than just vocal tract
  - Slightly overfit for typical speech analysis

- **Order 40:**
  - High detail but beginning to overfit
  - Models individual harmonics more than envelope
  - Computational cost increases significantly
  - Not recommended for standard LPC vocoding

**Key Insight:** For voiced speech, LPC order 24 provides optimal performance - it captures the vocal tract resonances (formants) without overfitting to the pitch harmonics.

---

### 2. **Unvoiced Frames Analysis**

#### Frame 68, 127 (Unvoiced Speech - Fricatives/Consonants)

**Characteristics:**
- Lower energy than voiced frames
- High zero-crossing rate
- Noise-like spectrum (no clear harmonics)
- Flatter, more distributed spectral energy

**LPC Performance by Order:**

- **Order 8:**
  - Very smooth, broad spectral estimate
  - Misses many spectral details
  - Captures only major resonances
  - Insufficient for unvoiced speech modeling

- **Order 16:**
  - Improved spectral detail
  - Captures more resonances
  - Still relatively smooth envelope

- **Order 24:**
  - Good representation of spectral shape
  - Captures important resonances
  - Reasonable match to FFT envelope

- **Order 32:**
  - Better tracking of spectral variations
  - More detailed than order 24
  - Follows noise spectrum contours more closely
  - **POTENTIALLY BETTER than order 24 for unvoiced**

- **Order 40:**
  - High spectral detail
  - Tracks fine variations in spectrum
  - May capture more noise characteristics
  - Computational cost may not justify improvement

**Key Insight:** Unvoiced speech benefits from slightly higher LPC orders (24-32) because the spectral structure is more complex and distributed compared to the harmonic structure of voiced speech.

---

## 3. **Comparison: FFT vs LPC Frequency Response**

### What the Plots Show:

**Blue Line (FFT of Speech Frame):**
- Shows the actual spectrum of the windowed speech frame
- For voiced: sharp harmonic peaks from vocal cord vibration
- For unvoiced: noise-like, distributed spectrum
- Includes both source (glottal/turbulence) and filter (vocal tract)

**Red Line (LPC Filter Response H(z) = 1/A(z)):**
- Shows the all-pole filter frequency response
- Models the **vocal tract filter** only
- Provides smooth spectral envelope
- Should follow the "peaks" or "envelope" of the FFT

### Key Observations:

1. **LPC as Envelope Estimator:**
   - LPC effectively estimates the spectral envelope
   - It smooths out the harmonic structure (voiced) or noise variations (unvoiced)
   - The all-pole model captures resonances (formants) well

2. **Source-Filter Model Validation:**
   - FFT = Source × Filter
   - LPC estimates Filter
   - The good match validates the source-filter model of speech production

3. **Formant Tracking:**
   - Clear formants (F1, F2, F3, F4, F5) visible in voiced frames
   - LPC peaks align with formant frequencies
   - Order 24 provides optimal formant resolution

---

## 4. **Effect of Increasing LPC Order**

### Trade-offs Observed:

**Advantages of Higher Order:**
- ✅ More detailed spectral representation
- ✅ Better formant resolution
- ✅ Captures higher frequency resonances
- ✅ Better for unvoiced/fricative sounds

**Disadvantages of Higher Order:**
- ❌ Increased computational cost (Levinson recursion: O(p²))
- ❌ Risk of overfitting to harmonics (voiced speech)
- ❌ May model excitation instead of just vocal tract
- ❌ Less robust to noise
- ❌ Potential numerical stability issues

### Recommended Orders:

| Speech Type | Recommended Order | Reasoning |
|-------------|------------------|-----------|
| Voiced (vowels) | 16-24 | Captures 4-6 formants optimally |
| Unvoiced (fricatives) | 24-32 | Needs more detail for complex spectrum |
| General purpose | **24** | Good balance, industry standard |
| Low-bitrate coding | 10-12 | Minimal representation |
| High-quality analysis | 32-40 | Maximum detail, research use |

---

## 5. **Effect of Decreasing LPC Order**

### Order 8 Analysis:

**Observations:**
- Very smooth spectral envelope
- Captures only 3-4 major resonances
- Misses higher formants (F4, F5)
- Insufficient for high-quality speech synthesis
- May be adequate for basic pitch/voicing detection

**Use Cases:**
- Very low bitrate applications
- Rough spectral shape estimation
- Real-time applications with strict latency requirements
- Formant tracking in controlled conditions

---

## 6. **Spectral Matching Quality**

### dB Scale Observations:

The dB plots (lower subplot) reveal:

1. **Dynamic Range:**
   - FFT shows ~40-50 dB dynamic range
   - LPC envelope follows major peaks well
   - Valleys don't need to match (they represent harmonics)

2. **Peak Tracking:**
   - Order 24+ tracks formant peaks within 2-3 dB
   - Lower orders may have 5-10 dB errors
   - Critical for perceptual quality

3. **High Frequency Behavior:**
   - LPC naturally provides spectral tilt
   - Matches the high-frequency roll-off in speech
   - Important for natural sound quality

---

## Conclusions

### Main Findings:

1. **LPC Order 24 is optimal for general speech processing:**
   - Captures 5-6 formants in voiced speech
   - Provides good envelope for unvoiced speech
   - Balances accuracy and computational cost
   - Industry standard for good reason

2. **The all-pole model works remarkably well:**
   - Successfully captures vocal tract resonances
   - Provides smooth spectral envelope
   - Validates source-filter theory of speech production

3. **Voiced vs. Unvoiced require different considerations:**
   - Voiced: Lower orders sufficient (harmonics shouldn't be modeled)
   - Unvoiced: Higher orders beneficial (complex spectrum)
   - Adaptive order could be beneficial

4. **Trade-off is fundamental:**
   - More poles = more detail but more complexity
   - Sweet spot around 16-32 for most applications
   - Task-dependent optimization needed

### Recommendations for Implementation:

- **Speech Coding:** Use order 10-16 (low bitrate) or 24 (toll quality)
- **Speech Recognition:** Order 12-16 (feature extraction)
- **Speech Synthesis:** Order 24-32 (high quality)
- **Formant Analysis:** Order 12-16 (avoid overfitting)
- **General Analysis:** Order 24 (as demonstrated in this project)

---

## Interesting Plot Highlights

**Recommended plots to examine:**

1. **`frame_0110_voiced_order_8.png` vs `frame_0110_voiced_order_24.png`**
   - Shows improvement from low to optimal order for voiced speech
   - Clear formant structure evolution

2. **`frame_0171_voiced_order_24.png`**
   - Excellent example of LPC envelope matching FFT peaks
   - Clear F1, F2, F3, F4 formants visible

3. **`frame_0127_unvoiced_order_8.png` vs `frame_0127_unvoiced_order_32.png`**
   - Demonstrates need for higher order in unvoiced speech
   - Shows spectral detail improvement

4. **`frame_0125_voiced_order_40.png`**
   - Example of potential overfitting
   - Order 40 starts modeling harmonics, not just envelope

---

**Analysis conducted using the Levinson-Durbin recursion algorithm as implemented in `my_levinson()` function.**

