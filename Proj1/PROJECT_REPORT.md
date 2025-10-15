# HY578 LPC Project - Complete Observations Report

**Student:** CSD5111 - CSD5136  
**Course:** HY578 - Digital Speech Signal Processing  
**Date:** October 2025

---

## Question 1: LPC Implementation (my_levinson and lpc_as_toyou)

### Implementation
Implemented **Levinson-Durbin recursion** algorithm to compute LPC coefficients from autocorrelation, and LPC analysis-synthesis system.

### Levinson-Durbin Algorithm
- **Input:** Autocorrelation sequence r[0...p], order p
- **Output:** LPC coefficients a = [1, -l₁, -l₂, ..., -lₚ]
- **Algorithm steps:**
  1. Initialize E⁰ = r[0]
  2. For i=1 to p: compute reflection coefficient kᵢ
  3. Update prediction coefficients lⱼⁱ
  4. Update prediction error Eⁱ = (1 - kᵢ²)Eⁱ⁻¹
  5. Return final coefficients with negative sign for IIR filter

### LPC Analysis-Synthesis System
- **Frame length:** 30ms (480 samples at 16kHz)
- **Frame shift:** 15ms (50% overlap)
- **Window:** Hanning
- **LPC order:** 24
- **Process:** 
  1. Extract frame and apply window
  2. Compute autocorrelation
  3. Use my_levinson to get LPC coefficients a
  4. Compute gain G
  5. Extract excitation: ex = lfilter(a, [1], frame)
  6. Synthesize: s = lfilter([G], a, ex)
  7. Energy normalize and overlap-add

### Observations
- **Synthesis quality:** LPC synthesis produces intelligible speech very similar to original
- **Levinson recursion:** Correctly computes LPC coefficients efficiently (O(p²) complexity)
- **Energy preservation:** Energy normalization maintains loudness across frames
- **Overlap-add:** 50% overlap prevents artifacts at frame boundaries
- **Order 24:** Adequate for capturing vocal tract characteristics at 16kHz

### Key Findings
- LPC analysis-synthesis successfully reconstructs speech
- Source-filter separation works: excitation + LPC filter = speech
- Levinson-Durbin provides efficient solution to normal equations
- System validates feasibility of LPC-based speech coding

---

## Question 2: Comparison and Analysis (LPC Filter vs FFT)

### Implementation
Performed frame-by-frame comparison of LPC filter frequency response H(z) = G/A(z) against FFT magnitude spectrum of speech frames.

### Method
- **Frame analysis:** 30ms Hanning window
- **Frames selected:** 
  - Voiced frame 33 (high energy, low zero-crossing)
  - Unvoiced frame 27 (moderate energy, high zero-crossing)
- **LPC orders tested:** 8, 16, 24, 32, 40
- **Visualization:** FFT spectrum (blue) vs LPC filter response (red)

### Observations

**Voiced Frame (Frame 33):**

- **Order 8:** 
  - Captures only F1-F2
  - Smooth envelope but insufficient spectral detail
  - Misses higher formants completely

- **Order 16:** 
  - Captures F1-F3 well, partial F4
  - Good compromise for basic processing
  - Envelope follows main resonances

- **Order 24:** 
  - **Optimal:** Captures F1-F5 with excellent detail
  - Smooth envelope without overfitting
  - Best match to FFT peaks (formants)
  - Follows design rule: Order ≈ Fs/1000 + 4-6

- **Order 32:** 
  - Captures all formants plus fine details
  - Begins following individual harmonics
  - Slightly over-parameterized

- **Order 40:** 
  - **Overfitting:** Models individual harmonics, not just formants
  - Excessive computational cost
  - Violates all-pole assumption for vocal tract only

**Unvoiced Frame (Frame 27):**

- **Lower orders (8-16):** Very smooth, miss noise spectral structure
- **Order 24:** Better captures broadband characteristics
- **Higher orders (32-40):** Best for unvoiced sounds, captures fine spectral details

**LPC as Spectral Envelope:**
- LPC provides smooth spectral envelope
- Should model formants (vocal tract resonances), not harmonics (pitch)
- All-pole model naturally emphasizes peaks
- Higher orders risk over-modeling

### Key Findings
- **LPC successfully models vocal tract resonances** as smooth spectral envelope
- **Order 24 optimal** for 16kHz: captures 5-6 formants without overfitting
- **Voiced speech:** Lower orders (16-24) sufficient
- **Unvoiced speech:** Benefits from higher orders (24-32) for noise spectrum
- **All-pole model effective:** Accurately represents formant structure with smooth envelope

---

## Question 3: Whisper and Robotic Voice

### Part A: Whisper Voice Synthesis

#### Implementation
Replaced natural excitation with **white Gaussian noise** while preserving LPC filter coefficients.

**Code modification:**
```python
# Original: ex = lfilter(a, [1], frame)  # natural residual
# Modified: ex = np.random.randn(frame_len)  # noise excitation
```

#### Orders Tested
- Order 12 (lower)
- Order 24 (baseline)
- Order 36 (higher)

#### Observations

**Order 12:** 
- Captures F1-F2, lacks higher formant detail
- Smooth, slightly muffled whisper
- Adequate for basic whisper, less natural

**Order 24:** 
- Captures F1-F4 with good resolution
- **Most natural-sounding whisper**
- Clear articulation, optimal balance

**Order 36:**
- All formants plus fine details
- Detailed but slightly harsh texture
- Diminishing returns over order 24

**Spectrograms:**
- No pitch harmonics → noise-like texture
- Formants preserved → resonant bands visible
- Speech timing maintained → intelligible

#### Key Findings (Whisper)
- Noise excitation successfully removes pitch
- Formants preserved → speech remains intelligible
- Source-filter validated: changing excitation alone transforms character
- Order 24 optimal for naturalness

---

### Part B: Robotic Voice

#### Implementation
Replaced natural excitation with **periodic impulse train** (constant pitch).

**Code modification:**
```python
# Original: ex = lfilter(a, [1], frame)  # natural residual
# Modified: ex = np.zeros(frame_len); ex[::pitch_period] = 1.0  # periodic impulses
```

#### Configurations Tested

**LPC Order Variation** (pitch = 80 samples = 200 Hz):
- Order 12, 24, 36

**Pitch Variation** (order = 24):
- 50 samples (320 Hz) - high pitch
- 80 samples (200 Hz) - medium pitch
- 120 samples (133 Hz) - low pitch

#### Observations

**Effect of LPC Order:**
- **Order 12:** Buzzy, harsh, less formant detail
- **Order 24:** Clear robotic voice, optimal balance
- **Order 36:** More natural formants, less robotic

**Effect of Pitch Period:**
- **50 samples (320 Hz):** High-pitched, cartoon-like robot
- **80 samples (200 Hz):** Classic robot voice, neutral
- **120 samples (133 Hz):** Deep, authoritative robot

**Spectrograms:**
- **Original:** Irregular pitch contours, varying harmonics
- **Robot:** Perfectly regular, evenly-spaced harmonics (striped pattern)
- **Pitch effect:** Lower pitch = denser stripes; higher pitch = wider stripes
- **Formants preserved:** Resonant bands maintained

#### Design Choices

**Single-sample impulses (k=1):** 
- Produces clearest robotic effect with sharp harmonics
- k=2-4 would blur harmonic structure

**Pitch period 80 samples (200 Hz):** 
- Neutral robotic character, classic robot voice
- Approximates typical male pitch

**Why robotic:**
- Constant pitch removes natural prosody
- Perfect periodicity creates mechanical quality
- No pitch variation = no emotion/intonation

#### Key Findings (Robot)
- Constant periodic excitation creates robotic effect
- Formants preserved → intelligibility maintained
- Source-filter model validated: source controls quality, filter controls content
- Pitch period controls robot "personality"

---

### Question 3 Summary

Both whisper and robotic voice demonstrate **source-filter separation:**

| Aspect | Whisper | Robot |
|--------|---------|-------|
| **Excitation** | White noise | Periodic impulses |
| **Filter** | LPC (unchanged) | LPC (unchanged) |
| **Effect** | No pitch, breathy | Constant pitch, mechanical |
| **Formants** | Preserved | Preserved |
| **Intelligibility** | Maintained | Maintained |

**Common insight:** Changing only excitation transforms voice character while LPC filter preserves linguistic content and speaker characteristics.

---

## Question 4: Formant Modification

### Implementation
Modified formant frequencies by adjusting **pole angles** to create younger and elderly voices.

### Method
1. Extract poles from LPC coefficients: `poles = np.roots(a)`
2. Select 3 most significant poles by magnitude (F1, F2, F3)
3. Modify pole angles: `θ_new = θ × (1 ± shift_percent/100)`
4. Maintain conjugate pairs for real-valued filter
5. Reconstruct filter: `np.convolve` to build A(z) = ∏(z - pᵢ)

### Transformations Generated
- **Elderly -20%:** Lower formants by 20%
- **Elderly -10%:** Lower formants by 10%
- **Younger +10%:** Raise formants by 10%
- **Younger +20%:** Raise formants by 20%

### Observations

**Elderly Voices (-10%, -20%):**
- Lower formant frequencies → deeper, more resonant quality
- -10%: Subtle aging, very natural
- -20%: Pronounced elderly effect, clearly older
- Voice sounds relaxed, lower-pitched
- Average shifts: F1 1190→952 Hz, F2 3749→3000 Hz, F3 2447→1957 Hz

**Younger Voices (+10%, +20%):**
- Higher formant frequencies → brighter, lighter quality
- +10%: Subtle youth, natural sounding
- +20%: Clear younger voice, energetic
- Voice sounds animated, higher-pitched
- Average shifts: F1 1190→1428 Hz, F2 3749→4437 Hz, F3 2447→2936 Hz

### Design Choices

**3 formants selected (F1, F2, F3):** 
- Most perceptually important for vowel quality and age
- Higher formants less impact on age perception
- Robust across different phonemes

**±10% and ±20% shifts:** 
- ±10%: Natural variation across age groups
- ±20%: Dramatic but intelligible
- Based on acoustic research on age-related changes

**Magnitude-based pole selection:** 
- Higher magnitude = stronger resonances (formants)
- Robust and efficient
- Consistently identifies true formants

**Angle modification only:**
- Changes frequency (angle) not bandwidth (magnitude)
- Preserves filter stability (poles stay inside unit circle)
- Maintains formant quality

**Why it works:**
- Formant frequencies inversely related to vocal tract length
- Children/youth: shorter tract → higher formants
- Elderly: lengthened tract, reduced tension → lower formants
- Shifting formants simulates these physical changes

### Key Findings
- Pole angle manipulation effectively shifts perceived age
- Conjugate pairs maintained → real-valued coefficients
- Excitation unchanged → preserves speaker identity and pitch
- Energy normalized → consistent loudness
- Naturalness preserved even at ±20% shifts
- Validates formant-age relationship from speech acoustics

---

## General Conclusions

### Source-Filter Model Validation

All experiments validate **source-filter theory:**

1. **Q1:** LPC separates excitation from vocal tract filter
2. **Q2:** All-pole filter models vocal tract resonances
3. **Q3:** Changing excitation (whisper/robot) transforms quality, not content
4. **Q4:** Changing filter (formants) transforms characteristics, not pitch

**Key principle:** Excitation and vocal tract operate independently.

### LPC Order Guidelines

| Order | Formants | Use Case | Notes |
|-------|----------|----------|-------|
| 8 | F1-F2 | Minimal | Insufficient for most tasks |
| 12 | F1-F2 | Basic synthesis | Adequate for simple applications |
| 16 | F1-F3 | Voiced speech | Good for telephony |
| **24** | **F1-F5** | **General purpose** | **Optimal for 16 kHz** |
| 32 | All + details | Unvoiced speech | Extra detail, some overfitting risk |
| 40 | Over-fitted | Not recommended | Models harmonics, not just formants |

**Rule:** Order ≈ Fs(kHz) + 4-6 poles → 16 + 8 = 24

### Practical Applications

1. **Speech coding:** LPC enables efficient compression (Q1)
2. **Speech analysis:** LPC reveals formant structure (Q2)
3. **Voice transformation:** Changing excitation creates effects (Q3)
4. **Speaker characteristics:** Formant manipulation changes age/gender (Q4)

### Technical Success

✅ Levinson-Durbin correctly computes LPC coefficients  
✅ LPC filter accurately models vocal tract resonances  
✅ Excitation modifications transform voice quality  
✅ Formant modifications shift perceived characteristics  
✅ All transformations maintain intelligibility  
✅ Source-filter separation validated experimentally  

---

## Files Generated

### Question 1: LPC Implementation
- **Code:** `my_levinson.py`, `LPC.py`
- **Output:** Synthesized speech (demonstrated in Q2)

### Question 2: Analysis Plots (10 files)
**Directory:** `plots/`
- Voiced frame 33: orders 8, 16, 24, 32, 40
- Unvoiced frame 27: orders 8, 16, 24, 32, 40
- Each shows FFT vs LPC comparison

### Question 3: Whisper + Robot (14 files)
**Whisper:** `whisper_output/`
- 3 audio files (orders 12, 24, 36)
- 3 spectrograms

**Robot:** `robot_output/`
- 5 audio files (order variations + pitch variations)
- 3 spectrograms (original vs robot, order comparison, pitch comparison)

### Question 4: Formant Modification (4 files)
**Directory:** `formant_output/`
- `elderly_minus20.wav`, `elderly_minus10.wav`
- `younger_plus10.wav`, `younger_plus20.wav`

**Total:** 28 output files

---

## Summary

This project successfully implements and validates Linear Predictive Coding (LPC) for speech analysis and synthesis:

**Q1:** Levinson-Durbin recursion and LPC analysis-synthesis system work correctly, enabling speech reconstruction.

**Q2:** LPC filter accurately models vocal tract as smooth spectral envelope capturing formant resonances, with order 24 optimal for 16 kHz.

**Q3:** Whisper (noise excitation) and robotic voice (periodic excitation) demonstrate that excitation type controls voice quality while LPC filter preserves intelligibility.

**Q4:** Formant modification through pole angle manipulation successfully shifts perceived speaker age by ±10-20% while maintaining naturalness.

**Key achievement:** All implementations validate the source-filter model, showing excitation and vocal tract can be modified independently while maintaining speech intelligibility.
