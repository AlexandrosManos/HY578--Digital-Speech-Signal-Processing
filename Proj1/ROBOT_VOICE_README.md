# Robotic Voice Generation - LPC Project

## Overview
Creates robotic voice by replacing natural excitation signal with artificial periodic pulse train while preserving formant structure through LPC filtering.

## Implementation
**File**: `LPCrobot.py` (158 lines)

### Key Modification
**Original (line 58)**: 
```python
ex = lfilter(a, [1], sigLPC)  # natural residual
```

**Modified**:
```python
ex = create_periodic_excitation(frame_len, pitch_period)  # artificial periodic pulses
```

### Artificial Excitation
```python
def create_periodic_excitation(frame_length, pitch_period):
    excitation = np.zeros(frame_length)
    excitation[::pitch_period] = 1.0  # impulses every pitch_period samples
    return excitation
```

## Generated Files

All files are saved in `robot_output/` directory.

### Audio Files (5)

**LPC Order Comparison** (constant pitch = 80 samples = 200 Hz):
1. **`robot_order12_pitch80.wav`** - Order 12 (lower than 24)
2. **`robot_order24_pitch80.wav`** - Order 24 (baseline)
3. **`robot_order36_pitch80.wav`** - Order 36 (higher than 24)

**Pitch Variation** (constant order = 24):
4. **`robot_order24_pitch50.wav`** - High pitch (320 Hz)
5. **`robot_order24_pitch120.wav`** - Low pitch (133 Hz)

### Spectrograms (3)

1. **`spectrogram_original_vs_robot.png`** - Comparison of original speech vs robot voice
   - Shows natural pitch variation vs constant periodic excitation
   - Clear harmonic structure in robot voice
   
2. **`spectrogram_order_comparison.png`** - LPC orders 12, 24, 36
   - Demonstrates effect of LPC order on spectral resolution
   - Higher order = more detailed formant structure
   
3. **`spectrogram_pitch_comparison.png`** - Pitch periods 50, 80, 120
   - Shows harmonic spacing differences with pitch
   - Higher pitch = wider harmonic spacing

## Observations

### Effect of LPC Order
- **Order 12**: Insufficient spectral detail → more buzzy, harsh robotic sound
- **Order 24**: Balanced formant preservation → clear robotic voice
- **Order 36**: High spectral detail → less robotic, more natural formants

### Effect of Pitch Period
- **50 samples (320 Hz)**: High-pitched robot (cartoon-like)
- **80 samples (200 Hz)**: Medium pitch (classic robot voice)
- **120 samples (133 Hz)**: Low-pitched robot (deeper, more authoritative)

### Spectrogram Observations
- **Regular harmonic structure**: Robot voice shows evenly-spaced harmonics (striped pattern)
- **Natural speech**: Irregular pitch contours and varying harmonics
- **Pitch effect**: Lower pitch = closer harmonics, higher pitch = wider harmonic spacing
- **Formant preservation**: All versions maintain similar formant bands (yellow regions)

## Design Choices

### 1. Pulse Train Structure
**Choice**: Single-sample impulses (`k=1`)

**Alternatives considered**:
- `k=2-4`: Multiple consecutive ones
- Count peaks in residual signal

**Justification**: Single impulses produce clearest robotic effect with well-defined harmonics at multiples of fundamental frequency.

### 2. Pitch Period Values
**Choice**: 50, 80, 120 samples

**Justification**:
- 80 samples (~200 Hz) approximates typical male pitch
- 50 samples (higher) and 120 samples (lower) show pitch effect
- All values create stable periodic excitation

### 3. LPC Orders Tested
**Choice**: 12, 24, 36

**Justification**:
- Order 12: Half of baseline → shows under-modeling
- Order 24: Standard (Fs/1000 + 16) → baseline
- Order 36: 1.5× baseline → shows over-modeling

### 4. Energy Normalization
**Choice**: Maintain frame-by-frame energy matching

**Justification**: Artificial excitation has different energy distribution than natural residual, so normalization ensures consistent loudness.

## Technical Details

### Why It Sounds Robotic
1. **Constant pitch**: No natural prosody or pitch variation
2. **Perfect periodicity**: No jitter or shimmer (natural voice variations)
3. **Sharp harmonics**: Impulse train creates strong harmonics at exact multiples of F0
4. **Preserved formants**: LPC filter maintains vowel quality (intelligibility)

### Source-Filter Theory
- **Source** (excitation): Changed from natural → periodic (creates robotic quality)
- **Filter** (LPC): Unchanged, preserves formants (maintains intelligibility)

## Usage
```bash
cd Proj1/
python LPCrobot.py
```

Generates:
- **5 WAV files**: Robotic voices with different LPC orders and pitch periods
- **3 Spectrograms**: Visual comparison of original vs robot and different parameters

All output files are saved in `robot_output/` directory.

