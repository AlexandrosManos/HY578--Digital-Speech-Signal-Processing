import numpy as np

# Uniform mid-rise scalar quantizer for LPC gains.
def uniform_scalar_quantize(gain, bits, gain_min, gain_max):
    if gain_max <= gain_min:
        raise ValueError("gain_max must be larger than gain_min")
    levels = 2 ** bits
    if levels < 2:
        raise ValueError("bits must be at least 1")
    step = (gain_max - gain_min) / (levels - 1)
    if step == 0:
        return gain_min, 0, step
    index = int(round((gain - gain_min) / step))
    index = max(0, min(levels - 1, index))
    quantized_gain = gain_min + index * step
    return quantized_gain, index, step

# Quantize an array of gains using the same parameters.
def uniform_scalar_quantize_array(gains, bits, gain_min, gain_max):
    gains = np.asarray(gains, dtype=np.float64)
    quantized = np.empty_like(gains)
    indices = np.empty(gains.shape, dtype=np.int32)
    step = None
    for i, g in np.ndenumerate(gains):
        q_gain, q_index, step = uniform_scalar_quantize(g, bits, gain_min, gain_max)
        quantized[i] = q_gain
        indices[i] = q_index
    return quantized, indices, step
