# Utility functions for LPC coefficient domain conversions.
#
# This module provides helpers for converting between standard LPC predictor
# coefficients, reflection coefficients, and their logarithmically companded
# variants. The transforms follow the step-up and step-down recursions that
# underpin Levinson-Durbin style algorithms.

import numpy as np


# Return *values* as a float64 NumPy array.
def _as_float_array(values):
    return np.asarray(list(values), dtype=np.float64)



# Convert LPC predictor coefficients ``a`` to reflection coefficients ``k``.
def lpc_to_reflection(a_coeffs):

    a = _as_float_array(a_coeffs)
    if a.ndim != 1:
        raise ValueError("LPC coefficients must be a one-dimensional sequence")
    if len(a) < 2:
        raise ValueError("At least two LPC coefficients are required")

    # Normalise so that a[0] == 1
    a /= a[0]
    order = len(a) - 1

    k = np.zeros(order, dtype=np.float64)
    a_curr = a.copy()
    for m in range(order, 0, -1):
        k_m = a_curr[m]
        if not np.isfinite(k_m) or abs(k_m) >= 1.0:
            raise ValueError(
                f"Unstable reflection coefficient encountered at order {m}: {k_m}"
            )
        k[m - 1] = k_m
        if m == 1:
            break
        denom = 1.0 - k_m**2
        if denom <= 0.0:
            raise ValueError(
                f"Invalid denominator encountered while converting at order {m}"
            )
        next_a = np.empty(m, dtype=np.float64)
        next_a[0] = 1.0
        for i in range(1, m):
            next_a[i] = (a_curr[i] - k_m * a_curr[m - i]) / denom
        a_curr = next_a

    return k


def reflection_to_lpc(reflection_coeffs):
    # Convert reflection coefficients ``k`` back to LPC predictor coefficients.
    k = _as_float_array(reflection_coeffs)
    if k.ndim != 1:
        raise ValueError("Reflection coefficients must be a one-dimensional sequence")
    if not np.all(np.isfinite(k)):
        raise ValueError("Reflection coefficients must be finite")
    if np.any(np.abs(k) >= 1.0):
        raise ValueError("All reflection coefficients must satisfy |k| < 1")

    order = len(k)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    for m in range(1, order + 1):
        a_prev = a.copy()
        a[m] = k[m - 1]
        for i in range(1, m):
            a[i] = a_prev[i] + k[m - 1] * a_prev[m - i]

    return a


# Apply logarithmic companding to reflection coefficients ``k``.
def reflection_to_companded(reflection_coeffs):

    k = _as_float_array(reflection_coeffs)
    if np.any(np.abs(k) >= 1.0):
        raise ValueError("Reflection coefficients must satisfy |k| < 1")

    return np.log((1.0 - k) / (1.0 + k))


# Invert the companding transform returning reflection coefficients ``k``.
def companded_to_reflection(companded_coeffs):

    g = _as_float_array(companded_coeffs)
    if not np.all(np.isfinite(g)):
        raise ValueError("Companded coefficients must be finite")

    exp_g = np.exp(g)
    return (1.0 - exp_g) / (1.0 + exp_g)


# Convenience wrapper: LPC ``a`` -> companded reflection coefficients ``g``.
def lpc_to_companded(a_coeffs):

    k = lpc_to_reflection(a_coeffs)
    return reflection_to_companded(k)


# Convenience wrapper: companded reflection coefficients ``g`` -> LPC ``a``.
def companded_to_lpc(companded_coeffs):

    k = companded_to_reflection(companded_coeffs)
    return reflection_to_lpc(k)


# Clamp reflection coefficients inside the open unit interval.
def stabilize_reflection(reflection_coeffs, eps=1e-10):

    k = _as_float_array(reflection_coeffs)
    return np.clip(k, -1.0 + eps, 1.0 - eps)
