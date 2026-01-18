"""
AETHERMOORE Constants Module

Centralizes all constants used across the SCBE-AETHERMOORE framework:
- Mathematical constants (PHI, harmonic ratios)
- Security parameters
- Scaling functions

These constants are derived from mathematical principles and musical ratios,
providing a consistent foundation for the governance system.
"""

import math
import numpy as np
from typing import Tuple


# =============================================================================
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# =============================================================================

# Golden Ratio - φ = (1 + √5) / 2
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.6180339887

# Perfect Fifth Ratio - R = 3/2 (musical harmony)
R_FIFTH = 3.0 / 2.0  # = 1.5

# AETHERMOORE-specific constants
PHI_AETHER = PHI * R_FIFTH  # ≈ 2.427 - Golden ratio scaled by perfect fifth
LAMBDA_ISSAC = 1.0 / PHI  # ≈ 0.618 - Inverse golden ratio (ISSAC coefficient)
OMEGA_SPIRAL = 2 * math.pi / PHI  # ≈ 3.883 - Golden angle in radians


# =============================================================================
# DEFAULT SECURITY PARAMETERS
# =============================================================================

# Default harmonic ratio for scaling
DEFAULT_R = R_FIFTH  # 1.5

# Maximum security dimension (6D for V₆ space)
DEFAULT_D_MAX = 6

# Base security bits (AES-128 equivalent)
DEFAULT_BASE_BITS = 128


# =============================================================================
# HARMONIC SCALING FUNCTIONS
# =============================================================================

def harmonic_scale(d: int, R: float = DEFAULT_R) -> float:
    """
    Compute harmonic scaling factor H(d, R) = R^(d²).

    This is the core AETHERMOORE scaling function that provides
    super-exponential security enhancement.

    Args:
        d: Security dimension (1-6)
        R: Harmonic ratio (default 1.5)

    Returns:
        Scaling factor H(d, R)

    Examples:
        >>> harmonic_scale(1, 1.5)  # R^1 = 1.5
        1.5
        >>> harmonic_scale(6, 1.5)  # R^36 ≈ 2.18M
        2184164.406...
    """
    return R ** (d * d)


def security_bits(base_bits: int, d: int, R: float = DEFAULT_R) -> float:
    """
    Calculate effective security bits with harmonic enhancement.

    S_effective = S_base + d² × log₂(R)

    Args:
        base_bits: Base security level in bits
        d: Security dimension
        R: Harmonic ratio

    Returns:
        Effective security bits
    """
    return base_bits + (d * d) * math.log2(R)


def security_level(effective_bits: float) -> str:
    """
    Get human-readable security level name.

    Args:
        effective_bits: Effective security bits

    Returns:
        Security level name
    """
    if effective_bits >= 400:
        return "MAXIMUM"
    elif effective_bits >= 300:
        return "CRITICAL"
    elif effective_bits >= 250:
        return "HIGH"
    elif effective_bits >= 200:
        return "ELEVATED"
    elif effective_bits >= 150:
        return "STANDARD"
    else:
        return "BASIC"


def harmonic_distance(
    v1: Tuple[float, ...],
    v2: Tuple[float, ...],
    R: float = DEFAULT_R
) -> float:
    """
    Compute harmonic-weighted Euclidean distance between 6D vectors.

    Each dimension is weighted by R^k where k is the dimension index,
    giving higher dimensions more influence on the distance.

    Args:
        v1: First 6D vector
        v2: Second 6D vector
        R: Harmonic ratio

    Returns:
        Weighted distance
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same dimension")

    total = 0.0
    for k, (a, b) in enumerate(zip(v1, v2)):
        weight = R ** k
        total += weight * (a - b) ** 2

    return math.sqrt(total)


# =============================================================================
# HYPERBOLIC GEOMETRY CONSTANTS
# =============================================================================

# Poincaré ball curvature
POINCARE_CURVATURE = -1.0

# Langues (6 Sacred Tongues) dimension count
LANGUES_DIMENSIONS = 6

# Epsilon thresholds for metric coupling
EPSILON_THRESHOLD_HARMONIC = 1.0 / (2 * PHI ** 17)  # ≈ 3.67e-4
EPSILON_THRESHOLD_UNIFORM = 1.0 / (2 * LANGUES_DIMENSIONS)  # ≈ 0.083


# =============================================================================
# GOVERNANCE THRESHOLDS
# =============================================================================

# Snap threshold (geometric divergence)
EPSILON_SNAP = 1.5

# Coherence minimum
TAU_COHERENCE = 0.9

# Entropy bounds
ETA_MIN = -2.0  # Allows negentropy (high structure)
ETA_MAX = 6.0   # Maximum entropy (high disorder)
ETA_TARGET = 4.0  # Target entropy

# Curvature bounds
KAPPA_MAX = 0.1
KAPPA_TAU_MAX = 0.1
KAPPA_ETA_MAX = 0.1

# Lyapunov stability bound
LAMBDA_LYAPUNOV_BOUND = 0.001

# Harmonic scaling maximum
H_MAX = 10.0

# Time flow minimum (causality)
DOT_TAU_MIN = 0.0


# =============================================================================
# CRYPTOGRAPHIC CONSTANTS
# =============================================================================

# Nonce size in bytes
NONCE_BYTES = 12

# Default key length
KEY_LENGTH = 32

# HMAC output size
HMAC_SIZE = 32


# =============================================================================
# AUDIO/SIGNAL CONSTANTS
# =============================================================================

# Carrier frequency (A440)
CARRIER_FREQ = 440.0

# Sample rate
SAMPLE_RATE = 44100

# Default signal duration
DURATION = 0.5


# =============================================================================
# SIX SACRED TONGUES
# =============================================================================

TONGUES = ["KO", "AV", "RU", "CA", "UM", "DR"]
TONGUE_WEIGHTS = [PHI ** k for k in range(6)]
