"""
SCBE Context Commitment

Turns 6 environmental measurements into a unique cryptographic fingerprint.
Implements both standard and harmonic-enhanced context commitment.

Reference: Section 1.1 of SCBE-AETHER-UNIFIED-2026-001
Claims: 1(b), 53
"""

import hashlib
import struct
from dataclasses import dataclass
from typing import Tuple, Optional
from .constants import PERFECT_FIFTH, HARMONIC_METRIC_TENSOR


@dataclass
class ContextVector:
    """
    The 6-dimensional context vector for SCBE authorization.

    SCBE dimensions:
        c₁: time - Unix timestamp
        c₂: device_id - Numeric device identifier
        c₃: threat_level - Current danger level (0-10)
        c₄: entropy - System randomness (0-1)
        c₅: server_load - System busyness (0-1)
        c₆: behavior_stability - How "normal" actions are (0-1)

    AETHERMOORE isomorphism:
        x: position_x (temporal position)
        y: position_y (identity coordinate)
        z: position_z (risk altitude)
        v: velocity (state velocity)
        p: priority (processing priority)
        s: security (security dimension)
    """
    time: float
    device_id: float
    threat_level: float
    entropy: float
    server_load: float
    behavior_stability: float

    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Convert to tuple for computation."""
        return (
            self.time,
            self.device_id,
            self.threat_level,
            self.entropy,
            self.server_load,
            self.behavior_stability
        )

    def to_aethermoore(self) -> dict:
        """Map to AETHERMOORE 6D vector notation."""
        return {
            "x": self.time,
            "y": self.device_id,
            "z": self.threat_level,
            "v": self.entropy,
            "p": self.server_load,
            "s": self.behavior_stability
        }

    def weighted_tuple(self, R: float = PERFECT_FIFTH) -> Tuple[float, ...]:
        """
        Apply harmonic weighting to context components.

        Weights: (1, 1, 1, R, R², R³)
        This makes behavior_stability contribute 3.375× more than time.
        """
        return (
            self.time,
            self.device_id,
            self.threat_level,
            R * self.entropy,
            (R ** 2) * self.server_load,
            (R ** 3) * self.behavior_stability
        )

    def validate(self) -> Tuple[bool, list]:
        """
        Validate context vector components are in expected ranges.

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        if self.time < 0:
            errors.append("time must be non-negative")

        if self.device_id < 0:
            errors.append("device_id must be non-negative")

        if not 0 <= self.threat_level <= 10:
            errors.append("threat_level must be in [0, 10]")

        if not 0 <= self.entropy <= 1:
            errors.append("entropy must be in [0, 1]")

        if not 0 <= self.server_load <= 1:
            errors.append("server_load must be in [0, 1]")

        if not 0 <= self.behavior_stability <= 1:
            errors.append("behavior_stability must be in [0, 1]")

        return (len(errors) == 0, errors)


def context_commitment(context: ContextVector) -> bytes:
    """
    Standard context commitment: SHA256(c₁, c₂, c₃, c₄, c₅, c₆)

    Produces a 256-bit cryptographic fingerprint of the context.

    Args:
        context: The 6-dimensional context vector

    Returns:
        32-byte SHA256 hash

    Reference: Section 1.1 "The math"
    """
    # Pack all 6 floats as little-endian doubles
    packed = struct.pack(
        '<6d',
        context.time,
        context.device_id,
        context.threat_level,
        context.entropy,
        context.server_load,
        context.behavior_stability
    )
    return hashlib.sha256(packed).digest()


def harmonic_context_commitment(
    context: ContextVector,
    R: float = PERFECT_FIFTH
) -> bytes:
    """
    Harmonic-enhanced context commitment: SHA256(c₁, c₂, c₃, R·c₄, R²·c₅, R³·c₆)

    Weights higher dimensions more heavily, making behavior_stability
    contribute 3.375× more entropy than time.

    Args:
        context: The 6-dimensional context vector
        R: Harmonic ratio (default: 1.5)

    Returns:
        32-byte SHA256 hash

    Reference: Section 1.1 AETHERMOORE Enhancement
    """
    weighted = context.weighted_tuple(R)
    packed = struct.pack('<6d', *weighted)
    return hashlib.sha256(packed).digest()


def context_distance(
    c1: ContextVector,
    c2: ContextVector,
    use_harmonic_metric: bool = True
) -> float:
    """
    Compute distance between two context vectors.

    Args:
        c1, c2: Context vectors to compare
        use_harmonic_metric: If True, use harmonic metric tensor

    Returns:
        Distance between contexts
    """
    from .harmonic import harmonic_metric_distance

    t1 = c1.to_tuple()
    t2 = c2.to_tuple()

    if use_harmonic_metric:
        return harmonic_metric_distance(t1, t2, HARMONIC_METRIC_TENSOR)
    else:
        # Euclidean distance
        return sum((a - b) ** 2 for a, b in zip(t1, t2)) ** 0.5


def derive_chaos_params(
    context: ContextVector,
    key: bytes,
    use_harmonic: bool = True
) -> Tuple[float, float]:
    """
    Derive chaos map parameters (r, x₀) from context and key.

    The parameters are derived such that:
    - r is in the chaotic regime [3.97, 4.0)
    - x₀ is in (0, 1)

    Args:
        context: The context vector
        key: Encryption key
        use_harmonic: Whether to use harmonic commitment

    Returns:
        Tuple of (r, x₀) for the logistic map

    Reference: Section 1.4
    """
    if use_harmonic:
        commitment = harmonic_context_commitment(context)
    else:
        commitment = context_commitment(context)

    # Combine commitment with key
    combined = hashlib.sha256(commitment + key).digest()

    # Extract r from first 8 bytes, scale to [3.97, 4.0)
    r_raw = struct.unpack('<Q', combined[:8])[0]
    r_normalized = r_raw / (2**64)  # [0, 1)
    r = 3.97 + r_normalized * 0.03  # [3.97, 4.0)

    # Extract x₀ from next 8 bytes, scale to (0, 1)
    x0_raw = struct.unpack('<Q', combined[8:16])[0]
    x0 = 0.001 + (x0_raw / (2**64)) * 0.998  # (0, 1)

    return (r, x0)


def context_commitment_hex(context: ContextVector, use_harmonic: bool = True) -> str:
    """Return context commitment as hex string."""
    if use_harmonic:
        return harmonic_context_commitment(context).hex()
    return context_commitment(context).hex()


def verify_context_binding(
    context: ContextVector,
    expected_commitment: bytes,
    use_harmonic: bool = True,
    tolerance: float = 0.0
) -> bool:
    """
    Verify that a context matches an expected commitment.

    Args:
        context: Context to verify
        expected_commitment: Expected hash
        use_harmonic: Whether to use harmonic commitment
        tolerance: If > 0, allows fuzzy matching (not recommended)

    Returns:
        True if context matches commitment
    """
    if use_harmonic:
        actual = harmonic_context_commitment(context)
    else:
        actual = context_commitment(context)

    if tolerance == 0:
        return actual == expected_commitment
    else:
        # Fuzzy matching: count matching bytes
        matching = sum(a == b for a, b in zip(actual, expected_commitment))
        return matching >= len(actual) * (1 - tolerance)
