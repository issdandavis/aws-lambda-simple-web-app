"""
Intent Configuration Module (AXIS 3)

Implements the Spiralverse vocabulary and intent-to-basin mapping.
Planetary frequencies seed the harmonic parameters.

Reference: Section 3.2 of SCBE-AETHER-UNIFIED-2026-001
Claims: 18, 52
"""

import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from .constants import (
    PERFECT_FIFTH,
    HARMONIC_FREQUENCY_MAP,
    D_MAJOR_7TH_CHORD,
    PLANETARY_FREQUENCIES
)
from .harmonic import harmonic_scaling


class PrimaryIntent(Enum):
    """Primary intent categories in the Spiralverse vocabulary."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELEGATE = "delegate"
    REVOKE = "revoke"
    QUERY = "query"
    TRANSFORM = "transform"
    VERIFY = "verify"


class IntentModifier(Enum):
    """Modifiers that adjust intent scope and behavior."""
    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    CONDITIONAL = "conditional"
    RECURSIVE = "recursive"
    BOUNDED = "bounded"
    UNLIMITED = "unlimited"
    AUDITED = "audited"
    SILENT = "silent"


@dataclass
class Intent:
    """
    A complete intent specification.

    Intent I = (primary, modifier, harmonic, phase)

    The harmonic parameter (1-7) selects the planetary frequency seed.
    """
    primary: PrimaryIntent
    modifier: IntentModifier
    harmonic: int  # 1-7, selects planetary frequency
    phase: float   # 0 to 2π

    def __post_init__(self):
        if not 1 <= self.harmonic <= 7:
            raise ValueError(f"Harmonic must be 1-7, got {self.harmonic}")
        if not 0 <= self.phase <= 2 * math.pi:
            self.phase = self.phase % (2 * math.pi)

    @property
    def frequency(self) -> float:
        """Get the planetary frequency for this intent's harmonic."""
        return HARMONIC_FREQUENCY_MAP[self.harmonic]

    @property
    def scale(self) -> float:
        """
        Compute harmonic scaling: scale = R^(h²)

        This is the core H(d, R) formula from AETHERMOORE.
        """
        return harmonic_scaling(self.harmonic, PERFECT_FIFTH)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "primary": self.primary.value,
            "modifier": self.modifier.value,
            "harmonic": self.harmonic,
            "phase": self.phase,
            "frequency": self.frequency,
            "scale": self.scale
        }


# Spiralverse Vocabulary: Maps intent combinations to chaos basin parameters
SPIRALVERSE_VOCABULARY: Dict[Tuple[str, str], dict] = {
    # Read intents
    ("read", "immediate"): {
        "basin_depth": 0.2,
        "attractor_radius": 0.1,
        "chaos_bias": 0.0
    },
    ("read", "conditional"): {
        "basin_depth": 0.3,
        "attractor_radius": 0.15,
        "chaos_bias": 0.1
    },
    ("read", "audited"): {
        "basin_depth": 0.25,
        "attractor_radius": 0.12,
        "chaos_bias": 0.05
    },

    # Write intents
    ("write", "immediate"): {
        "basin_depth": 0.4,
        "attractor_radius": 0.2,
        "chaos_bias": 0.2
    },
    ("write", "deferred"): {
        "basin_depth": 0.35,
        "attractor_radius": 0.18,
        "chaos_bias": 0.15
    },
    ("write", "bounded"): {
        "basin_depth": 0.45,
        "attractor_radius": 0.22,
        "chaos_bias": 0.25
    },

    # Execute intents
    ("execute", "immediate"): {
        "basin_depth": 0.5,
        "attractor_radius": 0.25,
        "chaos_bias": 0.3
    },
    ("execute", "conditional"): {
        "basin_depth": 0.55,
        "attractor_radius": 0.28,
        "chaos_bias": 0.35
    },
    ("execute", "recursive"): {
        "basin_depth": 0.6,
        "attractor_radius": 0.3,
        "chaos_bias": 0.4
    },

    # Delegate intents
    ("delegate", "bounded"): {
        "basin_depth": 0.65,
        "attractor_radius": 0.32,
        "chaos_bias": 0.45
    },
    ("delegate", "audited"): {
        "basin_depth": 0.7,
        "attractor_radius": 0.35,
        "chaos_bias": 0.5
    },

    # Revoke intents
    ("revoke", "immediate"): {
        "basin_depth": 0.8,
        "attractor_radius": 0.4,
        "chaos_bias": 0.6
    },
    ("revoke", "recursive"): {
        "basin_depth": 0.85,
        "attractor_radius": 0.42,
        "chaos_bias": 0.65
    },

    # Query intents
    ("query", "immediate"): {
        "basin_depth": 0.15,
        "attractor_radius": 0.08,
        "chaos_bias": -0.1
    },
    ("query", "bounded"): {
        "basin_depth": 0.2,
        "attractor_radius": 0.1,
        "chaos_bias": -0.05
    },

    # Transform intents
    ("transform", "bounded"): {
        "basin_depth": 0.55,
        "attractor_radius": 0.27,
        "chaos_bias": 0.32
    },
    ("transform", "audited"): {
        "basin_depth": 0.6,
        "attractor_radius": 0.3,
        "chaos_bias": 0.38
    },

    # Verify intents
    ("verify", "immediate"): {
        "basin_depth": 0.1,
        "attractor_radius": 0.05,
        "chaos_bias": -0.2
    },
    ("verify", "silent"): {
        "basin_depth": 0.12,
        "attractor_radius": 0.06,
        "chaos_bias": -0.15
    },
}


def intent_to_basin(intent: Intent) -> dict:
    """
    Map an intent to its chaos basin parameters.

    The basin parameters determine:
    - basin_depth: How "deep" the attractor well is
    - attractor_radius: Size of the valid region
    - chaos_bias: Offset applied to chaos sequence

    These are then scaled by the harmonic: scale = R^(h²)

    Args:
        intent: The intent to map

    Returns:
        Basin parameters with harmonic scaling applied

    Reference: Section 3.2
    Claim: 18
    """
    key = (intent.primary.value, intent.modifier.value)

    if key not in SPIRALVERSE_VOCABULARY:
        # Default basin for unknown combinations
        base = {
            "basin_depth": 0.5,
            "attractor_radius": 0.25,
            "chaos_bias": 0.0
        }
    else:
        base = SPIRALVERSE_VOCABULARY[key].copy()

    # Apply harmonic scaling
    scale = intent.scale

    return {
        "basin_depth": base["basin_depth"] * scale,
        "attractor_radius": base["attractor_radius"] / math.sqrt(scale),
        "chaos_bias": base["chaos_bias"],
        "harmonic": intent.harmonic,
        "frequency": intent.frequency,
        "scale": scale,
        "phase": intent.phase
    }


def frequency_to_chaos_offset(frequency: float, phase: float) -> float:
    """
    Convert planetary frequency and phase to chaos parameter offset.

    Args:
        frequency: Planetary frequency in Hz
        phase: Current phase in radians

    Returns:
        Offset to apply to chaos parameters
    """
    # Normalize frequency to [0, 1] range using Mars as reference
    mars_freq = D_MAJOR_7TH_CHORD["root"]
    normalized = (frequency - mars_freq) / mars_freq

    # Apply phase modulation
    modulated = normalized * math.cos(phase)

    return modulated


def planetary_seed_r(harmonic: int, phase: float = 0.0) -> float:
    """
    Derive chaos r parameter from planetary frequency.

    Args:
        harmonic: Harmonic level (1-7)
        phase: Phase offset

    Returns:
        r value in [3.97, 4.0)
    """
    freq = HARMONIC_FREQUENCY_MAP[harmonic]
    offset = frequency_to_chaos_offset(freq, phase)

    # Map to valid r range
    r_base = 3.985  # Center of valid range
    r_range = 0.01  # ±0.01 variation

    return r_base + offset * r_range


def planetary_seed_x0(harmonic: int, phase: float = 0.0) -> float:
    """
    Derive initial chaos value from planetary frequency.

    Args:
        harmonic: Harmonic level (1-7)
        phase: Phase offset

    Returns:
        x0 value in (0, 1)
    """
    freq = HARMONIC_FREQUENCY_MAP[harmonic]

    # Use frequency digits as seed
    digits = int(freq * 1000) % 1000
    base = digits / 1000

    # Apply phase modulation
    modulated = base + 0.1 * math.sin(phase)

    # Clamp to valid range
    return max(0.01, min(0.99, modulated))


def encode_intent(intent: Intent, key: bytes) -> bytes:
    """
    Encode an intent with a key for transmission.

    Args:
        intent: Intent to encode
        key: Encryption key

    Returns:
        Encoded intent bytes
    """
    # Serialize intent
    data = f"{intent.primary.value}:{intent.modifier.value}:{intent.harmonic}:{intent.phase}"

    # Hash with key
    combined = data.encode() + key
    return hashlib.sha256(combined).digest()


def validate_intent_chain(intents: List[Intent]) -> Tuple[bool, Optional[str]]:
    """
    Validate a chain of intents for consistency.

    Rules:
    - Revoke must follow delegate or write
    - Recursive requires bounded parent
    - Phase must progress monotonically

    Args:
        intents: List of intents in sequence

    Returns:
        Tuple of (valid, error_message)
    """
    if not intents:
        return (True, None)

    prev_phase = -1.0
    prev_primary = None

    for i, intent in enumerate(intents):
        # Phase must progress
        if intent.phase < prev_phase:
            return (False, f"Phase regression at intent {i}")

        # Revoke must follow delegate or write
        if intent.primary == PrimaryIntent.REVOKE:
            if prev_primary not in (PrimaryIntent.DELEGATE, PrimaryIntent.WRITE):
                return (False, f"Revoke without preceding delegate/write at {i}")

        # Recursive requires previous bounded
        if intent.modifier == IntentModifier.RECURSIVE:
            if i > 0 and intents[i-1].modifier != IntentModifier.BOUNDED:
                return (False, f"Recursive without bounded parent at {i}")

        prev_phase = intent.phase
        prev_primary = intent.primary

    return (True, None)


def chord_intent(
    primary: PrimaryIntent,
    modifier: IntentModifier,
    phase: float = 0.0
) -> List[Intent]:
    """
    Create a full D Major 7th chord intent (all 4 planetary frequencies).

    This is the maximum harmonic alignment (h=7 equivalent).

    Args:
        primary: Primary intent type
        modifier: Intent modifier
        phase: Base phase

    Returns:
        List of 4 intents for root, third, fifth, seventh
    """
    chord_harmonics = [1, 2, 3, 4]  # Mars, Jupiter, Venus, Earth
    phase_offsets = [0, math.pi/6, math.pi/3, math.pi/2]

    return [
        Intent(primary, modifier, h, phase + offset)
        for h, offset in zip(chord_harmonics, phase_offsets)
    ]
