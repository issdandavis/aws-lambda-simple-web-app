"""
AETHERMOORE Constants

Fundamental constants derived from the Golden Ratio (phi) and Perfect Fifth (R_5).
These constants are used throughout the SCBE-AETHERMOORE cryptographic framework.

Reference: Appendix A of SCBE-AETHER-UNIFIED-2026-001
"""

import math
from typing import Dict, NamedTuple

# Fundamental Mathematical Constants
GOLDEN_RATIO: float = (1 + math.sqrt(5)) / 2  # phi = 1.6180339887...
PERFECT_FIFTH: float = 3 / 2  # R_5 = 1.5 (exact)

# Derived AETHERMOORE Constants
# Phi_aether = phi^(1/R_5) - Chaos attenuation coefficient
PHI_AETHER: float = GOLDEN_RATIO ** (1 / PERFECT_FIFTH)  # 1.3782407720...

# Lambda_isaac = R_5 * phi^2 - Energy threshold
LAMBDA_ISAAC: float = PERFECT_FIFTH * (GOLDEN_RATIO ** 2)  # 3.9270509831...

# Omega_spiral = 2*pi / phi^3 - Entropy export rate
OMEGA_SPIRAL: float = (2 * math.pi) / (GOLDEN_RATIO ** 3)  # 0.9340017595...

# Alpha_abh = phi + R_5 - Acoustic Black Hole coefficient
ALPHA_ABH: float = GOLDEN_RATIO + PERFECT_FIFTH  # 3.1180339887...

# Critical Thresholds
EVENT_HORIZON_THRESHOLD: float = ALPHA_ABH * LAMBDA_ISAAC  # 12.2446...
SOLITON_THRESHOLD: float = PHI_AETHER * (1 - OMEGA_SPIRAL)  # 0.0909...
ENTROPY_EXPORT_RATE: float = 1 - OMEGA_SPIRAL  # 6.6%


class PlanetaryFrequency(NamedTuple):
    """Planetary orbital frequency transposed to audible range."""
    name: str
    period_days: float
    base_frequency_hz: float
    octaves: int
    audible_hz: float
    note: str
    chord_degree: str


# Planetary Harmonic Table (Appendix B)
PLANETARY_FREQUENCIES: Dict[str, PlanetaryFrequency] = {
    "mercury": PlanetaryFrequency(
        name="Mercury",
        period_days=87.97,
        base_frequency_hz=1.316e-7,
        octaves=30,
        audible_hz=141.27,
        note="C#3",
        chord_degree="Major 7th (alt)"
    ),
    "venus": PlanetaryFrequency(
        name="Venus",
        period_days=224.70,
        base_frequency_hz=5.151e-8,
        octaves=32,
        audible_hz=221.23,
        note="A3",
        chord_degree="Perfect 5th"
    ),
    "earth": PlanetaryFrequency(
        name="Earth",
        period_days=365.25,
        base_frequency_hz=3.169e-8,
        octaves=32,
        audible_hz=136.10,
        note="C#3",
        chord_degree="Major 7th"
    ),
    "mars": PlanetaryFrequency(
        name="Mars",
        period_days=687.00,
        base_frequency_hz=1.685e-8,
        octaves=33,
        audible_hz=144.72,
        note="D3",
        chord_degree="Root"
    ),
    "jupiter": PlanetaryFrequency(
        name="Jupiter",
        period_days=4333,
        base_frequency_hz=2.671e-9,
        octaves=36,
        audible_hz=183.58,
        note="F#3",
        chord_degree="Major 3rd"
    ),
    "saturn": PlanetaryFrequency(
        name="Saturn",
        period_days=10759,
        base_frequency_hz=1.076e-9,
        octaves=37,
        audible_hz=147.85,
        note="D3",
        chord_degree="Root (octave)"
    ),
}

# D Major 7th Chord frequencies (Mars, Jupiter, Venus, Earth)
D_MAJOR_7TH_CHORD = {
    "root": PLANETARY_FREQUENCIES["mars"].audible_hz,      # D3 - 144.72 Hz
    "third": PLANETARY_FREQUENCIES["jupiter"].audible_hz,  # F#3 - 183.58 Hz
    "fifth": PLANETARY_FREQUENCIES["venus"].audible_hz,    # A3 - 221.23 Hz
    "seventh": PLANETARY_FREQUENCIES["earth"].audible_hz,  # C#3 - 136.10 Hz
}

# Harmonic mapping by security dimension
HARMONIC_FREQUENCY_MAP = {
    1: PLANETARY_FREQUENCIES["mars"].audible_hz,     # 144.72 Hz
    2: PLANETARY_FREQUENCIES["jupiter"].audible_hz,  # 183.58 Hz
    3: PLANETARY_FREQUENCIES["venus"].audible_hz,    # 221.23 Hz
    4: PLANETARY_FREQUENCIES["earth"].audible_hz,    # 136.10 Hz
    5: PLANETARY_FREQUENCIES["saturn"].audible_hz,   # 147.85 Hz
    6: PLANETARY_FREQUENCIES["mercury"].audible_hz,  # 141.27 Hz
    7: sum(D_MAJOR_7TH_CHORD.values()) / 4,          # Full chord average
}

# Recommended parameter ranges (Appendix C)
PARAMETER_RANGES = {
    "r_chaos": {"min": 3.97, "max": 4.0, "default": 3.99},
    "R_harmonic": {"value": PERFECT_FIFTH},  # Exact 1.5
    "d_dimension": {"min": 1, "max": 7, "default": 6},
    "fractal_iterations": {"min": 30, "max": 100, "default": 50},
    "escape_radius": {"min": 1.5, "max": 3.0, "default": 2.0},
    "energy_threshold_k": {"min": 2, "max": 4, "default": 3},
    "trust_alpha": {"min": 0.8, "max": 0.95, "default": 0.9},
    "tau_participate": {"min": 0.2, "max": 0.4, "default": 0.3},
    "epsilon_coherence": {"min": 0.1, "max": 0.25, "default": 0.15},
    "phase_period_seconds": {"min": 30, "max": 120, "default": 60},
}

# 6D Harmonic Metric Tensor g = diag(1, 1, 1, R_5, R_5^2, R_5^3)
HARMONIC_METRIC_TENSOR = (
    1.0,                    # time / x
    1.0,                    # device_id / y
    1.0,                    # threat_level / z
    PERFECT_FIFTH,          # entropy / velocity: 1.5
    PERFECT_FIFTH ** 2,     # server_load / priority: 2.25
    PERFECT_FIFTH ** 3,     # behavior_stability / security: 3.375
)


def validate_constants() -> Dict[str, bool]:
    """Validate all AETHERMOORE constants against expected values."""
    results = {}

    # Golden ratio validation
    results["golden_ratio"] = abs(GOLDEN_RATIO - 1.6180339887) < 1e-9

    # Perfect fifth is exact
    results["perfect_fifth"] = PERFECT_FIFTH == 1.5

    # Derived constants (10-digit precision)
    results["phi_aether"] = abs(PHI_AETHER - 1.3782407720) < 1e-9
    results["lambda_isaac"] = abs(LAMBDA_ISAAC - 3.9270509831) < 1e-9
    results["omega_spiral"] = abs(OMEGA_SPIRAL - 0.9340017595) < 1e-9
    results["alpha_abh"] = abs(ALPHA_ABH - 3.1180339887) < 1e-9

    # Critical thresholds
    results["event_horizon"] = abs(EVENT_HORIZON_THRESHOLD - 12.2446) < 0.001
    results["soliton_threshold"] = abs(SOLITON_THRESHOLD - 0.0909) < 0.001
    results["entropy_export"] = abs(ENTROPY_EXPORT_RATE - 0.066) < 0.001

    # Planetary ratio validation (Mars → Venus = 1.5 ± 2%)
    mars_venus_ratio = (
        PLANETARY_FREQUENCIES["venus"].audible_hz /
        PLANETARY_FREQUENCIES["mars"].audible_hz
    )
    results["planetary_fifth"] = 1.47 < mars_venus_ratio < 1.53

    return results
