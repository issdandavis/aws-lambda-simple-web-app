"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AXIOM 4.1: HARMONIC SPACE CORE                            ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  The Harmonic Scaling Law: H(d, R) = R^(d²)                                  ║
║                                                                              ║
║  This axiom defines the fundamental mathematical foundation of AETHERMOORE:  ║
║  - Harmonic ratio constants (R₅, R₄, R₃, φ)                                  ║
║  - Super-exponential scaling H(d, R) = R^(d²)                                ║
║  - 6D harmonic vector space V₆                                               ║
║  - Harmonic distance metric g_H = diag(1, 1, 1, R, R², R³)                   ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Version: 1.0.0                                                              ║
║  Author: Isaac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from constants module for backward compatibility
from ..constants import (
    # Mathematical Constants
    PI, E, PHI, SQRT2, SQRT5,

    # Harmonic Ratios
    R_FIFTH, R_FOURTH, R_THIRD, R_SIXTH, R_OCTAVE, R_PHI,

    # AETHERMOORE Constants
    PHI_AETHER, LAMBDA_ISAAC, OMEGA_SPIRAL, ALPHA_ABH,

    # Physical Constants
    C_LIGHT, PLANCK_LENGTH, PLANCK_TIME, PLANCK_CONSTANT,

    # Defaults
    DEFAULT_R, DEFAULT_D_MAX, DEFAULT_L, DEFAULT_TOLERANCE, DEFAULT_BASE_BITS,

    # Functions
    harmonic_scale, security_bits, security_level, harmonic_distance, octave_transpose,
    get_harmonic_scale_table,

    # Data Structures
    AethermooreDimension, DIMENSIONS, CONSTANTS, HARMONIC_SCALE_TABLE,
)

__all__ = [
    # Constants
    'PI', 'E', 'PHI', 'SQRT2', 'SQRT5',
    'R_FIFTH', 'R_FOURTH', 'R_THIRD', 'R_SIXTH', 'R_OCTAVE', 'R_PHI',
    'PHI_AETHER', 'LAMBDA_ISAAC', 'OMEGA_SPIRAL', 'ALPHA_ABH',
    'C_LIGHT', 'PLANCK_LENGTH', 'PLANCK_TIME', 'PLANCK_CONSTANT',
    'DEFAULT_R', 'DEFAULT_D_MAX', 'DEFAULT_L', 'DEFAULT_TOLERANCE', 'DEFAULT_BASE_BITS',
    # Functions
    'harmonic_scale', 'security_bits', 'security_level', 'harmonic_distance', 'octave_transpose',
    'get_harmonic_scale_table',
    # Data
    'AethermooreDimension', 'DIMENSIONS', 'CONSTANTS', 'HARMONIC_SCALE_TABLE',
]

AXIOM_ID = "4.1"
AXIOM_TITLE = "Harmonic Space Core"
AXIOM_FORMULA = "H(d, R) = R^(d²)"
