"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  AXIOM 4.5: VACUUM-ACOUSTICS KERNEL                          ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Wave propagation and nodal patterns in harmonic space                       ║
║                                                                              ║
║  Core Equations:                                                             ║
║  - Nodal Surface: N(x; n, m) = cos(nπx₁/L)cos(mπx₂/L) -                     ║
║                                cos(mπx₁/L)cos(nπx₂/L) = 0                    ║
║  - Chladni Pattern: C(x, y) = sin(nπx/L)sin(mπy/L) ±                        ║
║                               sin(mπx/L)sin(nπy/L)                           ║
║  - Bottle Beam: I(r) = I₀ · |Σₖ exp(i·k·r + φₖ)|²                           ║
║  - Flux Redistribution: E_corner = E_total / 4                               ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 5 (Vacuum-Acoustics Kernel)                                        ║
║  Author: Issac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from vacuum_acoustics module
from ..vacuum_acoustics import (
    # Configuration
    VacuumAcousticsConfig,
    WaveSource,
    FluxResult,
    BottleBeamResult,

    # Core Functions
    nodal_surface,
    is_on_nodal_line,
    compute_chladni_pattern,
    visualize_chladni_pattern,
    find_nodal_points,

    # Resonance
    extract_mode_parameters,
    check_cymatic_resonance,
    resonance_strength,

    # Wave Operations
    bottle_beam_intensity,
    flux_redistribution,
    compute_interference_pattern,
    harmonic_pressure_field,

    # Beam Creation
    create_bottle_beam_sources,
    analyze_bottle_beam,

    # Analysis
    get_vacuum_acoustics_stats,
)

__all__ = [
    'VacuumAcousticsConfig',
    'WaveSource',
    'FluxResult',
    'BottleBeamResult',
    'nodal_surface',
    'is_on_nodal_line',
    'compute_chladni_pattern',
    'visualize_chladni_pattern',
    'find_nodal_points',
    'extract_mode_parameters',
    'check_cymatic_resonance',
    'resonance_strength',
    'bottle_beam_intensity',
    'flux_redistribution',
    'compute_interference_pattern',
    'harmonic_pressure_field',
    'create_bottle_beam_sources',
    'analyze_bottle_beam',
    'get_vacuum_acoustics_stats',
]

AXIOM_ID = "4.5"
AXIOM_TITLE = "Vacuum-Acoustics Kernel"
AXIOM_FORMULA = "N(x; n, m) = cos(nπx₁/L)cos(mπx₂/L) - cos(mπx₁/L)cos(nπx₂/L) = 0"
