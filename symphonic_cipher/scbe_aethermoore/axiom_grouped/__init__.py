"""
Axiom-Grouped Module - SCBE-AETHERMOORE

This module provides the axiom-grouped components for the SCBE governance system:

1. Langues Metric (langues_metric.py)
   - 6D phase-shifted exponential cost function
   - Six Sacred Tongues: KO, AV, RU, CA, UM, DR
   - Golden ratio weight progression: wₗ = φˡ
   - Fluxing dimensions (Polly/Quasi/Demi)

2. Audio Axis (audio_axis.py) - Layer 14
   - FFT-based telemetry without metric modification
   - Feature vector: f_audio(t) = [E_a, C_a, F_a, r_HF,a]
   - Stability score: S_audio = 1 - r_HF,a

3. Hamiltonian CFI (hamiltonian_cfi.py)
   - Control Flow Integrity via spectral embedding
   - Golden path = Hamiltonian path through CFG
   - Dirac/Ore theorem verification
   - Dimensional lifting for obstruction resolution

Mathematical Proofs:
- Langues: 7 proofs (monotonicity, phase bounded, golden weights, etc.)
- Audio: 3 proofs (stability bounded, HF detection, flux sensitivity)
- CFI: 3 proofs (Dirac theorem, bipartite detection, deviation detection)

Reference: SCBE Patent Specification, Axioms A1-A12
"""

from .langues_metric import (
    PHI,
    SacredTongue,
    FluxState,
    GovernanceDecision,
    TongueParameters,
    DimensionFlux,
    LanguesMetricResult,
    LanguesMetric,
    FluxingLanguesMetric,
    project_to_1d,
    compute_six_fold_symmetry_error,
    verify_golden_weights,
)

from .audio_axis import (
    EPSILON,
    DEFAULT_SAMPLE_RATE,
    HF_CUTOFF_RATIO,
    AudioStabilityLevel,
    AudioFeatures,
    SpectralState,
    AudioAxisProcessor,
    AudioAxisTelemetry,
    verify_stability_bounded,
    verify_hf_detection,
    verify_flux_sensitivity,
    generate_test_signal,
)

from .hamiltonian_cfi import (
    CFIResult,
    BipartiteStatus,
    CFGVertex,
    ControlFlowGraph,
    HamiltonianPathResult,
    SpectralEmbedding,
    HamiltonianCFI,
    verify_dirac_theorem,
    verify_ore_theorem,
    verify_bipartite_obstruction,
    verify_deviation_detection,
    lift_to_higher_dimension,
    create_complete_graph,
    create_cycle_graph,
)

__all__ = [
    # Constants
    'PHI',
    'EPSILON',
    'DEFAULT_SAMPLE_RATE',
    'HF_CUTOFF_RATIO',

    # Langues Metric
    'SacredTongue',
    'FluxState',
    'GovernanceDecision',
    'TongueParameters',
    'DimensionFlux',
    'LanguesMetricResult',
    'LanguesMetric',
    'FluxingLanguesMetric',
    'project_to_1d',
    'compute_six_fold_symmetry_error',
    'verify_golden_weights',

    # Audio Axis
    'AudioStabilityLevel',
    'AudioFeatures',
    'SpectralState',
    'AudioAxisProcessor',
    'AudioAxisTelemetry',
    'verify_stability_bounded',
    'verify_hf_detection',
    'verify_flux_sensitivity',
    'generate_test_signal',

    # Hamiltonian CFI
    'CFIResult',
    'BipartiteStatus',
    'CFGVertex',
    'ControlFlowGraph',
    'HamiltonianPathResult',
    'SpectralEmbedding',
    'HamiltonianCFI',
    'verify_dirac_theorem',
    'verify_ore_theorem',
    'verify_bipartite_obstruction',
    'verify_deviation_detection',
    'lift_to_higher_dimension',
    'create_complete_graph',
    'create_cycle_graph',
]

__version__ = '1.0.0'
