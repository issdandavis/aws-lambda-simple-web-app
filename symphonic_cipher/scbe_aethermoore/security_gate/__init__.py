"""
SCBE Security Gate Module
=========================

Integrated security components from the scbe-security-gate repository:

1. Computational Immune System
   - Magic number finder through physics-style simulation
   - Multi-dimensional expert mind for drift optimization
   - Context vectors (5W1H) for drift calculation

2. DNA Multi-Layer Encoding
   - Morse/DNA base layer encoding
   - Temporal vectors (when)
   - Emotional intent gravity (why/how/for whom)
   - Spatial routing vectors (where)
   - Six Sacred Tongues cipher integration

3. Entropic Dual-Quantum System
   - ForwardSecureRatchet (Signal Double Ratchet style)
   - MarsReceiver (0-RTT with anti-replay)
   - AdaptiveKController (quantum threat response)

Patent Claims:
- Multi-Nodal Semantic Drift Control System
- Syntactic Authentication for AI Communication
- Geometric Routing with Cipher Constraints
- Entropic Escape Velocity Theorem
"""

__version__ = "1.0.0"

# Computational Immune System
from .computational_immune_system import (
    ScienceDomain,
    MultidimensionalExpertMind,
    ContextVector,
    ErrorAccumulator,
    DriftSimulationEngine,
)

# DNA Multi-Layer Encoding
from .dna_multi_layer_encoding import (
    MORSE_TO_DNA,
    DNA_TO_MORSE,
    encode_morse_dna,
    text_to_morse,
    TemporalVector,
    EmotionalIntentVector,
    SpatialVector,
    DNAMultiLayerMessage,
    test_attack_resistance,
    integrate_with_six_sacred_tongues,
)

# Entropic Dual-Quantum System
from .entropic_quantum_system import (
    N0_BITS,
    N0,
    K_DEFAULT,
    C_QUANTUM,
    C_CLASSICAL,
    ForwardSecureRatchet,
    ReplayError,
    MarsReceiver,
    AdaptiveKController,
)

__all__ = [
    # Version
    "__version__",

    # Computational Immune System
    "ScienceDomain",
    "MultidimensionalExpertMind",
    "ContextVector",
    "ErrorAccumulator",
    "DriftSimulationEngine",

    # DNA Multi-Layer Encoding
    "MORSE_TO_DNA",
    "DNA_TO_MORSE",
    "encode_morse_dna",
    "text_to_morse",
    "TemporalVector",
    "EmotionalIntentVector",
    "SpatialVector",
    "DNAMultiLayerMessage",
    "test_attack_resistance",
    "integrate_with_six_sacred_tongues",

    # Entropic Dual-Quantum System
    "N0_BITS",
    "N0",
    "K_DEFAULT",
    "C_QUANTUM",
    "C_CLASSICAL",
    "ForwardSecureRatchet",
    "ReplayError",
    "MarsReceiver",
    "AdaptiveKController",
]
