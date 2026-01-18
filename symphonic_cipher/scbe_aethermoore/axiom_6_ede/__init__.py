"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  AXIOM 6: ENTROPIC DEFENSE ENGINE (EDE)                      ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Mars-Ready Zero-Latency Cryptographic Defense System                        ║
║                                                                              ║
║  Components:                                                                 ║
║  - SpiralRing: Zero-latency stream cipher using Golden Angle rotation        ║
║  - Chemistry Agent: Biological defense simulation with harmonic scaling      ║
║  - Mars Protocol: 3-22 minute latency tolerance with precomputed keys        ║
║                                                                              ║
║  SpiralRing Transform:                                                       ║
║    θ = 2π / φ² (Golden Angle ≈ 137.5°)                                       ║
║    rotate(block, round) = block ⊕ spiral_key(round)                          ║
║                                                                              ║
║  Chemistry Defense:                                                          ║
║    Threat response via harmonic scaling: H(d) = R^(d²)                       ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 6 (Entropic Defense Engine)                                        ║
║  Author: Isaac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from ede modules
from ..ede.spiral_ring import (
    # Configuration
    RingConfig,
    RingState,
    SpiralPosition,
    # Core Classes
    SpiralRing,
    SynchronizedRingPair,
    # Constants
    RING_SIZE,
    SPIRAL_PHI,
    SPIRAL_R,
    SPIRAL_TWIST,
    TIME_QUANTUM,
    EXPANSION_RATE,
    MAX_EXPANSION,
    # Mars Constants
    LIGHT_SPEED,
    MARS_DISTANCE_MIN,
    MARS_DISTANCE_MAX,
    MARS_LIGHT_TIME_MIN,
    MARS_LIGHT_TIME_MAX,
    # Functions
    create_entropy_stream,
    calculate_light_delay,
    mars_light_delay,
)

from ..ede.chemistry_agent import (
    # Classes
    ChemistryAgent,
    WaveSimulation,
    Unit,
    AgentState,
    ThreatType,
    # Constants
    HEALING_RATE,
    ENERGY_DECAY,
    REFRACTION_BASE,
    ANTIBODY_EFFICIENCY_BASE,
    ANTIBODY_EFFICIENCY_BOOST,
    ANTIBODY_RESPONSE_DELAY,
    MALICIOUS_SPAWN_RATE,
    MALICIOUS_SQUARED_FACTOR,
    THREAT_LEVEL_MIN,
    THREAT_LEVEL_MAX,
    DEFAULT_THREAT_LEVEL,
    # Functions
    reaction_rate,
    squared_energy,
    equilibrium_force,
    harmonic_sink,
    ray_refraction,
    self_heal,
    quick_defense_check,
    run_threat_simulation,
)

from ..ede.ede_protocol import (
    # Classes
    EDEStation,
    MarsLink,
    EDEHeader,
    EDEMessage,
    MessageType,
    # Constants
    PROTOCOL_VERSION,
    HEADER_SIZE,
    TIMESTAMP_SIZE,
    SEQUENCE_SIZE,
    MAC_SIZE,
    ERROR_DETECTION_OVERHEAD,
    PQC_AVAILABLE,
    # Functions
    lorentz_factor,
    apply_time_dilation,
    add_error_detection,
    verify_error_detection,
    quick_mars_encode,
    quick_mars_decode,
)

__all__ = [
    # SpiralRing
    'RingConfig', 'RingState', 'SpiralPosition',
    'SpiralRing', 'SynchronizedRingPair',
    'RING_SIZE', 'SPIRAL_PHI', 'SPIRAL_R', 'SPIRAL_TWIST',
    'TIME_QUANTUM', 'EXPANSION_RATE', 'MAX_EXPANSION',
    'LIGHT_SPEED', 'MARS_DISTANCE_MIN', 'MARS_DISTANCE_MAX',
    'MARS_LIGHT_TIME_MIN', 'MARS_LIGHT_TIME_MAX',
    'create_entropy_stream', 'calculate_light_delay', 'mars_light_delay',
    # Chemistry Agent
    'ChemistryAgent', 'WaveSimulation', 'Unit', 'AgentState', 'ThreatType',
    'HEALING_RATE', 'ENERGY_DECAY', 'REFRACTION_BASE',
    'ANTIBODY_EFFICIENCY_BASE', 'ANTIBODY_EFFICIENCY_BOOST', 'ANTIBODY_RESPONSE_DELAY',
    'MALICIOUS_SPAWN_RATE', 'MALICIOUS_SQUARED_FACTOR',
    'THREAT_LEVEL_MIN', 'THREAT_LEVEL_MAX', 'DEFAULT_THREAT_LEVEL',
    'reaction_rate', 'squared_energy', 'equilibrium_force', 'harmonic_sink',
    'ray_refraction', 'self_heal', 'quick_defense_check', 'run_threat_simulation',
    # EDE Protocol
    'EDEStation', 'MarsLink', 'EDEHeader', 'EDEMessage', 'MessageType',
    'PROTOCOL_VERSION', 'HEADER_SIZE', 'TIMESTAMP_SIZE', 'SEQUENCE_SIZE',
    'MAC_SIZE', 'ERROR_DETECTION_OVERHEAD', 'PQC_AVAILABLE',
    'lorentz_factor', 'apply_time_dilation',
    'add_error_detection', 'verify_error_detection',
    'quick_mars_encode', 'quick_mars_decode',
]

AXIOM_ID = "6"
AXIOM_TITLE = "Entropic Defense Engine (EDE)"
AXIOM_FORMULA = "θ_golden = 2π / φ² ≈ 137.5°"
