"""
PQC (Post-Quantum Cryptography) Module for AETHERMOORE

Provides post-quantum cryptographic primitives integrated with
AETHERMOORE harmonic scaling for enhanced security.

Components:
- pqc_core: Kyber768 and Dilithium3 primitives
- pqc_harmonic: Harmonic-enhanced PQC operations
"""

from .pqc_core import (
    Kyber768,
    KyberKeyPair,
    EncapsulationResult,
    Dilithium3,
    DilithiumKeyPair,
    derive_hybrid_key,
)

from .pqc_harmonic import (
    SecurityDimension,
    HarmonicKeyMaterial,
    harmonic_key_stretch,
    fast_harmonic_key,
    HarmonicPQCSession,
    create_harmonic_pqc_session,
    verify_harmonic_pqc_session,
    Vector6DKey,
    derive_key_from_vector,
    vector_proximity_key,
    analyze_harmonic_security,
    print_security_table,
    HarmonicKyberOrchestrator,
)

__all__ = [
    # Core PQC
    "Kyber768",
    "KyberKeyPair",
    "EncapsulationResult",
    "Dilithium3",
    "DilithiumKeyPair",
    "derive_hybrid_key",
    # Harmonic PQC
    "SecurityDimension",
    "HarmonicKeyMaterial",
    "harmonic_key_stretch",
    "fast_harmonic_key",
    "HarmonicPQCSession",
    "create_harmonic_pqc_session",
    "verify_harmonic_pqc_session",
    "Vector6DKey",
    "derive_key_from_vector",
    "vector_proximity_key",
    "analyze_harmonic_security",
    "print_security_table",
    "HarmonicKyberOrchestrator",
]
