"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  AXIOM 5: POST-QUANTUM CRYPTOGRAPHY HARMONIC                 ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  PQC integration with harmonic scaling for quantum-resistant security        ║
║                                                                              ║
║  Security Levels with H(d, R):                                               ║
║  - S_bits(d, R, B) = B + d² × log₂(R)                                        ║
║  - d=6, R=1.5: 128 bits → 149+ effective bits                                ║
║                                                                              ║
║  Components:                                                                 ║
║  - Kyber-768 key encapsulation (NIST PQC)                                    ║
║  - Dilithium-3 signatures                                                    ║
║  - HMAC with harmonic key derivation                                         ║
║  - 6D vector-based session keys                                              ║
║  - Harmonic security analysis                                                ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 5 (PQC Harmonic Integration)                                       ║
║  Author: Issac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from pqc modules
from ..pqc.pqc_harmonic import (
    fast_harmonic_key,
    Vector6DKey,
    derive_key_from_vector,
    vector_proximity_key,
    analyze_harmonic_security,
    HarmonicPQCSession,
)

from ..pqc.pqc_core import (
    Kyber768,
    KyberKeyPair,
    EncapsulationResult,
    Dilithium3,
    DilithiumKeyPair,
    SignatureResult,
    PQCBackend,
    get_backend,
    is_liboqs_available,
    generate_pqc_session_keys,
    verify_pqc_session,
    derive_hybrid_key,
)

from ..pqc.pqc_hmac import (
    # Classes
    PQCHMACChain,
    PQCHMACState,
    PQCKeyMaterial,
    KeyDerivationMode,
    # Functions
    create_pqc_hmac_state,
    pqc_hmac_chain_tag,
    pqc_verify_hmac_chain,
    pqc_derive_hmac_key,
    pqc_recover_hmac_key,
    migrate_classical_chain,
)

from ..kyber_orchestrator import (
    SCBE_AETHERMOORE_Kyber,
    KyberKEM,
    L6_SessionKey,
    HyperbolicAgent,
    run_governance_test,
    quick_validate,
)

__all__ = [
    # Harmonic PQC
    'fast_harmonic_key',
    'Vector6DKey',
    'derive_key_from_vector',
    'vector_proximity_key',
    'analyze_harmonic_security',
    'HarmonicPQCSession',
    # Kyber
    'Kyber768',
    'KyberKeyPair',
    'EncapsulationResult',
    # Dilithium
    'Dilithium3',
    'DilithiumKeyPair',
    'SignatureResult',
    # Backend
    'PQCBackend',
    'get_backend',
    'is_liboqs_available',
    'generate_pqc_session_keys',
    'verify_pqc_session',
    'derive_hybrid_key',
    # PQC HMAC
    'PQCHMACChain',
    'PQCHMACState',
    'PQCKeyMaterial',
    'KeyDerivationMode',
    'create_pqc_hmac_state',
    'pqc_hmac_chain_tag',
    'pqc_verify_hmac_chain',
    'pqc_derive_hmac_key',
    'pqc_recover_hmac_key',
    'migrate_classical_chain',
    # Orchestrator
    'SCBE_AETHERMOORE_Kyber',
    'KyberKEM',
    'L6_SessionKey',
    'HyperbolicAgent',
    'run_governance_test',
    'quick_validate',
]

AXIOM_ID = "5"
AXIOM_TITLE = "Post-Quantum Cryptography Harmonic"
AXIOM_FORMULA = "S_bits(d, R, B) = B + d² × log₂(R)"
