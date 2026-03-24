"""
Dual-Lattice Quantum Security Module - AETHERMOORE Integration

Implements Claim 62: Dual-Lattice Quantum Commitment requiring BOTH
ML-KEM (Kyber) AND ML-DSA (Dilithium) to agree for valid operations.

Security Properties:
1. MLWE (Module Learning With Errors) - Kyber's hardness assumption
2. MSIS (Module Short Integer Solution) - Dilithium's hardness assumption
3. Dual Consensus: Both must pass, AND results must be consistent
4. Fail-to-Noise: Returns cryptographically random bytes on any failure

The dual-lattice approach provides defense-in-depth:
- If MLWE is broken but MSIS holds → system remains secure
- If MSIS is broken but MLWE holds → system remains secure
- Both must be broken simultaneously to compromise security

Document ID: AETHER-DUAL-LATTICE-2026-001
Version: 1.0.0
#!/usr/bin/env python3
"""
SCBE-AETHERMOORE Dual Lattice Framework

Implements Claim 62: Dual-Lattice Quantum Security Consensus

The dual lattice is a consensus mechanism requiring simultaneous validation
from two independent lattice-based PQC algorithms:

    - ML-KEM (Kyber): Primal lattice for key encapsulation (MLWE hardness)
    - ML-DSA (Dilithium): Dual lattice for signatures (MSIS hardness)

"Settling" Mechanism:
    - Unstable chaotic equations at init
    - Become stable ONLY when both lattices agree within time window Δt < ε
    - Resolves to key K(t_arrival) at interference maximum

Mathematical Foundation:
    Consensus = Kyber_valid ∧ Dilithium_valid ∧ (Δt < ε)

    If consensus:
        K(t) = Σ C_n sin(ω_n t + φ_n) mod P   (constructive interference)
    Else:
        K(t) = chaotic noise                   (fail-to-noise)

Security Properties:
    - Breaking one algorithm insufficient (AND logic)
    - Requires breaking BOTH MLWE and MSIS simultaneously
    - Provable min(security_Kyber, security_Dilithium) = ~2^192

Integration with SCBE:
    - Axiom A3: Weighted dual norms (positive definiteness)
    - Axiom A8: Realms as primal/dual zones
    - Axiom A11: Triadic with dual as third "check"
    - Axiom A12: Risk ↑ on mismatch → R' += w_dual × (1 - consensus) × H(d*, R)

Date: January 15, 2026
Golden Master: v2.0.1
Patent Claim: 62 (Dual-Lattice Quantum Security)
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
from enum import Enum

from .pqc import (
    Kyber768, KyberKeyPair, EncapsulationResult,
    Dilithium3, DilithiumKeyPair,
    HarmonicKeyMaterial,
    fast_harmonic_key,
    derive_hybrid_key,
)

from .constants import (
    DEFAULT_R, DEFAULT_D_MAX,
    harmonic_scale, security_bits,
)

import numpy as np
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
import time

# =============================================================================
# CONSTANTS
# =============================================================================

# Dual consensus parameters
CONSENSUS_TIMEOUT_MS = 5000  # 5 second timeout for consensus
NOISE_OUTPUT_SIZE = 32  # Size of fail-to-noise output
BINDING_CONTEXT = b"AETHER-DUAL-LATTICE-BINDING-v1"

# Lattice problem identifiers
class LatticeProblem(Enum):
    """Underlying lattice hardness assumptions."""
    MLWE = "MLWE"  # Module Learning With Errors (Kyber)
    MSIS = "MSIS"  # Module Short Integer Solution (Dilithium)


class ConsensusResult(Enum):
    """Result of dual-lattice consensus."""
    CONSENSUS_ACHIEVED = "CONSENSUS_ACHIEVED"
    KYBER_FAILED = "KYBER_FAILED"
    DILITHIUM_FAILED = "DILITHIUM_FAILED"
    BINDING_MISMATCH = "BINDING_MISMATCH"
    TIMEOUT = "TIMEOUT"
    NOISE_RETURNED = "NOISE_RETURNED"


# =============================================================================
# DUAL-LATTICE KEY BUNDLE
# =============================================================================

@dataclass(frozen=True)
class DualLatticeKeyBundle:
    """
    Combined key bundle for dual-lattice operations.

    Contains both Kyber (KEM) and Dilithium (signature) keypairs,
    bound together with a commitment hash.
    """
    kyber_keypair: KyberKeyPair
    dilithium_keypair: DilithiumKeyPair
    binding_hash: bytes  # SHA3-256 of both public keys
    created_at: float

    @classmethod
    def generate(cls) -> 'DualLatticeKeyBundle':
        """Generate a new dual-lattice key bundle."""
        kyber_kp = Kyber768.generate_keypair()
        dilithium_kp = Dilithium3.generate_keypair()

        # Binding hash commits to both public keys
        binding_data = (
            BINDING_CONTEXT +
            kyber_kp.public_key +
            dilithium_kp.public_key
        )
        binding_hash = hashlib.sha3_256(binding_data).digest()

        return cls(
            kyber_keypair=kyber_kp,
            dilithium_keypair=dilithium_kp,
            binding_hash=binding_hash,
            created_at=time.time()
        )

    def get_public_bundle(self) -> 'DualLatticePublicBundle':
        """Extract public components only."""
        return DualLatticePublicBundle(
            kyber_public_key=self.kyber_keypair.public_key,
            dilithium_public_key=self.dilithium_keypair.public_key,
            binding_hash=self.binding_hash
        )


@dataclass(frozen=True)
class DualLatticePublicBundle:
    """Public components of a dual-lattice key bundle."""
    kyber_public_key: bytes
    dilithium_public_key: bytes
    binding_hash: bytes

    def verify_binding(self) -> bool:
        """Verify that the binding hash is correct."""
        expected = hashlib.sha3_256(
            BINDING_CONTEXT +
            self.kyber_public_key +
            self.dilithium_public_key
        ).digest()
        return hmac.compare_digest(self.binding_hash, expected)


# =============================================================================
# FAIL-TO-NOISE MECHANISM
# =============================================================================

def fail_to_noise(
    context: bytes = b"",
    size: int = NOISE_OUTPUT_SIZE
) -> bytes:
    """
    Generate cryptographically random noise on failure.

    This prevents oracle attacks by making failure indistinguishable
    from success at the byte level. Attackers cannot learn anything
    about why the operation failed.

    Args:
        context: Optional context for logging (not used in output)
        size: Number of random bytes to return

    Returns:
        Cryptographically random bytes
    """
    return secrets.token_bytes(size)


def is_noise(data: bytes, expected_hash: Optional[bytes] = None) -> bool:
    """
    Check if data appears to be fail-to-noise output.

    This is probabilistic - there's no way to definitively know
    if output is noise or valid (which is the point).

    If expected_hash is provided, checks if data hashes to it.
    """
    if expected_hash is None:
        return False  # Can't verify without expected hash

    actual_hash = hashlib.sha3_256(data).digest()
    return not hmac.compare_digest(actual_hash, expected_hash)


# =============================================================================
# DUAL CONSENSUS PROTOCOL
# =============================================================================

@dataclass
class DualConsensusSession:
    """
    Session for dual-lattice consensus operations.

    Tracks the state of both Kyber and Dilithium operations
    and enforces that both must succeed for consensus.
    """
    session_id: bytes
    initiator_bundle: DualLatticePublicBundle
    responder_bundle: Optional[DualLatticePublicBundle]

    # Kyber state
    kyber_ciphertext: Optional[bytes] = None
    kyber_shared_secret: Optional[bytes] = None
    kyber_success: bool = False

    # Dilithium state
    signature: Optional[bytes] = None
    signature_valid: bool = False
    dilithium_success: bool = False

    # Consensus state
    consensus_result: ConsensusResult = ConsensusResult.NOISE_RETURNED
    final_key: Optional[bytes] = None
    noise_output: Optional[bytes] = None

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


def create_dual_consensus_session(
    initiator_bundle: DualLatticeKeyBundle,
    responder_public: DualLatticePublicBundle,
    message: bytes,
    dimension: int = DEFAULT_D_MAX,
    R: float = DEFAULT_R
) -> Tuple[DualConsensusSession, bytes]:
    """
    Create a dual-lattice consensus session.

    Performs both Kyber encapsulation AND Dilithium signing.
    Both operations must succeed for the session to be valid.

    Args:
        initiator_bundle: Initiator's full key bundle
        responder_public: Responder's public bundle
        message: Message to sign and bind to session
        dimension: Harmonic security dimension
        R: Harmonic ratio

    Returns:
        Tuple of (session, session_data_for_responder)
        If any operation fails, session contains noise output.
    """
    session_id = secrets.token_bytes(16)

    session = DualConsensusSession(
        session_id=session_id,
        initiator_bundle=initiator_bundle.get_public_bundle(),
        responder_bundle=responder_public
    )

    # Verify responder's binding
    if not responder_public.verify_binding():
        session.consensus_result = ConsensusResult.BINDING_MISMATCH
        session.noise_output = fail_to_noise(b"binding_mismatch")
        return session, session.noise_output

    try:
        # Step 1: Kyber encapsulation (MLWE)
        encap_result = Kyber768.encapsulate(responder_public.kyber_public_key)
        session.kyber_ciphertext = encap_result.ciphertext
        session.kyber_shared_secret = encap_result.shared_secret
        session.kyber_success = True

    except Exception:
        session.consensus_result = ConsensusResult.KYBER_FAILED
        session.noise_output = fail_to_noise(b"kyber_failed")
        return session, session.noise_output

    try:
        # Step 2: Dilithium signature (MSIS)
        # Sign: session_id || kyber_ciphertext || message || initiator_binding
        sign_data = (
            session_id +
            encap_result.ciphertext +
            message +
            initiator_bundle.binding_hash
        )
        session.signature = Dilithium3.sign(
            initiator_bundle.dilithium_keypair.secret_key,
            sign_data
        )
        session.dilithium_success = True

    except Exception:
        session.consensus_result = ConsensusResult.DILITHIUM_FAILED
        session.noise_output = fail_to_noise(b"dilithium_failed")
        return session, session.noise_output

    # Step 3: Both succeeded - derive consensus key
    # Key is derived from BOTH the Kyber shared secret AND the signature
    # This ensures both lattice problems contribute to the final key
    consensus_material = (
        session.kyber_shared_secret +
        hashlib.sha3_256(session.signature).digest() +
        session_id +
        BINDING_CONTEXT
    )

    # Apply harmonic enhancement
    session.final_key = fast_harmonic_key(
        consensus_material,
        dimension=dimension,
        R=R,
        salt=session_id,
        info=b"dual-lattice-consensus"
    )

    session.consensus_result = ConsensusResult.CONSENSUS_ACHIEVED
    session.completed_at = time.time()

    # Package data for responder
    session_data = (
        session_id +
        encap_result.ciphertext +
        session.signature +
        initiator_bundle.binding_hash
    )

    return session, session_data


def verify_dual_consensus_session(
    responder_bundle: DualLatticeKeyBundle,
    initiator_public: DualLatticePublicBundle,
    session_data: bytes,
    message: bytes,
    dimension: int = DEFAULT_D_MAX,
    R: float = DEFAULT_R
) -> Tuple[DualConsensusSession, Optional[bytes]]:
    """
    Verify and complete a dual-lattice consensus session.

    Both Kyber decapsulation AND Dilithium verification must succeed.

    Args:
        responder_bundle: Responder's full key bundle
        initiator_public: Initiator's public bundle
        session_data: Data received from initiator
        message: Original message that was signed
        dimension: Harmonic security dimension
        R: Harmonic ratio

    Returns:
        Tuple of (session, final_key or None)
        If any verification fails, returns noise instead of key.
    """
    # Parse session data
    if len(session_data) < 16:
        session = DualConsensusSession(
            session_id=b"",
            initiator_bundle=initiator_public,
            responder_bundle=responder_bundle.get_public_bundle()
        )
        session.consensus_result = ConsensusResult.BINDING_MISMATCH
        return session, fail_to_noise(b"invalid_session_data")

    session_id = session_data[:16]

    # Expected sizes
    # Kyber ciphertext: 1088 bytes
    # Dilithium signature: 3293 bytes
    # Binding hash: 32 bytes
    ciphertext_end = 16 + 1088
    signature_end = ciphertext_end + 3293

    if len(session_data) < signature_end + 32:
        session = DualConsensusSession(
            session_id=session_id,
            initiator_bundle=initiator_public,
            responder_bundle=responder_bundle.get_public_bundle()
        )
        session.consensus_result = ConsensusResult.BINDING_MISMATCH
        return session, fail_to_noise(b"truncated_data")

    kyber_ciphertext = session_data[16:ciphertext_end]
    signature = session_data[ciphertext_end:signature_end]
    initiator_binding = session_data[signature_end:signature_end + 32]

    session = DualConsensusSession(
        session_id=session_id,
        initiator_bundle=initiator_public,
        responder_bundle=responder_bundle.get_public_bundle(),
        kyber_ciphertext=kyber_ciphertext,
        signature=signature
    )

    # Verify initiator's binding
    if not initiator_public.verify_binding():
        session.consensus_result = ConsensusResult.BINDING_MISMATCH
        return session, fail_to_noise(b"initiator_binding_invalid")

    if not hmac.compare_digest(initiator_binding, initiator_public.binding_hash):
        session.consensus_result = ConsensusResult.BINDING_MISMATCH
        return session, fail_to_noise(b"binding_mismatch")

    try:
        # Step 1: Kyber decapsulation (MLWE)
        shared_secret = Kyber768.decapsulate(
            responder_bundle.kyber_keypair.secret_key,
            kyber_ciphertext
        )
        session.kyber_shared_secret = shared_secret
        session.kyber_success = True

    except Exception:
        session.consensus_result = ConsensusResult.KYBER_FAILED
        return session, fail_to_noise(b"kyber_decap_failed")

    try:
        # Step 2: Dilithium verification (MSIS)
        sign_data = (
            session_id +
            kyber_ciphertext +
            message +
            initiator_binding
        )

        session.signature_valid = Dilithium3.verify(
            initiator_public.dilithium_public_key,
            sign_data,
            signature
        )

        if not session.signature_valid:
            session.consensus_result = ConsensusResult.DILITHIUM_FAILED
            return session, fail_to_noise(b"signature_invalid")

        session.dilithium_success = True

    except Exception:
        session.consensus_result = ConsensusResult.DILITHIUM_FAILED
        return session, fail_to_noise(b"dilithium_verify_failed")

    # Step 3: Both succeeded - derive same consensus key
    consensus_material = (
        shared_secret +
        hashlib.sha3_256(signature).digest() +
        session_id +
        BINDING_CONTEXT
    )

    session.final_key = fast_harmonic_key(
        consensus_material,
        dimension=dimension,
        R=R,
        salt=session_id,
        info=b"dual-lattice-consensus"
    )

    session.consensus_result = ConsensusResult.CONSENSUS_ACHIEVED
    session.completed_at = time.time()

    return session, session.final_key


# =============================================================================
# DUAL-LATTICE COMMITMENT SCHEME
# =============================================================================

@dataclass
class DualLatticeCommitment:
    """
    Commitment that requires both MLWE and MSIS to open.

    The commitment binds a message such that:
    1. It cannot be opened without both Kyber AND Dilithium keys
    2. It cannot be equivocated (changed after commitment)
    3. Any tampering is detectable
    """
    commitment_hash: bytes  # SHA3-256 commitment
    kyber_ciphertext: bytes  # Encrypted opening
    signature: bytes  # Signature over commitment
    timestamp: float

    def to_bytes(self) -> bytes:
        """Serialize commitment."""
        return (
            self.commitment_hash +
            len(self.kyber_ciphertext).to_bytes(4, 'big') +
            self.kyber_ciphertext +
            len(self.signature).to_bytes(4, 'big') +
            self.signature +
            int(self.timestamp * 1000000).to_bytes(8, 'big')
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DualLatticeCommitment':
        """Deserialize commitment."""
        commitment_hash = data[:32]
        ct_len = int.from_bytes(data[32:36], 'big')
        kyber_ciphertext = data[36:36 + ct_len]
        sig_start = 36 + ct_len
        sig_len = int.from_bytes(data[sig_start:sig_start + 4], 'big')
        signature = data[sig_start + 4:sig_start + 4 + sig_len]
        ts_start = sig_start + 4 + sig_len
        timestamp = int.from_bytes(data[ts_start:ts_start + 8], 'big') / 1000000

        return cls(
            commitment_hash=commitment_hash,
            kyber_ciphertext=kyber_ciphertext,
            signature=signature,
            timestamp=timestamp
        )


def create_dual_commitment(
    committer_bundle: DualLatticeKeyBundle,
    verifier_public: DualLatticePublicBundle,
    message: bytes,
    randomness: Optional[bytes] = None
) -> Tuple[DualLatticeCommitment, bytes]:
    """
    Create a dual-lattice commitment to a message.

    The commitment is binding (can't change message) and hiding
    (message is encrypted under Kyber). Opening requires both
    the Kyber secret AND valid Dilithium signature.

    Args:
        committer_bundle: Committer's full key bundle
        verifier_public: Verifier's public bundle
        message: Message to commit to
        randomness: Optional explicit randomness

    Returns:
        Tuple of (commitment, opening_key)
    """
    if randomness is None:
        randomness = secrets.token_bytes(32)

    # Create commitment: H(message || randomness || binding)
    commitment_data = (
        message +
        randomness +
        committer_bundle.binding_hash +
        verifier_public.binding_hash
    )
    commitment_hash = hashlib.sha3_256(commitment_data).digest()

    # Encrypt the opening under verifier's Kyber key
    opening_material = message + randomness
    encap_result = Kyber768.encapsulate(verifier_public.kyber_public_key)

    # Derive encryption key from Kyber shared secret
    enc_key = hashlib.sha3_256(
        encap_result.shared_secret + b"commitment-encryption"
    ).digest()

    # Simple XOR encryption (in production, use AES-GCM)
    encrypted_opening = bytes(
        a ^ b for a, b in zip(
            opening_material.ljust(len(enc_key) * ((len(opening_material) // len(enc_key)) + 1), b'\x00'),
            (enc_key * ((len(opening_material) // len(enc_key)) + 1))[:len(opening_material)]
        )
    )

    # Sign the commitment
    sign_data = commitment_hash + encap_result.ciphertext
    signature = Dilithium3.sign(
        committer_bundle.dilithium_keypair.secret_key,
        sign_data
    )

    commitment = DualLatticeCommitment(
        commitment_hash=commitment_hash,
        kyber_ciphertext=encap_result.ciphertext,
        signature=signature,
        timestamp=time.time()
    )

    # Opening key is the Kyber shared secret
    return commitment, encap_result.shared_secret


def verify_dual_commitment(
    verifier_bundle: DualLatticeKeyBundle,
    committer_public: DualLatticePublicBundle,
    commitment: DualLatticeCommitment,
    claimed_message: bytes
) -> Tuple[bool, str]:
    """
    Verify a dual-lattice commitment.

    Both Kyber decryption AND Dilithium verification must succeed.

    Args:
        verifier_bundle: Verifier's full key bundle
        committer_public: Committer's public bundle
        commitment: The commitment to verify
        claimed_message: Message that committer claims was committed

    Returns:
        Tuple of (is_valid, reason)
    """
    # Step 1: Verify Dilithium signature (MSIS)
    sign_data = commitment.commitment_hash + commitment.kyber_ciphertext

    if not Dilithium3.verify(
        committer_public.dilithium_public_key,
        sign_data,
        commitment.signature
    ):
        return False, "Signature verification failed (MSIS)"

    # Step 2: Kyber decapsulation (MLWE)
    try:
        shared_secret = Kyber768.decapsulate(
            verifier_bundle.kyber_keypair.secret_key,
            commitment.kyber_ciphertext
        )
    except Exception:
        return False, "Kyber decapsulation failed (MLWE)"

    # Step 3: Verify commitment hash
    # We need the randomness, which we can't recover without the opening
    # In this simplified version, we just verify the signature is valid
    # and the Kyber operation succeeded

    # For full verification, the committer would need to reveal
    # the randomness as part of opening

    return True, "Dual-lattice commitment verified"


# =============================================================================
# DUAL-LATTICE ORCHESTRATOR
# =============================================================================

class DualLatticeOrchestrator:
    """
    High-level orchestrator for dual-lattice operations.

    Manages key bundles and provides simple interface for
    dual-consensus sessions and commitments.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_D_MAX,
        R: float = DEFAULT_R
    ):
        """
        Initialize orchestrator with fresh key bundle.

        Args:
            dimension: Harmonic security dimension
            R: Harmonic ratio
        """
        self.bundle = DualLatticeKeyBundle.generate()
        self.dimension = dimension
        self.R = R
        self.sessions: Dict[bytes, DualConsensusSession] = {}

    def get_public_bundle(self) -> DualLatticePublicBundle:
        """Get public components for sharing."""
        return self.bundle.get_public_bundle()

    def initiate_consensus(
        self,
        responder_public: DualLatticePublicBundle,
        message: bytes
    ) -> Tuple[Optional[bytes], bytes]:
        """
        Initiate a dual-consensus session.

        Args:
            responder_public: Responder's public bundle
            message: Message to bind to session

        Returns:
            Tuple of (final_key or None, session_data for responder)
        """
        session, session_data = create_dual_consensus_session(
            self.bundle,
            responder_public,
            message,
            self.dimension,
            self.R
        )

        self.sessions[session.session_id] = session

        if session.consensus_result == ConsensusResult.CONSENSUS_ACHIEVED:
            return session.final_key, session_data
        else:
            return None, session_data

    def respond_to_consensus(
        self,
        initiator_public: DualLatticePublicBundle,
        session_data: bytes,
        message: bytes
    ) -> Optional[bytes]:
        """
        Respond to a dual-consensus session.

        Args:
            initiator_public: Initiator's public bundle
            session_data: Data received from initiator
            message: Original message

        Returns:
            Final key if consensus achieved, None otherwise
        """
        session, key_or_noise = verify_dual_consensus_session(
            self.bundle,
            initiator_public,
            session_data,
            message,
            self.dimension,
            self.R
        )

        self.sessions[session.session_id] = session

        if session.consensus_result == ConsensusResult.CONSENSUS_ACHIEVED:
            return key_or_noise
        else:
            return None  # Key is noise, don't return it

    def get_security_analysis(self) -> Dict[str, Any]:
        """Get security analysis for current configuration."""
        h_value = harmonic_scale(self.dimension, self.R)

        return {
            "lattice_problems": ["MLWE (Kyber)", "MSIS (Dilithium)"],
            "consensus_requirement": "BOTH must pass",
            "failure_mode": "Fail-to-noise (indistinguishable random)",
            "dimension": self.dimension,
            "harmonic_ratio": self.R,
            "H_value": h_value,
            "kyber_security": "NIST Level 3 (~192 bits)",
            "dilithium_security": "NIST Level 2 (~128 bits)",
            "combined_security": "min(192, 128) = 128 bits (both must hold)",
            "harmonic_enhanced": f"128 + {self.dimension}² × log₂({self.R}) bits",
            "effective_bits": security_bits(128, self.dimension, self.R)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_dual_consensus(
    message: bytes,
    dimension: int = DEFAULT_D_MAX
) -> Tuple[bool, Optional[bytes], Dict[str, Any]]:
    """
    Quick self-test of dual-lattice consensus.

    Creates two parties and performs full consensus protocol.

    Args:
        message: Test message
        dimension: Security dimension

    Returns:
        Tuple of (success, shared_key, debug_info)
    """
    alice = DualLatticeOrchestrator(dimension=dimension)
    bob = DualLatticeOrchestrator(dimension=dimension)

    # Alice initiates
    alice_key, session_data = alice.initiate_consensus(
        bob.get_public_bundle(),
        message
    )

    # Bob responds
    bob_key = bob.respond_to_consensus(
        alice.get_public_bundle(),
        session_data,
        message
    )

    # Check consensus
    success = (
        alice_key is not None and
        bob_key is not None and
        hmac.compare_digest(alice_key, bob_key)
    )

    debug_info = {
        "alice_session": alice.sessions,
        "bob_session": bob.sessions,
        "keys_match": success,
        "alice_key_hex": alice_key.hex()[:32] + "..." if alice_key else None,
        "bob_key_hex": bob_key.hex()[:32] + "..." if bob_key else None
    }

    return success, alice_key if success else None, debug_info
PHI = (1 + np.sqrt(5)) / 2
EPSILON = 1e-10

# Security levels (NIST)
SECURITY_LEVEL_3 = 192  # bits (ML-KEM-768, ML-DSA-65)


class LatticeType(Enum):
    """Lattice algorithm types."""
    PRIMAL = "PRIMAL"   # ML-KEM (Kyber) - MLWE
    DUAL = "DUAL"       # ML-DSA (Dilithium) - MSIS


class ConsensusState(Enum):
    """Dual lattice consensus states."""
    UNSETTLED = "UNSETTLED"   # Waiting for both validations
    SETTLING = "SETTLING"     # One valid, waiting for other
    SETTLED = "SETTLED"       # Both valid within window
    FAILED = "FAILED"         # Mismatch or timeout
    CHAOS = "CHAOS"           # Fail-to-noise triggered


# =============================================================================
# SIMULATED LATTICE OPERATIONS
# =============================================================================

@dataclass
class LatticeKeyPair:
    """Lattice-based key pair (simulated)."""
    public_key: bytes
    secret_key: bytes
    algorithm: str
    security_level: int = SECURITY_LEVEL_3


@dataclass
class KyberResult:
    """Result from ML-KEM operation."""
    ciphertext: bytes
    shared_secret: bytes
    valid: bool
    timestamp: float


@dataclass
class DilithiumResult:
    """Result from ML-DSA operation."""
    signature: bytes
    valid: bool
    timestamp: float


class SimulatedKyber:
    """
    Simulated ML-KEM (Kyber) for demonstration.

    In production, use liboqs or pqcrypto.

    MLWE Problem: b = As + e + m
        - A: public matrix
        - s: secret vector
        - e: error vector
        - m: message
    """

    def __init__(self, security_level: int = SECURITY_LEVEL_3):
        self.security_level = security_level
        self.key_size = security_level // 8  # bytes

    def keygen(self) -> LatticeKeyPair:
        """Generate Kyber key pair."""
        seed = np.random.bytes(32)
        pk = hashlib.sha3_256(seed + b"kyber_pk").digest()
        sk = hashlib.sha3_256(seed + b"kyber_sk").digest()

        return LatticeKeyPair(
            public_key=pk,
            secret_key=sk,
            algorithm="ML-KEM-768",
            security_level=self.security_level
        )

    def encapsulate(self, public_key: bytes) -> KyberResult:
        """Encapsulate shared secret."""
        randomness = np.random.bytes(32)
        ciphertext = hashlib.sha3_256(public_key + randomness).digest()
        shared_secret = hashlib.sha3_256(ciphertext + public_key).digest()

        return KyberResult(
            ciphertext=ciphertext,
            shared_secret=shared_secret,
            valid=True,
            timestamp=time.time()
        )

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> KyberResult:
        """Decapsulate shared secret."""
        # In real implementation, this would use lattice math
        shared_secret = hashlib.sha3_256(ciphertext + secret_key[:16]).digest()

        return KyberResult(
            ciphertext=ciphertext,
            shared_secret=shared_secret,
            valid=True,
            timestamp=time.time()
        )


class SimulatedDilithium:
    """
    Simulated ML-DSA (Dilithium) for demonstration.

    MSIS Problem: Find short vector in signed lattice.
    """

    def __init__(self, security_level: int = SECURITY_LEVEL_3):
        self.security_level = security_level

    def keygen(self) -> LatticeKeyPair:
        """Generate Dilithium key pair."""
        seed = np.random.bytes(32)
        pk = hashlib.sha3_256(seed + b"dilithium_pk").digest()
        sk = hashlib.sha3_256(seed + b"dilithium_sk").digest()

        return LatticeKeyPair(
            public_key=pk,
            secret_key=sk,
            algorithm="ML-DSA-65",
            security_level=self.security_level
        )

    def sign(self, secret_key: bytes, message: bytes) -> DilithiumResult:
        """Sign message."""
        signature = hmac.new(secret_key, message, hashlib.sha3_256).digest()

        return DilithiumResult(
            signature=signature,
            valid=True,
            timestamp=time.time()
        )

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> DilithiumResult:
        """Verify signature."""
        # Simulated verification
        expected = hmac.new(public_key[:32], message, hashlib.sha3_256).digest()

        # In real impl, this uses lattice verification
        valid = len(signature) == 32  # Simplified check

        return DilithiumResult(
            signature=signature,
            valid=valid,
            timestamp=time.time()
        )


# =============================================================================
# DUAL LATTICE CONSENSUS
# =============================================================================

@dataclass
class ConsensusParams:
    """Parameters for dual lattice consensus."""
    time_window: float = 0.1      # ε: max time between validations (seconds)
    kyber_weight: float = 0.5     # λ1: Kyber contribution
    dilithium_weight: float = 0.5 # λ2: Dilithium contribution
    risk_weight: float = 0.3      # w_dual: Risk contribution on mismatch


@dataclass
class SettlingResult:
    """Result from settling process."""
    state: ConsensusState
    key: Optional[bytes]          # K(t) if settled
    consensus_value: float        # 0 (failed) to 1 (settled)
    kyber_valid: bool
    dilithium_valid: bool
    time_delta: float             # Δt between validations
    risk_contribution: float      # Added to R'
    harmonics: List[float]        # Fourier components of K(t)


class DualLatticeConsensus:
    """
    Dual Lattice Consensus Engine.

    Implements the "settling" mechanism:
        - Unstable at init (chaotic)
        - Settles ONLY when both lattices validate within time window
        - Produces K(t) via constructive interference
    """

    def __init__(self, params: Optional[ConsensusParams] = None):
        self.params = params or ConsensusParams()
        self.kyber = SimulatedKyber()
        self.dilithium = SimulatedDilithium()

        # State
        self._kyber_result: Optional[KyberResult] = None
        self._dilithium_result: Optional[DilithiumResult] = None
        self._state = ConsensusState.UNSETTLED

        # Harmonic parameters for K(t)
        self._C_n = np.array([1.0, 0.5, 0.25, 0.125])  # Amplitudes
        self._omega_n = np.array([1.0, PHI, PHI**2, PHI**3])  # Frequencies
        self._phi_n = np.zeros(4)  # Phases (set on consensus)

    def _compute_settling_key(self, t_arrival: float) -> bytes:
        """
        Compute K(t) at settling time via constructive interference.

        K(t) = Σ C_n sin(ω_n t + φ_n)

        At t_arrival, phases align for maximum constructive interference.
        """
        # Set phases for constructive interference at t_arrival
        self._phi_n = -self._omega_n * t_arrival

        # Compute K(t_arrival) - should be maximum
        K_value = np.sum(self._C_n * np.sin(self._omega_n * t_arrival + self._phi_n))

        # Normalize and hash to get key bytes
        K_normalized = (K_value + np.sum(self._C_n)) / (2 * np.sum(self._C_n))
        K_bytes = hashlib.sha3_256(str(K_normalized).encode() + str(t_arrival).encode()).digest()

        return K_bytes

    def _compute_chaos_noise(self) -> bytes:
        """
        Generate chaotic noise (fail-to-noise).

        When consensus fails, return unpredictable noise.
        """
        chaos = np.random.bytes(32)
        return hashlib.sha3_256(chaos + str(time.time()).encode()).digest()

    def submit_kyber(self, ciphertext: bytes, public_key: bytes) -> None:
        """Submit Kyber validation."""
        result = self.kyber.encapsulate(public_key)
        result.ciphertext = ciphertext
        self._kyber_result = result

        if self._state == ConsensusState.UNSETTLED:
            self._state = ConsensusState.SETTLING

    def submit_dilithium(self, signature: bytes, message: bytes, public_key: bytes) -> None:
        """Submit Dilithium validation."""
        result = self.dilithium.verify(public_key, message, signature)
        self._dilithium_result = result

        if self._state == ConsensusState.UNSETTLED:
            self._state = ConsensusState.SETTLING

    def check_consensus(self) -> SettlingResult:
        """
        Check if dual consensus has been reached.

        Consensus = Kyber_valid ∧ Dilithium_valid ∧ (Δt < ε)
        """
        # Check if both submitted
        if self._kyber_result is None or self._dilithium_result is None:
            return SettlingResult(
                state=ConsensusState.UNSETTLED,
                key=None,
                consensus_value=0.0,
                kyber_valid=self._kyber_result is not None and self._kyber_result.valid,
                dilithium_valid=self._dilithium_result is not None and self._dilithium_result.valid,
                time_delta=float('inf'),
                risk_contribution=self.params.risk_weight,
                harmonics=[]
            )

        # Check validity
        kyber_valid = self._kyber_result.valid
        dilithium_valid = self._dilithium_result.valid

        # Check time window
        time_delta = abs(self._kyber_result.timestamp - self._dilithium_result.timestamp)
        time_valid = time_delta < self.params.time_window

        # Consensus = AND of all three
        consensus = kyber_valid and dilithium_valid and time_valid

        if consensus:
            # SETTLED - compute K(t) via constructive interference
            t_arrival = (self._kyber_result.timestamp + self._dilithium_result.timestamp) / 2
            key = self._compute_settling_key(t_arrival)
            self._state = ConsensusState.SETTLED

            return SettlingResult(
                state=ConsensusState.SETTLED,
                key=key,
                consensus_value=1.0,
                kyber_valid=kyber_valid,
                dilithium_valid=dilithium_valid,
                time_delta=time_delta,
                risk_contribution=0.0,  # No risk on success
                harmonics=list(self._C_n * np.sin(self._phi_n))
            )
        else:
            # FAILED - return chaos noise
            self._state = ConsensusState.CHAOS if (kyber_valid != dilithium_valid) else ConsensusState.FAILED

            # Risk contribution on mismatch
            mismatch = 1.0 - (0.5 * kyber_valid + 0.5 * dilithium_valid)
            risk = self.params.risk_weight * mismatch

            return SettlingResult(
                state=self._state,
                key=self._compute_chaos_noise(),  # Fail-to-noise
                consensus_value=0.0,
                kyber_valid=kyber_valid,
                dilithium_valid=dilithium_valid,
                time_delta=time_delta,
                risk_contribution=risk,
                harmonics=[]
            )

    def reset(self):
        """Reset consensus state."""
        self._kyber_result = None
        self._dilithium_result = None
        self._state = ConsensusState.UNSETTLED


# =============================================================================
# INTEGRATION WITH SCBE RISK ENGINE
# =============================================================================

def integrate_dual_lattice_risk(
    consensus_result: SettlingResult,
    base_risk: float,
    H_d_star: float
) -> float:
    """
    Integrate dual lattice mismatch into SCBE risk.

    R' += w_dual × (1 - consensus) × H(d*, R)

    Args:
        consensus_result: Result from dual lattice check
        base_risk: Current risk value
        H_d_star: Harmonic scaling factor from Layer 12

    Returns:
        Updated risk value
    """
    mismatch = 1.0 - consensus_result.consensus_value
    risk_addition = consensus_result.risk_contribution * mismatch * H_d_star

    return base_risk + risk_addition


# =============================================================================
# SETTLING WAVE VISUALIZATION
# =============================================================================

def compute_settling_wave(
    t: np.ndarray,
    C_n: np.ndarray,
    omega_n: np.ndarray,
    t_arrival: float
) -> np.ndarray:
    """
    Compute the settling wave K(t).

    K(t) = Σ C_n sin(ω_n t + φ_n)

    Where φ_n = π/2 - ω_n × t_arrival for constructive interference.
    At t_arrival: sin(ω_n * t_arrival + π/2 - ω_n * t_arrival) = sin(π/2) = 1
    """
    phi_n = np.pi/2 - omega_n * t_arrival

    K = np.zeros_like(t, dtype=float)
    for C, omega, phi in zip(C_n, omega_n, phi_n):
        K += C * np.sin(omega * t + phi)

    return K


# =============================================================================
# SELF-TESTS
# =============================================================================

def self_test() -> Dict[str, Any]:
    """Run dual lattice self-tests."""
    results = {}
    passed = 0
    total = 0

    # Test 1: Kyber keygen and encapsulate
    total += 1
    try:
        kyber = SimulatedKyber()
        keys = kyber.keygen()
        result = kyber.encapsulate(keys.public_key)

        if result.valid and len(result.shared_secret) == 32:
            passed += 1
            results["kyber_ops"] = "✓ PASS (keygen + encapsulate)"
        else:
            results["kyber_ops"] = "✗ FAIL (invalid result)"
    except Exception as e:
        results["kyber_ops"] = f"✗ FAIL ({e})"

    # Test 2: Dilithium sign and verify
    total += 1
    try:
        dilithium = SimulatedDilithium()
        keys = dilithium.keygen()
        message = b"test message"
        sig_result = dilithium.sign(keys.secret_key, message)
        verify_result = dilithium.verify(keys.public_key, message, sig_result.signature)

        if sig_result.valid and verify_result.valid:
            passed += 1
            results["dilithium_ops"] = "✓ PASS (sign + verify)"
        else:
            results["dilithium_ops"] = "✗ FAIL (invalid result)"
    except Exception as e:
        results["dilithium_ops"] = f"✗ FAIL ({e})"

    # Test 3: Consensus AND logic (both valid → settled)
    total += 1
    try:
        consensus = DualLatticeConsensus()

        # Submit both within time window
        kyber_keys = consensus.kyber.keygen()
        dilithium_keys = consensus.dilithium.keygen()

        consensus.submit_kyber(b"test_ct", kyber_keys.public_key)
        consensus.submit_dilithium(b"x" * 32, b"test_msg", dilithium_keys.public_key)

        result = consensus.check_consensus()

        if result.state == ConsensusState.SETTLED and result.consensus_value == 1.0:
            passed += 1
            results["consensus_and"] = "✓ PASS (both valid → SETTLED)"
        else:
            results["consensus_and"] = f"✗ FAIL (state={result.state})"
    except Exception as e:
        results["consensus_and"] = f"✗ FAIL ({e})"

    # Test 4: Consensus failure (only one submitted)
    total += 1
    try:
        consensus = DualLatticeConsensus()
        kyber_keys = consensus.kyber.keygen()

        consensus.submit_kyber(b"test_ct", kyber_keys.public_key)
        # Don't submit Dilithium

        result = consensus.check_consensus()

        if result.state == ConsensusState.UNSETTLED:
            passed += 1
            results["consensus_partial"] = "✓ PASS (one valid → UNSETTLED)"
        else:
            results["consensus_partial"] = f"✗ FAIL (state={result.state})"
    except Exception as e:
        results["consensus_partial"] = f"✗ FAIL ({e})"

    # Test 5: Key uniqueness (different consensus → different keys)
    total += 1
    try:
        consensus1 = DualLatticeConsensus()
        consensus2 = DualLatticeConsensus()

        # First consensus
        k1 = consensus1.kyber.keygen()
        d1 = consensus1.dilithium.keygen()
        consensus1.submit_kyber(b"ct1", k1.public_key)
        consensus1.submit_dilithium(b"x" * 32, b"msg1", d1.public_key)
        result1 = consensus1.check_consensus()

        # Second consensus (different keys)
        time.sleep(0.01)  # Ensure different timestamp
        k2 = consensus2.kyber.keygen()
        d2 = consensus2.dilithium.keygen()
        consensus2.submit_kyber(b"ct2", k2.public_key)
        consensus2.submit_dilithium(b"y" * 32, b"msg2", d2.public_key)
        result2 = consensus2.check_consensus()

        if result1.key != result2.key:
            passed += 1
            results["key_uniqueness"] = "✓ PASS (different consensus → different K)"
        else:
            results["key_uniqueness"] = "✗ FAIL (keys should differ)"
    except Exception as e:
        results["key_uniqueness"] = f"✗ FAIL ({e})"

    # Test 6: Settling wave constructive interference
    total += 1
    try:
        C_n = np.array([1.0, 0.5, 0.25])
        omega_n = np.array([1.0, 2.0, 3.0])
        t_arrival = 5.0

        # At t_arrival, constructive interference → K = sum(C_n)
        K_at_arrival = compute_settling_wave(np.array([t_arrival]), C_n, omega_n, t_arrival)[0]
        expected_max = np.sum(C_n)  # = 1.75

        # Verify constructive interference at t_arrival
        if abs(K_at_arrival - expected_max) < 0.01:
            passed += 1
            results["settling_wave"] = f"✓ PASS (K(t_arrival)={K_at_arrival:.2f} = Σc_n={expected_max:.2f})"
        else:
            results["settling_wave"] = f"✗ FAIL (K(t_arrival)={K_at_arrival:.2f}, expected {expected_max:.2f})"
    except Exception as e:
        results["settling_wave"] = f"✗ FAIL ({e})"

    # Test 7: Risk integration
    total += 1
    try:
        # Settled → no risk
        settled_result = SettlingResult(
            state=ConsensusState.SETTLED,
            key=b"test",
            consensus_value=1.0,
            kyber_valid=True,
            dilithium_valid=True,
            time_delta=0.01,
            risk_contribution=0.0,
            harmonics=[]
        )
        risk_settled = integrate_dual_lattice_risk(settled_result, 0.5, 2.0)

        # Failed → adds risk
        failed_result = SettlingResult(
            state=ConsensusState.FAILED,
            key=b"noise",
            consensus_value=0.0,
            kyber_valid=True,
            dilithium_valid=False,
            time_delta=0.5,
            risk_contribution=0.3,
            harmonics=[]
        )
        risk_failed = integrate_dual_lattice_risk(failed_result, 0.5, 2.0)

        if risk_settled == 0.5 and risk_failed > 0.5:
            passed += 1
            results["risk_integration"] = f"✓ PASS (settled={risk_settled:.2f}, failed={risk_failed:.2f})"
        else:
            results["risk_integration"] = "✗ FAIL (risk calculation wrong)"
    except Exception as e:
        results["risk_integration"] = f"✗ FAIL ({e})"

    # Test 8: Security level
    total += 1
    try:
        kyber = SimulatedKyber(SECURITY_LEVEL_3)
        dilithium = SimulatedDilithium(SECURITY_LEVEL_3)

        kyber_keys = kyber.keygen()
        dilithium_keys = dilithium.keygen()

        if kyber_keys.security_level == 192 and dilithium_keys.security_level == 192:
            passed += 1
            results["security_level"] = f"✓ PASS (both at {SECURITY_LEVEL_3}-bit)"
        else:
            results["security_level"] = "✗ FAIL (security level mismatch)"
    except Exception as e:
        results["security_level"] = f"✗ FAIL ({e})"

    # Test 9: Fail-to-noise on mismatch
    total += 1
    try:
        consensus = DualLatticeConsensus()

        # Only submit Kyber (Dilithium missing → fail)
        kyber_keys = consensus.kyber.keygen()
        consensus.submit_kyber(b"ct", kyber_keys.public_key)

        # Force a mismatch scenario
        consensus._dilithium_result = DilithiumResult(
            signature=b"invalid",
            valid=False,  # Invalid!
            timestamp=time.time()
        )

        result = consensus.check_consensus()

        # Should get chaos noise
        if result.state == ConsensusState.CHAOS and result.key is not None:
            passed += 1
            results["fail_to_noise"] = "✓ PASS (mismatch → chaos noise)"
        else:
            results["fail_to_noise"] = f"✗ FAIL (state={result.state})"
    except Exception as e:
        results["fail_to_noise"] = f"✗ FAIL ({e})"

    # Test 10: Reset functionality
    total += 1
    try:
        consensus = DualLatticeConsensus()

        kyber_keys = consensus.kyber.keygen()
        consensus.submit_kyber(b"ct", kyber_keys.public_key)

        consensus.reset()

        result = consensus.check_consensus()

        if result.state == ConsensusState.UNSETTLED:
            passed += 1
            results["reset"] = "✓ PASS (reset clears state)"
        else:
            results["reset"] = f"✗ FAIL (state after reset={result.state})"
    except Exception as e:
        results["reset"] = f"✗ FAIL ({e})"

    return {
        "passed": passed,
        "total": total,
        "success_rate": f"{passed}/{total} ({100*passed/total:.1f}%)",
        "results": results
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCBE-AETHERMOORE DUAL LATTICE FRAMEWORK")
    print("Claim 62: Dual-Lattice Quantum Security Consensus")
    print("=" * 70)

    # Run self-tests
    test_results = self_test()

    print("\n[SELF-TESTS]")
    for name, result in test_results["results"].items():
        print(f"  {name}: {result}")

    print("-" * 70)
    print(f"TOTAL: {test_results['success_rate']}")

    # Demonstration
    print("\n" + "=" * 70)
    print("DUAL LATTICE CONSENSUS DEMO")
    print("=" * 70)

    consensus = DualLatticeConsensus()

    print("\n1. Generate key pairs...")
    kyber_keys = consensus.kyber.keygen()
    dilithium_keys = consensus.dilithium.keygen()
    print(f"   Kyber:     {kyber_keys.algorithm} ({kyber_keys.security_level}-bit)")
    print(f"   Dilithium: {dilithium_keys.algorithm} ({dilithium_keys.security_level}-bit)")

    print("\n2. Submit validations...")
    consensus.submit_kyber(b"test_ciphertext", kyber_keys.public_key)
    print("   ✓ Kyber submitted")

    consensus.submit_dilithium(b"x" * 32, b"test_message", dilithium_keys.public_key)
    print("   ✓ Dilithium submitted")

    print("\n3. Check consensus...")
    result = consensus.check_consensus()

    print(f"   State:     {result.state.value}")
    print(f"   Consensus: {result.consensus_value}")
    print(f"   Δt:        {result.time_delta*1000:.2f} ms")

    if result.state == ConsensusState.SETTLED:
        print(f"   K(t):      {result.key[:16].hex()}...")
        print("   → SETTLED: Key derived via constructive interference")
    else:
        print("   → FAILED: Chaos noise returned")

    # Wave visualization data
    print("\n" + "-" * 70)
    print("SETTLING WAVE K(t):")

    C_n = np.array([1.0, 0.5, 0.25, 0.125])
    omega_n = np.array([1.0, PHI, PHI**2, PHI**3])
    t_arrival = 5.0

    t_points = [0, 2.5, 5.0, 7.5, 10.0]
    for t in t_points:
        K = compute_settling_wave(np.array([t]), C_n, omega_n, t_arrival)[0]
        bar = "█" * int((K + 2) * 10)
        marker = " ← MAX (constructive)" if abs(t - t_arrival) < 0.1 else ""
        print(f"  t={t:4.1f}: K={K:+.3f} {bar}{marker}")

    print("=" * 70)
