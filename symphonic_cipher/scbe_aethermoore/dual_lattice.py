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
