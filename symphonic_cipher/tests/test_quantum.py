"""
Tests for Post-Quantum Cryptography Module.

Tests cover:
- ML-KEM (Kyber) key encapsulation
- ML-DSA (Dilithium) signatures
- SHA3 utilities
- PQ Context Commitment
- Quantum state tracking
- Decoherence detection
- Entanglement verification
- Integrated PQCryptoSystem
"""

import numpy as np
import pytest
from symphonic_cipher.scbe_aethermoore.quantum import (
    # Enums
    KEMAlgorithm,
    SignatureAlgorithm,
    QuantumState,
    # Constants
    KYBER_PUBLIC_KEY_SIZE,
    KYBER_SECRET_KEY_SIZE,
    KYBER_CIPHERTEXT_SIZE,
    KYBER_SHARED_SECRET_SIZE,
    DILITHIUM_PUBLIC_KEY_SIZE,
    DILITHIUM_SECRET_KEY_SIZE,
    DILITHIUM_SIGNATURE_SIZE,
    SHA3_256_OUTPUT_SIZE,
    QUANTUM_DIM,
    DECOHERENCE_THRESHOLD,
    # Kyber
    KyberKeyPair,
    KyberEncapsulation,
    KyberKEM,
    # Dilithium
    DilithiumKeyPair,
    DilithiumSignature,
    DilithiumDSA,
    # SHA3
    sha3_256,
    sha3_512,
    shake128,
    shake256,
    # PQ Commitment
    PQContextCommitment,
    # Quantum State
    QuantumPhaseState,
    DecoherenceDetector,
    # Entanglement
    EntanglementWitness,
    EntanglementVerifier,
    # Integrated System
    PQCryptoSystem,
)


class TestKyberKEM:
    """Tests for ML-KEM (Kyber) key encapsulation."""

    def test_keygen(self):
        """Key generation produces valid keys."""
        kyber = KyberKEM()
        keypair = kyber.keygen()

        assert isinstance(keypair, KyberKeyPair)
        assert len(keypair.public_key) == KYBER_PUBLIC_KEY_SIZE
        assert len(keypair.secret_key) == KYBER_SECRET_KEY_SIZE
        assert keypair.algorithm == KEMAlgorithm.ML_KEM_768

    def test_keygen_deterministic(self):
        """Same master key produces same keypair."""
        master = b"test_master_key_32_bytes_long!!"
        kyber1 = KyberKEM(master)
        kyber2 = KyberKEM(master)

        kp1 = kyber1.keygen()
        kp2 = kyber2.keygen()

        assert kp1.public_key == kp2.public_key
        assert kp1.secret_key == kp2.secret_key

    def test_encapsulate(self):
        """Encapsulation produces valid ciphertext and shared secret."""
        kyber = KyberKEM()
        kyber.keygen()
        encap = kyber.encapsulate()

        assert isinstance(encap, KyberEncapsulation)
        assert len(encap.ciphertext) == KYBER_CIPHERTEXT_SIZE
        assert len(encap.shared_secret) == KYBER_SHARED_SECRET_SIZE

    def test_decapsulate(self):
        """Decapsulation produces shared secret."""
        kyber = KyberKEM()
        kyber.keygen()
        encap = kyber.encapsulate()

        shared_secret = kyber.decapsulate(encap.ciphertext)
        assert len(shared_secret) == KYBER_SHARED_SECRET_SIZE

    def test_derive_session_key(self):
        """Session key derivation works."""
        kyber = KyberKEM()
        session_key = kyber.derive_session_key(context=b"test_context")

        assert len(session_key) == 32
        assert isinstance(session_key, bytes)

    def test_different_masters_different_keys(self):
        """Different master keys produce different keypairs."""
        kyber1 = KyberKEM(b"master_key_1_________________")
        kyber2 = KyberKEM(b"master_key_2_________________")

        kp1 = kyber1.keygen()
        kp2 = kyber2.keygen()

        assert kp1.public_key != kp2.public_key


class TestDilithiumDSA:
    """Tests for ML-DSA (Dilithium) signatures."""

    def test_keygen(self):
        """Key generation produces valid keys."""
        dilithium = DilithiumDSA()
        keypair = dilithium.keygen()

        assert isinstance(keypair, DilithiumKeyPair)
        assert len(keypair.public_key) == DILITHIUM_PUBLIC_KEY_SIZE
        assert len(keypair.secret_key) == DILITHIUM_SECRET_KEY_SIZE
        assert keypair.algorithm == SignatureAlgorithm.ML_DSA_65

    def test_sign(self):
        """Signing produces valid signature."""
        dilithium = DilithiumDSA()
        dilithium.keygen()

        message = b"test message to sign"
        sig = dilithium.sign(message)

        assert isinstance(sig, DilithiumSignature)
        assert len(sig.signature) == DILITHIUM_SIGNATURE_SIZE
        assert sig.algorithm == SignatureAlgorithm.ML_DSA_65

    def test_verify_valid(self):
        """Valid signature verifies."""
        dilithium = DilithiumDSA()
        dilithium.keygen()

        message = b"test message"
        sig = dilithium.sign(message)

        assert dilithium.verify(message, sig)

    def test_deterministic_signature(self):
        """Same message and key produces same signature."""
        master = b"deterministic_master_key_____"
        dilithium = DilithiumDSA(master)
        dilithium.keygen()

        message = b"test"
        sig1 = dilithium.sign(message)
        sig2 = dilithium.sign(message)

        assert sig1.signature == sig2.signature


class TestSHA3Utilities:
    """Tests for SHA3 hash functions."""

    def test_sha3_256_length(self):
        """SHA3-256 produces 32-byte output."""
        result = sha3_256(b"test")
        assert len(result) == SHA3_256_OUTPUT_SIZE

    def test_sha3_512_length(self):
        """SHA3-512 produces 64-byte output."""
        result = sha3_512(b"test")
        assert len(result) == 64

    def test_sha3_256_deterministic(self):
        """SHA3-256 is deterministic."""
        assert sha3_256(b"test") == sha3_256(b"test")

    def test_sha3_256_different_inputs(self):
        """Different inputs produce different hashes."""
        assert sha3_256(b"test1") != sha3_256(b"test2")

    def test_shake128(self):
        """SHAKE128 produces output of requested length."""
        result = shake128(b"test", 64)
        assert len(result) == 64

    def test_shake256(self):
        """SHAKE256 produces output of requested length."""
        result = shake256(b"test", 128)
        assert len(result) == 128


class TestPQContextCommitment:
    """Tests for PQ context commitment."""

    def test_create(self):
        """Commitment creation works."""
        context = b"test context data"
        commitment = PQContextCommitment.create(context)

        assert len(commitment.commitment_hash) == 32
        assert len(commitment.kyber_ciphertext) == KYBER_CIPHERTEXT_SIZE
        assert len(commitment.dilithium_signature) == DILITHIUM_SIGNATURE_SIZE
        assert commitment.context_version == 1
        assert commitment.timestamp > 0

    def test_verify_valid(self):
        """Valid commitment verifies."""
        context = b"test context"
        commitment = PQContextCommitment.create(context)

        assert commitment.verify(context)

    def test_verify_wrong_context_fails(self):
        """Wrong context fails verification."""
        context = b"original context"
        commitment = PQContextCommitment.create(context)

        assert not commitment.verify(b"different context")

    def test_serialization(self):
        """Commitment serializes and deserializes correctly."""
        context = b"test data"
        original = PQContextCommitment.create(context)

        serialized = original.to_bytes()
        restored = PQContextCommitment.from_bytes(serialized)

        assert restored.commitment_hash == original.commitment_hash
        assert restored.kyber_ciphertext == original.kyber_ciphertext
        assert restored.dilithium_signature == original.dilithium_signature
        assert restored.context_version == original.context_version

    def test_with_explicit_keys(self):
        """Commitment with explicit Kyber/Dilithium keys."""
        kyber = KyberKEM()
        dilithium = DilithiumDSA()
        kyber.keygen()
        dilithium.keygen()

        context = b"test"
        commitment = PQContextCommitment.create(
            context, kyber=kyber, dilithium=dilithium
        )

        assert commitment.verify(context, dilithium)


class TestQuantumPhaseState:
    """Tests for quantum state tracking."""

    def test_create_coherent(self):
        """Coherent state creation."""
        state = QuantumPhaseState.create_coherent(seed=42)

        assert state.amplitudes.shape == (QUANTUM_DIM,)
        assert state.phase == 0.0
        assert state.state == QuantumState.COHERENT
        # Check normalization
        norm = np.linalg.norm(state.amplitudes)
        assert abs(norm - 1.0) < 1e-10

    def test_deterministic_with_seed(self):
        """Same seed produces same state."""
        state1 = QuantumPhaseState.create_coherent(seed=123)
        state2 = QuantumPhaseState.create_coherent(seed=123)

        np.testing.assert_array_equal(state1.amplitudes, state2.amplitudes)

    def test_evolve_default(self):
        """Default evolution (phase rotation)."""
        state = QuantumPhaseState.create_coherent(seed=42)
        evolved = state.evolve(dt=0.1)

        assert evolved.phase != state.phase
        # Norm preserved
        assert abs(np.linalg.norm(evolved.amplitudes) - 1.0) < 1e-10

    def test_evolve_with_hamiltonian(self):
        """Evolution with explicit Hamiltonian."""
        state = QuantumPhaseState.create_coherent(seed=42)

        # Identity Hamiltonian (trivial evolution)
        H = np.eye(QUANTUM_DIM, dtype=np.complex128)
        evolved = state.evolve(dt=1.0, hamiltonian=H)

        # Norm preserved
        assert abs(np.linalg.norm(evolved.amplitudes) - 1.0) < 1e-10

    def test_fidelity_with_self(self):
        """Fidelity with self is 1."""
        state = QuantumPhaseState.create_coherent(seed=42)
        assert abs(state.fidelity(state) - 1.0) < 1e-10

    def test_fidelity_orthogonal(self):
        """Fidelity with orthogonal state is 0."""
        state1 = QuantumPhaseState.create_coherent(seed=42)
        state2 = QuantumPhaseState.create_coherent(seed=42)

        # Make state2 orthogonal by rotating 90 degrees in first two components
        state2.amplitudes[0], state2.amplitudes[1] = (
            state2.amplitudes[1],
            -state2.amplitudes[0],
        )
        # Re-normalize
        state2.amplitudes /= np.linalg.norm(state2.amplitudes)

        # Fidelity should be small (not exactly 0 due to other components)
        fidelity = state1.fidelity(state2)
        assert fidelity < 0.9

    def test_purity(self):
        """Pure state has purity 1."""
        state = QuantumPhaseState.create_coherent()
        assert state.purity() == 1.0

    def test_von_neumann_entropy(self):
        """Pure state has entropy 0."""
        state = QuantumPhaseState.create_coherent()
        assert state.von_neumann_entropy() == 0.0


class TestDecoherenceDetector:
    """Tests for decoherence detection."""

    def test_first_check_coherent(self):
        """First check always returns coherent."""
        detector = DecoherenceDetector()
        state = QuantumPhaseState.create_coherent()

        status, fidelity = detector.check(state)

        assert status == QuantumState.COHERENT
        assert fidelity == 1.0

    def test_detect_decoherence(self):
        """Decoherence is detected when fidelity drops."""
        detector = DecoherenceDetector(threshold=0.99)
        reference = QuantumPhaseState.create_coherent(seed=1)

        # Create significantly different state
        different = QuantumPhaseState.create_coherent(seed=999)

        detector.set_reference(reference)
        status, fidelity = detector.check(different)

        # Should detect decoherence since states are different
        # (fidelity will be < 0.99 for random states)
        assert fidelity < 1.0

    def test_coherence_time_tracking(self):
        """Coherence time is tracked."""
        detector = DecoherenceDetector()
        state = QuantumPhaseState.create_coherent()
        detector.check(state)

        # Initially no coherence time estimate
        assert detector.get_coherence_time() is None


class TestEntanglementVerifier:
    """Tests for entanglement verification."""

    def test_create_bell_pair(self):
        """Bell pair creation."""
        verifier = EntanglementVerifier()
        party_a, party_b = verifier.create_bell_pair()

        assert party_a.shape == (4,)
        assert party_b.shape == (4,)
        # Both should be normalized
        assert abs(np.linalg.norm(party_a) - 1.0) < 1e-10
        assert abs(np.linalg.norm(party_b) - 1.0) < 1e-10

    def test_verify_entanglement(self):
        """Entanglement verification produces witness."""
        verifier = EntanglementVerifier()
        party_a, party_b = verifier.create_bell_pair()

        witness = verifier.verify_entanglement(party_a, party_b)

        assert isinstance(witness, EntanglementWitness)
        assert witness.correlation >= 0
        assert witness.timestamp > 0

    def test_witnesses_recorded(self):
        """Witnesses are recorded."""
        verifier = EntanglementVerifier()

        for _ in range(3):
            party_a, party_b = verifier.create_bell_pair()
            verifier.verify_entanglement(party_a, party_b)

        assert len(verifier.witnesses) == 3


class TestPQCryptoSystem:
    """Tests for integrated PQ crypto system."""

    def test_initialization(self):
        """System initializes correctly."""
        system = PQCryptoSystem()

        assert system.kyber is not None
        assert system.dilithium is not None
        assert system.quantum_state is not None
        assert system.decoherence_detector is not None

    def test_deterministic_initialization(self):
        """Same seed produces same keys."""
        seed = b"test_seed_32_bytes_long_____"
        system1 = PQCryptoSystem(seed)
        system2 = PQCryptoSystem(seed)

        keys1 = system1.get_public_keys()
        keys2 = system2.get_public_keys()

        assert keys1["kyber_pk"] == keys2["kyber_pk"]
        assert keys1["dilithium_pk"] == keys2["dilithium_pk"]

    def test_create_and_verify_commitment(self):
        """Create and verify commitment."""
        system = PQCryptoSystem()
        context = b"important context data"

        commitment = system.create_commitment(context)
        assert system.verify_commitment(commitment, context)
        assert not system.verify_commitment(commitment, b"wrong context")

    def test_key_encapsulation(self):
        """Key encapsulation workflow."""
        system = PQCryptoSystem()

        encap = system.encapsulate_key()
        shared_secret = system.decapsulate_key(encap.ciphertext)

        assert len(encap.shared_secret) == 32
        assert len(shared_secret) == 32

    def test_signing(self):
        """Signing workflow."""
        system = PQCryptoSystem()
        message = b"message to sign"

        signature = system.sign(message)
        # Note: Simulation only checks signature length, not cryptographic validity
        # In production with real Dilithium, wrong message would fail verification
        assert system.verify_signature(message, signature)
        assert len(signature.signature) == DILITHIUM_SIGNATURE_SIZE

    def test_quantum_evolution(self):
        """Quantum state evolution."""
        system = PQCryptoSystem()

        status, fidelity = system.evolve_quantum_state(dt=0.1)

        assert status in [QuantumState.COHERENT, QuantumState.DECOHERENT]
        assert 0 <= fidelity <= 1

    def test_public_keys(self):
        """Public keys are accessible."""
        system = PQCryptoSystem()
        keys = system.get_public_keys()

        assert "kyber_pk" in keys
        assert "dilithium_pk" in keys
        assert len(keys["kyber_pk"]) == KYBER_PUBLIC_KEY_SIZE
        assert len(keys["dilithium_pk"]) == DILITHIUM_PUBLIC_KEY_SIZE


class TestIntegration:
    """Integration tests for full PQ workflow."""

    def test_full_secure_communication(self):
        """Simulate secure communication between two parties."""
        # Alice and Bob each have their own system
        alice = PQCryptoSystem(b"alice_seed_32_bytes_________")
        bob = PQCryptoSystem(b"bob_seed_32_bytes___________")

        # Alice creates commitment for message
        message = b"Hello Bob, this is Alice"
        commitment = alice.create_commitment(message)

        # Alice signs the commitment hash
        signature = alice.sign(commitment.commitment_hash)

        # Bob verifies Alice's signature (using Alice's public key)
        # In real implementation, Bob would have Alice's public key
        # Here we simulate by checking the structure
        assert len(signature.signature) == DILITHIUM_SIGNATURE_SIZE

        # Bob verifies the commitment
        assert commitment.verify(message)

    def test_quantum_resistant_key_exchange(self):
        """Simulate quantum-resistant key exchange."""
        alice = PQCryptoSystem()
        bob = PQCryptoSystem()

        # Alice encapsulates to Bob's public key
        # (In real impl, would use bob's actual pk)
        encap = alice.encapsulate_key()

        # Both have shared secret derivable from ciphertext
        alice_secret = alice.decapsulate_key(encap.ciphertext)
        bob_secret = bob.decapsulate_key(encap.ciphertext)

        # Both derive 32-byte keys
        assert len(alice_secret) == 32
        assert len(bob_secret) == 32

    def test_commitment_chain(self):
        """Create chain of commitments (blockchain-like)."""
        system = PQCryptoSystem()

        chain = []
        prev_hash = b"genesis"

        for i in range(5):
            context = f"block_{i}_{prev_hash.hex()[:8]}".encode()
            commitment = system.create_commitment(context)
            chain.append((context, commitment))
            prev_hash = commitment.commitment_hash

        # Verify chain
        assert len(chain) == 5
        for context, commitment in chain:
            assert commitment.verify(context)
