"""
Tests for PQC (Post-Quantum Cryptography) Module

Tests cover:
- Kyber768 key encapsulation mechanism
- Dilithium3 digital signatures
- Harmonic key derivation
- Harmonic PQC sessions
- 6D vector key operations
- Security analysis utilities
"""

import pytest
import math

from symphonic_cipher.scbe_aethermoore.pqc import (
    # Core PQC
    Kyber768,
    KyberKeyPair,
    EncapsulationResult,
    Dilithium3,
    DilithiumKeyPair,
    derive_hybrid_key,
    # Harmonic PQC
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

from symphonic_cipher.scbe_aethermoore.constants import (
    harmonic_scale,
    security_bits,
    DEFAULT_R,
    DEFAULT_D_MAX,
)

from symphonic_cipher.scbe_aethermoore.pqc.pqc_core import (
    KYBER768_PUBLIC_KEY_SIZE,
    KYBER768_SECRET_KEY_SIZE,
    KYBER768_CIPHERTEXT_SIZE,
    KYBER768_SHARED_SECRET_SIZE,
    DILITHIUM3_PUBLIC_KEY_SIZE,
    DILITHIUM3_SECRET_KEY_SIZE,
    DILITHIUM3_SIGNATURE_SIZE,
)


# =============================================================================
# KYBER768 TESTS
# =============================================================================

class TestKyber768:
    """Tests for Kyber768 KEM."""

    def test_generate_keypair(self):
        """Test keypair generation produces correct sizes."""
        kp = Kyber768.generate_keypair()
        assert isinstance(kp, KyberKeyPair)
        assert len(kp.public_key) == KYBER768_PUBLIC_KEY_SIZE
        assert len(kp.secret_key) == KYBER768_SECRET_KEY_SIZE

    def test_keypairs_are_unique(self):
        """Test that each keypair generation is unique."""
        kp1 = Kyber768.generate_keypair()
        kp2 = Kyber768.generate_keypair()
        assert kp1.public_key != kp2.public_key
        assert kp1.secret_key != kp2.secret_key

    def test_encapsulate(self):
        """Test encapsulation produces correct sizes."""
        kp = Kyber768.generate_keypair()
        result = Kyber768.encapsulate(kp.public_key)
        assert isinstance(result, EncapsulationResult)
        assert len(result.ciphertext) == KYBER768_CIPHERTEXT_SIZE
        assert len(result.shared_secret) == KYBER768_SHARED_SECRET_SIZE

    def test_encapsulate_invalid_key_size(self):
        """Test encapsulation rejects invalid key sizes."""
        with pytest.raises(ValueError, match="Invalid public key size"):
            Kyber768.encapsulate(b"too_short")

    def test_decapsulate(self):
        """Test decapsulation produces correct size."""
        kp = Kyber768.generate_keypair()
        encap = Kyber768.encapsulate(kp.public_key)
        shared = Kyber768.decapsulate(kp.secret_key, encap.ciphertext)
        assert len(shared) == KYBER768_SHARED_SECRET_SIZE

    def test_decapsulate_invalid_sizes(self):
        """Test decapsulation rejects invalid sizes."""
        kp = Kyber768.generate_keypair()
        encap = Kyber768.encapsulate(kp.public_key)

        with pytest.raises(ValueError, match="Invalid secret key size"):
            Kyber768.decapsulate(b"short", encap.ciphertext)

        with pytest.raises(ValueError, match="Invalid ciphertext size"):
            Kyber768.decapsulate(kp.secret_key, b"short")


# =============================================================================
# DILITHIUM3 TESTS
# =============================================================================

class TestDilithium3:
    """Tests for Dilithium3 signatures."""

    def test_generate_keypair(self):
        """Test keypair generation produces correct sizes."""
        kp = Dilithium3.generate_keypair()
        assert isinstance(kp, DilithiumKeyPair)
        assert len(kp.public_key) == DILITHIUM3_PUBLIC_KEY_SIZE
        assert len(kp.secret_key) == DILITHIUM3_SECRET_KEY_SIZE

    def test_keypairs_are_unique(self):
        """Test that each keypair generation is unique."""
        kp1 = Dilithium3.generate_keypair()
        kp2 = Dilithium3.generate_keypair()
        assert kp1.public_key != kp2.public_key
        assert kp1.secret_key != kp2.secret_key

    def test_sign(self):
        """Test signing produces correct signature size."""
        kp = Dilithium3.generate_keypair()
        message = b"test message to sign"
        signature = Dilithium3.sign(kp.secret_key, message)
        assert len(signature) == DILITHIUM3_SIGNATURE_SIZE

    def test_sign_invalid_key_size(self):
        """Test signing rejects invalid key sizes."""
        with pytest.raises(ValueError, match="Invalid secret key size"):
            Dilithium3.sign(b"short", b"message")

    def test_verify_correct_signature(self):
        """Test verification accepts correct signatures."""
        kp = Dilithium3.generate_keypair()
        message = b"test message"
        signature = Dilithium3.sign(kp.secret_key, message)
        assert Dilithium3.verify(kp.public_key, message, signature)

    def test_verify_invalid_signature_size(self):
        """Test verification rejects wrong signature sizes."""
        kp = Dilithium3.generate_keypair()
        assert not Dilithium3.verify(kp.public_key, b"msg", b"short")

    def test_verify_invalid_public_key_size(self):
        """Test verification rejects invalid public key sizes."""
        with pytest.raises(ValueError, match="Invalid public key size"):
            Dilithium3.verify(b"short", b"msg", b"x" * DILITHIUM3_SIGNATURE_SIZE)


# =============================================================================
# HYBRID KEY DERIVATION TESTS
# =============================================================================

class TestHybridKeyDerivation:
    """Tests for hybrid key derivation."""

    def test_derive_hybrid_key(self):
        """Test hybrid key derivation produces correct length."""
        pqc_secret = b"pqc_secret_material_32_bytes!!"
        classical_secret = b"classical_ecdh_secret_32bytes!"
        key = derive_hybrid_key(pqc_secret, classical_secret)
        assert len(key) == 32

    def test_derive_hybrid_key_custom_length(self):
        """Test hybrid key derivation with custom output length."""
        key = derive_hybrid_key(b"pqc", b"classical", output_length=64)
        assert len(key) == 64

    def test_derive_hybrid_key_deterministic(self):
        """Test hybrid key derivation is deterministic."""
        k1 = derive_hybrid_key(b"pqc", b"classical", b"context")
        k2 = derive_hybrid_key(b"pqc", b"classical", b"context")
        assert k1 == k2

    def test_derive_hybrid_key_context_matters(self):
        """Test different contexts produce different keys."""
        k1 = derive_hybrid_key(b"pqc", b"classical", b"context1")
        k2 = derive_hybrid_key(b"pqc", b"classical", b"context2")
        assert k1 != k2


# =============================================================================
# HARMONIC KEY DERIVATION TESTS
# =============================================================================

class TestHarmonicKeyDerivation:
    """Tests for harmonic key derivation functions."""

    def test_fast_harmonic_key_length(self):
        """Test fast harmonic key produces correct length."""
        key = fast_harmonic_key(b"input_key", dimension=3)
        assert len(key) == 32

    def test_fast_harmonic_key_custom_length(self):
        """Test fast harmonic key with custom output length."""
        key = fast_harmonic_key(b"input", dimension=3, output_length=64)
        assert len(key) == 64

    def test_fast_harmonic_key_deterministic(self):
        """Test fast harmonic key is deterministic with same salt."""
        salt = b"fixed_salt_16!!"
        k1 = fast_harmonic_key(b"input", dimension=3, salt=salt)
        k2 = fast_harmonic_key(b"input", dimension=3, salt=salt)
        assert k1 == k2

    def test_fast_harmonic_key_dimension_matters(self):
        """Test different dimensions produce different keys."""
        salt = b"fixed_salt_16!!"
        k1 = fast_harmonic_key(b"input", dimension=3, salt=salt)
        k2 = fast_harmonic_key(b"input", dimension=4, salt=salt)
        assert k1 != k2

    def test_harmonic_key_stretch_low_dimension(self):
        """Test harmonic key stretch with low dimension (fast)."""
        result = harmonic_key_stretch(b"input_key", dimension=1)
        assert isinstance(result, HarmonicKeyMaterial)
        assert len(result.base_key) == 32
        assert result.dimension == 1
        assert result.iteration_count == 2  # ceil(1.5)

    def test_harmonic_key_stretch_invalid_dimension(self):
        """Test harmonic key stretch rejects invalid dimensions."""
        with pytest.raises(ValueError, match="Dimension must be 1-6"):
            harmonic_key_stretch(b"input", dimension=0)
        with pytest.raises(ValueError, match="Dimension must be 1-6"):
            harmonic_key_stretch(b"input", dimension=7)

    def test_harmonic_multiplier_property(self):
        """Test harmonic multiplier property calculation."""
        result = harmonic_key_stretch(b"input", dimension=2)
        expected = harmonic_scale(2, DEFAULT_R)
        assert abs(result.harmonic_multiplier - expected) < 0.001


# =============================================================================
# HARMONIC PQC SESSION TESTS
# =============================================================================

class TestHarmonicPQCSession:
    """Tests for harmonic PQC session creation and verification."""

    @pytest.fixture
    def alice_keys(self):
        """Generate Alice's keypairs."""
        return {
            'kem': Kyber768.generate_keypair(),
            'sig': Dilithium3.generate_keypair()
        }

    @pytest.fixture
    def bob_keys(self):
        """Generate Bob's keypairs."""
        return {
            'kem': Kyber768.generate_keypair(),
            'sig': Dilithium3.generate_keypair()
        }

    def test_create_session(self, alice_keys, bob_keys):
        """Test session creation produces valid session."""
        session = create_harmonic_pqc_session(
            initiator_kem_keypair=alice_keys['kem'],
            responder_kem_public_key=bob_keys['kem'].public_key,
            initiator_sig_keypair=alice_keys['sig'],
            dimension=4
        )

        assert isinstance(session, HarmonicPQCSession)
        assert session.dimension == 4
        assert len(session.session_id) == 16
        assert len(session.encryption_key.base_key) == 32
        assert len(session.mac_key.base_key) == 32

    def test_session_effective_security_bits(self, alice_keys, bob_keys):
        """Test session has correct effective security bits."""
        session = create_harmonic_pqc_session(
            initiator_kem_keypair=alice_keys['kem'],
            responder_kem_public_key=bob_keys['kem'].public_key,
            initiator_sig_keypair=alice_keys['sig'],
            dimension=6
        )

        # Kyber768 base = 192, d=6 adds 21.06 bits
        expected = security_bits(192, 6, DEFAULT_R)
        assert abs(session.effective_security_bits - expected) < 0.01

    def test_session_security_level_name(self, alice_keys, bob_keys):
        """Test session security level names are correct."""
        for d in range(1, 7):
            session = create_harmonic_pqc_session(
                initiator_kem_keypair=alice_keys['kem'],
                responder_kem_public_key=bob_keys['kem'].public_key,
                initiator_sig_keypair=alice_keys['sig'],
                dimension=d
            )
            assert f"{d}D Harmonic" in session.get_security_level_name()

    def test_session_with_vector_key(self, alice_keys, bob_keys):
        """Test session creation with 6D vector key."""
        vector = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        session = create_harmonic_pqc_session(
            initiator_kem_keypair=alice_keys['kem'],
            responder_kem_public_key=bob_keys['kem'].public_key,
            initiator_sig_keypair=alice_keys['sig'],
            vector_key=vector
        )

        assert session.vector_key == vector

    def test_session_invalid_vector_key(self, alice_keys, bob_keys):
        """Test session rejects invalid vector key dimensions."""
        with pytest.raises(ValueError, match="6-dimensional"):
            create_harmonic_pqc_session(
                initiator_kem_keypair=alice_keys['kem'],
                responder_kem_public_key=bob_keys['kem'].public_key,
                initiator_sig_keypair=alice_keys['sig'],
                vector_key=(1.0, 2.0, 3.0)  # Only 3D
            )

    def test_verify_session(self, alice_keys, bob_keys):
        """Test session verification succeeds with correct keys."""
        session = create_harmonic_pqc_session(
            initiator_kem_keypair=alice_keys['kem'],
            responder_kem_public_key=bob_keys['kem'].public_key,
            initiator_sig_keypair=alice_keys['sig']
        )

        verified = verify_harmonic_pqc_session(
            session=session,
            responder_kem_keypair=bob_keys['kem'],
            initiator_sig_public_key=alice_keys['sig'].public_key
        )

        assert verified is not None
        assert verified.dimension == session.dimension


# =============================================================================
# VECTOR 6D KEY TESTS
# =============================================================================

class TestVector6DKey:
    """Tests for 6D vector key operations."""

    def test_vector_creation(self):
        """Test vector key creation."""
        v = Vector6DKey(x=1.0, y=2.0, z=3.0, velocity=4.0, priority=5.0, security=6.0)
        assert v.x == 1.0
        assert v.security == 6.0

    def test_as_tuple(self):
        """Test conversion to tuple."""
        v = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        t = v.as_tuple()
        assert t == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    def test_to_bytes_and_back(self):
        """Test serialization roundtrip."""
        v = Vector6DKey(1.5, -2.5, 3.5, 4.5, 5.5, 6.0)
        data = v.to_bytes()
        assert len(data) == 48  # 6 × 8 bytes

        v2 = Vector6DKey.from_bytes(data)
        assert abs(v2.x - v.x) < 1e-6
        assert abs(v2.y - v.y) < 1e-6
        assert abs(v2.z - v.z) < 1e-6

    def test_from_bytes_invalid_length(self):
        """Test deserialization rejects invalid length."""
        with pytest.raises(ValueError, match="Invalid vector key bytes length"):
            Vector6DKey.from_bytes(b"too_short")

    def test_distance_to_self_is_zero(self):
        """Test distance to self is zero."""
        v = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert v.distance_to(v) == 0.0

    def test_distance_is_symmetric(self):
        """Test distance is symmetric."""
        v1 = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        v2 = Vector6DKey(2.0, 3.0, 4.0, 5.0, 6.0, 1.0)
        assert abs(v1.distance_to(v2) - v2.distance_to(v1)) < 1e-10

    def test_random_generation(self):
        """Test random vector generation."""
        v1 = Vector6DKey.random()
        v2 = Vector6DKey.random()
        # Very unlikely to be equal
        assert v1.as_tuple() != v2.as_tuple()


class TestVectorKeyDerivation:
    """Tests for vector-based key derivation."""

    def test_derive_key_from_vector(self):
        """Test key derivation from vector."""
        v = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        key = derive_key_from_vector(v, salt=b"salt_value_16!!!")
        assert len(key) == 32

    def test_derive_key_deterministic(self):
        """Test vector key derivation is deterministic."""
        v = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        salt = b"fixed_salt_16!!!"
        k1 = derive_key_from_vector(v, salt=salt)
        k2 = derive_key_from_vector(v, salt=salt)
        assert k1 == k2

    def test_proximity_key_within_tolerance(self):
        """Test proximity key derivation within tolerance."""
        v1 = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        v2 = Vector6DKey(1.1, 2.1, 3.1, 4.1, 5.1, 6.0)  # Close
        salt = b"proximity_salt!!"

        key = vector_proximity_key(v1, v2, tolerance=10.0, salt=salt)
        assert key is not None
        assert len(key) == 32

    def test_proximity_key_outside_tolerance(self):
        """Test proximity key returns None when outside tolerance."""
        v1 = Vector6DKey(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        v2 = Vector6DKey(100.0, 100.0, 100.0, 100.0, 100.0, 1.0)  # Far
        salt = b"proximity_salt!!"

        key = vector_proximity_key(v1, v2, tolerance=1.0, salt=salt)
        assert key is None


# =============================================================================
# HARMONIC KYBER ORCHESTRATOR TESTS
# =============================================================================

class TestHarmonicKyberOrchestrator:
    """Tests for the high-level orchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orch = HarmonicKyberOrchestrator(dimension=5)
        assert orch.dimension == 5
        assert orch.R == DEFAULT_R

    def test_get_public_keys(self):
        """Test getting public keys."""
        orch = HarmonicKyberOrchestrator()
        kem_pk, sig_pk = orch.get_public_keys()
        assert len(kem_pk) == KYBER768_PUBLIC_KEY_SIZE
        assert len(sig_pk) == DILITHIUM3_PUBLIC_KEY_SIZE

    def test_create_and_verify_session(self):
        """Test full session workflow between two orchestrators."""
        alice = HarmonicKyberOrchestrator(dimension=4)
        bob = HarmonicKyberOrchestrator(dimension=4)

        # Alice creates session for Bob
        session = alice.create_session(bob.kem_keypair.public_key)
        assert session.dimension == 4

        # Bob verifies and completes session
        _, alice_sig_pk = alice.get_public_keys()
        verified = bob.verify_session(session, alice_sig_pk)
        assert verified is not None

    def test_create_session_with_vector(self):
        """Test session creation with vector key."""
        alice = HarmonicKyberOrchestrator()
        bob = HarmonicKyberOrchestrator()

        v = Vector6DKey(1.0, 2.0, 3.0, 4.0, 5.0, 3.0)
        session = alice.create_session(
            bob.kem_keypair.public_key,
            vector_key=v
        )

        assert session.vector_key == v.as_tuple()

    def test_get_security_analysis(self):
        """Test security analysis retrieval."""
        orch = HarmonicKyberOrchestrator(dimension=6)
        analysis = orch.get_security_analysis()

        assert analysis['base_algorithm'] == 'Kyber768'
        assert analysis['dimension'] == 6
        assert 'H_value' in analysis
        assert 'effective_security_bits' in analysis


# =============================================================================
# SECURITY ANALYSIS TESTS
# =============================================================================

class TestSecurityAnalysis:
    """Tests for security analysis utilities."""

    def test_analyze_harmonic_security(self):
        """Test security analysis function."""
        analysis = analyze_harmonic_security("Kyber768", dimension=6)

        assert analysis['base_security_bits'] == 192
        assert analysis['dimension'] == 6
        assert analysis['d_squared'] == 36

        # H(6, 1.5) should be ~2.18M
        assert analysis['H_value'] > 2_000_000
        assert analysis['H_value'] < 2_500_000

        # Effective bits should be ~213
        assert 212 < analysis['effective_security_bits'] < 214

    def test_analyze_unknown_algorithm(self):
        """Test analysis with unknown algorithm uses default."""
        analysis = analyze_harmonic_security("Unknown", dimension=3)
        assert analysis['base_security_bits'] == 128  # Default

    def test_print_security_table(self):
        """Test security table generation."""
        table = print_security_table()

        assert "AETHERMOORE" in table
        assert "R = 1.5" in table
        assert "AES-" in table

        # Should have entries for all 6 dimensions
        for d in range(1, 7):
            assert f"  {d} |" in table


# =============================================================================
# CONSTANTS VERIFICATION TESTS
# =============================================================================

class TestHarmonicConstants:
    """Tests verifying harmonic scaling constants are correct."""

    def test_harmonic_scale_d1(self):
        """Test H(1, 1.5) = 1.5^1 = 1.5."""
        h = harmonic_scale(1, 1.5)
        assert abs(h - 1.5) < 1e-10

    def test_harmonic_scale_d2(self):
        """Test H(2, 1.5) = 1.5^4 = 5.0625."""
        h = harmonic_scale(2, 1.5)
        assert abs(h - 5.0625) < 1e-10

    def test_harmonic_scale_d6(self):
        """Test H(6, 1.5) = 1.5^36."""
        h = harmonic_scale(6, 1.5)
        expected = 1.5 ** 36
        assert abs(h - expected) < 1e-5

    def test_security_bits_formula(self):
        """Test security bits formula: S = base + d² × log₂(R)."""
        base = 128
        d = 4
        R = 1.5

        s = security_bits(base, d, R)
        expected = base + (d * d) * math.log2(R)
        assert abs(s - expected) < 1e-10

    def test_security_dimension_enum_values(self):
        """Test SecurityDimension enum has correct values."""
        assert SecurityDimension.D1_BASIC.value == 1
        assert SecurityDimension.D6_MAXIMUM.value == 6
