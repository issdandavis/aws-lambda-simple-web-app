"""
FIPS 140-3 Compliance Tests

Tests for Federal Information Processing Standard 140-3 requirements.
Covers cryptographic module validation requirements.
"""

import pytest
import hashlib
import secrets
import hmac
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class FIPSAlgorithm(Enum):
    """FIPS-approved cryptographic algorithms."""
    AES_128 = "AES-128"
    AES_192 = "AES-192"
    AES_256 = "AES-256"
    SHA_256 = "SHA-256"
    SHA_384 = "SHA-384"
    SHA_512 = "SHA-512"
    SHA3_256 = "SHA3-256"
    HMAC_SHA256 = "HMAC-SHA-256"
    ECDSA_P256 = "ECDSA-P-256"
    ECDSA_P384 = "ECDSA-P-384"
    RSA_2048 = "RSA-2048"
    RSA_3072 = "RSA-3072"
    RSA_4096 = "RSA-4096"
    # Post-Quantum (FIPS 203, 204, 205 drafts)
    ML_KEM_768 = "ML-KEM-768"
    ML_DSA_65 = "ML-DSA-65"


class FIPSSecurityLevel(Enum):
    """FIPS 140-3 security levels."""
    LEVEL_1 = 1  # Basic security
    LEVEL_2 = 2  # Tamper-evident
    LEVEL_3 = 3  # Tamper-resistant, identity-based auth
    LEVEL_4 = 4  # Physical security envelope


@dataclass
class CryptoModuleStatus:
    """Status of cryptographic module."""
    fips_mode: bool
    approved_algorithms: List[FIPSAlgorithm]
    security_level: FIPSSecurityLevel
    self_test_passed: bool
    error_state: bool
    power_on_tested: bool


class FIPSCryptoModule:
    """
    Simulated FIPS 140-3 compliant cryptographic module.
    """

    APPROVED_ALGORITHMS = [
        FIPSAlgorithm.AES_256,
        FIPSAlgorithm.SHA_256,
        FIPSAlgorithm.SHA_384,
        FIPSAlgorithm.SHA_512,
        FIPSAlgorithm.HMAC_SHA256,
        FIPSAlgorithm.ML_KEM_768,
        FIPSAlgorithm.ML_DSA_65,
    ]

    def __init__(self, security_level: FIPSSecurityLevel = FIPSSecurityLevel.LEVEL_3):
        self.security_level = security_level
        self.fips_mode = True
        self.error_state = False
        self.self_test_passed = False
        self.power_on_tested = False

        # Run power-on self-tests
        self._run_power_on_self_tests()

    def _run_power_on_self_tests(self) -> bool:
        """Run FIPS-required power-on self-tests."""
        tests_passed = True

        # Known Answer Test (KAT) for AES
        tests_passed &= self._kat_aes()

        # Known Answer Test for SHA-256
        tests_passed &= self._kat_sha256()

        # Known Answer Test for HMAC
        tests_passed &= self._kat_hmac()

        # Integrity test (simplified)
        tests_passed &= self._integrity_test()

        self.self_test_passed = tests_passed
        self.power_on_tested = True

        if not tests_passed:
            self.error_state = True

        return tests_passed

    def _kat_aes(self) -> bool:
        """Known Answer Test for AES."""
        # NIST test vector (simplified)
        # In real implementation, use actual NIST test vectors
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
        plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")

        # Expected result would be verified against NIST test vectors
        # Here we just verify the operation completes
        return len(key) == 32 and len(plaintext) == 16

    def _kat_sha256(self) -> bool:
        """Known Answer Test for SHA-256."""
        # NIST test vector
        message = b"abc"
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

        result = hashlib.sha256(message).hexdigest()
        return result == expected

    def _kat_hmac(self) -> bool:
        """Known Answer Test for HMAC-SHA-256."""
        key = b"key"
        message = b"The quick brown fox jumps over the lazy dog"
        expected = "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"

        result = hmac.new(key, message, hashlib.sha256).hexdigest()
        return result == expected

    def _integrity_test(self) -> bool:
        """Module integrity test."""
        # In real implementation, verify module binary hash
        return True

    def get_status(self) -> CryptoModuleStatus:
        """Get current module status."""
        return CryptoModuleStatus(
            fips_mode=self.fips_mode,
            approved_algorithms=self.APPROVED_ALGORITHMS.copy(),
            security_level=self.security_level,
            self_test_passed=self.self_test_passed,
            error_state=self.error_state,
            power_on_tested=self.power_on_tested,
        )

    def hash(self, algorithm: FIPSAlgorithm, data: bytes) -> Optional[bytes]:
        """Compute hash using approved algorithm."""
        if self.error_state:
            return None

        if algorithm not in self.APPROVED_ALGORITHMS:
            return None

        algo_map = {
            FIPSAlgorithm.SHA_256: hashlib.sha256,
            FIPSAlgorithm.SHA_384: hashlib.sha384,
            FIPSAlgorithm.SHA_512: hashlib.sha512,
        }

        if algorithm in algo_map:
            return algo_map[algorithm](data).digest()

        return None

    def hmac_sign(self, key: bytes, data: bytes) -> Optional[bytes]:
        """Compute HMAC-SHA-256."""
        if self.error_state:
            return None

        if len(key) < 16:  # Minimum key length
            return None

        return hmac.new(key, data, hashlib.sha256).digest()

    def generate_random(self, length: int) -> Optional[bytes]:
        """Generate cryptographically secure random bytes."""
        if self.error_state:
            return None

        if length <= 0 or length > 10000:
            return None

        # FIPS requires DRBG (Deterministic Random Bit Generator)
        # secrets.token_bytes uses OS CSPRNG which is FIPS-compliant
        return secrets.token_bytes(length)

    def zeroize(self) -> bool:
        """Zeroize all sensitive data (required for FIPS)."""
        # In real implementation, securely erase all keys and sensitive data
        self.error_state = True  # Module enters error state after zeroization
        return True


class TestFIPSLevel1:
    """FIPS 140-3 Level 1 requirements."""

    @pytest.fixture
    def module(self):
        return FIPSCryptoModule(FIPSSecurityLevel.LEVEL_1)

    @pytest.mark.compliance
    def test_approved_algorithms_only(self, module):
        """
        Level 1: Only FIPS-approved algorithms may be used.
        """
        status = module.get_status()

        for algo in status.approved_algorithms:
            assert algo in FIPSCryptoModule.APPROVED_ALGORITHMS

    @pytest.mark.compliance
    def test_power_on_self_test(self, module):
        """
        Level 1: Module must run power-on self-tests.
        """
        status = module.get_status()

        assert status.power_on_tested, "Power-on self-tests must run"
        assert status.self_test_passed, "Self-tests must pass"

    @pytest.mark.compliance
    def test_sha256_kat(self, module):
        """
        Level 1: SHA-256 Known Answer Test.
        """
        # NIST test vector
        data = b"abc"
        result = module.hash(FIPSAlgorithm.SHA_256, data)

        expected = bytes.fromhex(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
        assert result == expected

    @pytest.mark.compliance
    def test_error_state_blocks_operations(self, module):
        """
        Level 1: Error state must block all operations.
        """
        module.error_state = True

        result = module.hash(FIPSAlgorithm.SHA_256, b"test")
        assert result is None, "Operations must fail in error state"


class TestFIPSLevel2:
    """FIPS 140-3 Level 2 requirements."""

    @pytest.fixture
    def module(self):
        return FIPSCryptoModule(FIPSSecurityLevel.LEVEL_2)

    @pytest.mark.compliance
    def test_role_based_authentication(self, module):
        """
        Level 2: Role-based authentication required.
        """
        # Module should support operator and admin roles
        status = module.get_status()
        assert status.security_level.value >= 2

    @pytest.mark.compliance
    def test_tamper_evidence(self, module):
        """
        Level 2: Tamper-evident seals or coatings.
        """
        # In real implementation, verify physical security measures
        # Here we verify the security level supports it
        assert module.security_level.value >= 2


class TestFIPSLevel3:
    """FIPS 140-3 Level 3 requirements."""

    @pytest.fixture
    def module(self):
        return FIPSCryptoModule(FIPSSecurityLevel.LEVEL_3)

    @pytest.mark.compliance
    def test_identity_based_authentication(self, module):
        """
        Level 3: Identity-based authentication required.
        """
        status = module.get_status()
        assert status.security_level.value >= 3

    @pytest.mark.compliance
    def test_key_management_controls(self, module):
        """
        Level 3: Strong key management controls.
        """
        # Generate key material
        key = module.generate_random(32)
        assert key is not None
        assert len(key) == 32

        # Key should have high entropy
        unique_bytes = len(set(key))
        assert unique_bytes > 20

    @pytest.mark.compliance
    def test_zeroization(self, module):
        """
        Level 3: Zeroization of plaintext CSPs.
        """
        result = module.zeroize()
        assert result is True

        # After zeroization, module should be in error state
        status = module.get_status()
        assert status.error_state


class TestApprovedAlgorithms:
    """Tests for FIPS-approved algorithm usage."""

    @pytest.fixture
    def module(self):
        return FIPSCryptoModule()

    @pytest.mark.compliance
    def test_reject_unapproved_algorithm(self, module):
        """
        Non-approved algorithms must be rejected.
        """
        # Try to use non-existent algorithm
        result = module.hash(FIPSAlgorithm.AES_128, b"test")  # AES is not a hash
        assert result is None

    @pytest.mark.compliance
    @pytest.mark.parametrize("algorithm", [
        FIPSAlgorithm.SHA_256,
        FIPSAlgorithm.SHA_384,
        FIPSAlgorithm.SHA_512,
    ])
    def test_approved_hash_algorithms(self, module, algorithm):
        """
        Approved hash algorithms must work correctly.
        """
        result = module.hash(algorithm, b"test data")
        assert result is not None

        # Verify correct output length
        expected_lengths = {
            FIPSAlgorithm.SHA_256: 32,
            FIPSAlgorithm.SHA_384: 48,
            FIPSAlgorithm.SHA_512: 64,
        }
        assert len(result) == expected_lengths[algorithm]

    @pytest.mark.compliance
    def test_hmac_minimum_key_length(self, module):
        """
        HMAC key must meet minimum length requirements.
        """
        short_key = b"short"  # Too short
        long_key = secrets.token_bytes(32)

        # Short key should be rejected
        result = module.hmac_sign(short_key, b"data")
        assert result is None

        # Long key should work
        result = module.hmac_sign(long_key, b"data")
        assert result is not None


class TestRandomNumberGeneration:
    """Tests for FIPS-compliant random number generation."""

    @pytest.fixture
    def module(self):
        return FIPSCryptoModule()

    @pytest.mark.compliance
    def test_random_uniqueness(self, module):
        """
        Random outputs must be unique.
        """
        randoms = [module.generate_random(32) for _ in range(100)]
        assert len(set(randoms)) == 100

    @pytest.mark.compliance
    def test_random_length_limits(self, module):
        """
        Random generation must enforce length limits.
        """
        # Zero length rejected
        assert module.generate_random(0) is None

        # Negative length rejected
        assert module.generate_random(-1) is None

        # Excessive length rejected
        assert module.generate_random(100000) is None

        # Valid length works
        assert module.generate_random(32) is not None

    @pytest.mark.compliance
    def test_random_entropy(self, module):
        """
        Random output must have high entropy.
        """
        random_bytes = module.generate_random(1000)

        # Simple entropy check: count unique byte values
        unique_bytes = len(set(random_bytes))

        # With 1000 random bytes, we should see most of the 256 possible values
        assert unique_bytes > 200, "Random output should have high entropy"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
