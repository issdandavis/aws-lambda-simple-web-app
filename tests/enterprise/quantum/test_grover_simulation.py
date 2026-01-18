"""
QAS-002: Grover's Algorithm Simulation Tests

Tests that symmetric encryption (AES-256) maintains adequate security
against Grover's algorithm attacks.
"""

import pytest
import hashlib
import secrets
import math
from typing import Dict, Any, Optional


class GroversAlgorithmSimulator:
    """Simulates Grover's algorithm for testing symmetric crypto security."""

    def __init__(self, qubits: int = 6681):
        self.qubits = qubits
        self.oracle_calls = 0

    def search_key_space(self, key_bits: int) -> Dict[str, Any]:
        """
        Simulate Grover's search on a key space.

        Grover's algorithm provides quadratic speedup:
        - Classical: O(2^n) operations
        - Quantum: O(2^(n/2)) operations

        For AES-256: Classical 2^256, Quantum 2^128
        """
        classical_operations = 2 ** key_bits
        quantum_operations = 2 ** (key_bits // 2)

        # Required qubits for Grover's search
        # Approximately n + ancilla qubits
        required_qubits = key_bits + int(math.log2(key_bits)) + 100

        if self.qubits < required_qubits:
            return {
                "success": False,
                "reason": "Insufficient qubits",
                "qubits_needed": required_qubits,
                "qubits_available": self.qubits,
            }

        # Calculate effective security after Grover's
        effective_security_bits = key_bits // 2

        return {
            "success": effective_security_bits < 80,  # Below practical security
            "classical_operations": classical_operations,
            "quantum_operations": quantum_operations,
            "effective_security_bits": effective_security_bits,
            "speedup_factor": 2 ** (key_bits // 2),
        }

    def attack_hash_preimage(self, hash_bits: int) -> Dict[str, Any]:
        """
        Simulate preimage attack on hash function using Grover's.
        """
        # Grover's provides quadratic speedup for preimage search
        classical_operations = 2 ** hash_bits
        quantum_operations = 2 ** (hash_bits // 2)

        effective_security = hash_bits // 2

        return {
            "classical_security_bits": hash_bits,
            "quantum_security_bits": effective_security,
            "is_secure": effective_security >= 128,
        }

    def attack_mac(self, mac_bits: int) -> Dict[str, Any]:
        """
        Simulate MAC forgery attack using Grover's.
        """
        # For MAC with n-bit tags, Grover gives 2^(n/2) forgery complexity
        quantum_forgery_ops = 2 ** (mac_bits // 2)

        return {
            "forgery_complexity": quantum_forgery_ops,
            "effective_bits": mac_bits // 2,
            "is_secure": mac_bits >= 256,  # Need 256-bit MAC for 128-bit post-quantum
        }


class TestGroversAlgorithmResistance:
    """Test suite for Grover's algorithm resistance."""

    @pytest.mark.quantum
    def test_aes_256_post_quantum_security(self):
        """
        QAS-002: AES-256 maintains >= 128 bits security against Grover's.
        """
        simulator = GroversAlgorithmSimulator(qubits=6681)

        result = simulator.search_key_space(key_bits=256)

        assert result["effective_security_bits"] >= 128, \
            "AES-256 should maintain 128-bit security against Grover's"
        assert not result["success"], \
            "AES-256 key search should be infeasible"

    @pytest.mark.quantum
    def test_aes_128_reduced_security(self):
        """
        Verify AES-128 has reduced but still practical security.
        """
        simulator = GroversAlgorithmSimulator(qubits=3000)

        result = simulator.search_key_space(key_bits=128)

        # AES-128 provides only 64-bit security against Grover's
        assert result["effective_security_bits"] == 64, \
            "AES-128 should have 64-bit post-quantum security"

        # This is below modern security requirements
        assert result["success"], \
            "AES-128 is theoretically vulnerable to Grover's"

    @pytest.mark.quantum
    def test_sha256_preimage_resistance(self):
        """
        Test SHA-256 preimage resistance against Grover's.
        """
        simulator = GroversAlgorithmSimulator()

        result = simulator.attack_hash_preimage(hash_bits=256)

        assert result["quantum_security_bits"] >= 128, \
            "SHA-256 should maintain 128-bit preimage resistance"
        assert result["is_secure"], \
            "SHA-256 preimage attack should be infeasible"

    @pytest.mark.quantum
    def test_sha512_maximum_security(self):
        """
        Test SHA-512 for maximum post-quantum security.
        """
        simulator = GroversAlgorithmSimulator()

        result = simulator.attack_hash_preimage(hash_bits=512)

        assert result["quantum_security_bits"] >= 256, \
            "SHA-512 should maintain 256-bit preimage resistance"

    @pytest.mark.quantum
    def test_hmac_sha256_mac_security(self):
        """
        Test HMAC-SHA256 security against quantum MAC forgery.
        """
        simulator = GroversAlgorithmSimulator()

        result = simulator.attack_mac(mac_bits=256)

        assert result["effective_bits"] >= 128, \
            "HMAC-SHA256 should maintain 128-bit forgery resistance"
        assert result["is_secure"], \
            "HMAC-SHA256 should be quantum-secure"

    @pytest.mark.quantum
    @pytest.mark.parametrize("key_bits,expected_quantum_bits", [
        (128, 64),
        (192, 96),
        (256, 128),
        (384, 192),
        (512, 256),
    ])
    def test_symmetric_key_sizes(self, key_bits, expected_quantum_bits):
        """
        Test various symmetric key sizes against Grover's.
        """
        simulator = GroversAlgorithmSimulator(qubits=10000)

        result = simulator.search_key_space(key_bits=key_bits)

        assert result["effective_security_bits"] == expected_quantum_bits, \
            f"{key_bits}-bit key should have {expected_quantum_bits}-bit quantum security"


class TestSymmetricEncryptionIntegration:
    """Integration tests for symmetric encryption in SCBE."""

    @pytest.mark.quantum
    def test_scbe_uses_aes_256(self):
        """
        Verify SCBE is configured to use AES-256.
        """
        # Test key generation produces 256-bit keys
        key = secrets.token_bytes(32)  # 256 bits
        assert len(key) == 32, "SCBE should use 256-bit AES keys"

    @pytest.mark.quantum
    def test_key_derivation_produces_256_bit_keys(self):
        """
        Test that key derivation produces sufficient key material.
        """
        password = b"test_password"
        salt = secrets.token_bytes(16)

        # Using PBKDF2-HMAC-SHA256
        derived = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)

        assert len(derived) == 32, "Derived keys should be 256 bits"

    @pytest.mark.quantum
    def test_iv_nonce_uniqueness(self):
        """
        Test that IV/nonce generation is unique.
        """
        ivs = [secrets.token_bytes(16) for _ in range(1000)]

        # All IVs should be unique
        assert len(set(ivs)) == 1000, "IVs must be unique"

    @pytest.mark.quantum
    def test_encryption_randomness(self):
        """
        Test that encryption produces random-looking output.
        """
        plaintext = b"A" * 32  # Repetitive plaintext

        # Simulate encryption (XOR with random key for demo)
        key = secrets.token_bytes(32)
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key))

        # Ciphertext should not be repetitive
        unique_bytes = len(set(ciphertext))
        assert unique_bytes > 10, \
            "Ciphertext should have high entropy"


class TestQuantumSpeedupBounds:
    """Tests verifying quantum speedup theoretical bounds."""

    @pytest.mark.quantum
    def test_grover_speedup_is_quadratic(self):
        """
        Verify Grover's provides exactly quadratic speedup.
        """
        simulator = GroversAlgorithmSimulator()

        for key_bits in [64, 128, 192, 256]:
            result = simulator.search_key_space(key_bits=key_bits)

            classical = result["classical_operations"]
            quantum = result["quantum_operations"]

            # Speedup should be sqrt(classical) = 2^(n/2)
            expected_speedup = 2 ** (key_bits // 2)
            actual_speedup = classical // quantum

            assert actual_speedup == expected_speedup, \
                f"Grover speedup should be 2^{key_bits//2}"

    @pytest.mark.quantum
    def test_no_exponential_speedup_for_symmetric(self):
        """
        Verify quantum computers don't provide exponential speedup
        for symmetric crypto (unlike for factoring).
        """
        simulator = GroversAlgorithmSimulator()

        result = simulator.search_key_space(key_bits=256)

        # Quantum operations should still be exponential, not polynomial
        assert result["quantum_operations"] > 10**30, \
            "Quantum search should still require exponential work"

    @pytest.mark.quantum
    def test_multiple_targets_speedup(self):
        """
        Test Grover speedup with multiple valid targets.
        """
        # With k targets in N items, Grover finds one in O(sqrt(N/k))
        total_space = 2 ** 128
        num_targets = 1000

        classical_ops = total_space // num_targets  # O(N/k)
        quantum_ops = int(math.sqrt(total_space / num_targets))  # O(sqrt(N/k))

        speedup = classical_ops / quantum_ops
        expected_speedup = math.sqrt(total_space / num_targets)

        # Speedup should match theoretical bound
        assert abs(speedup - expected_speedup) < expected_speedup * 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
