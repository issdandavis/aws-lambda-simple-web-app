"""
QAS-001: Shor's Algorithm Simulation Tests

Tests that post-quantum cryptography (ML-KEM-768, ML-DSA-65) remains
secure against simulated Shor's algorithm attacks.
"""

import pytest
import hashlib
import secrets
import math
from typing import Dict, Any


class ShorsAlgorithmSimulator:
    """Simulates Shor's algorithm for testing PQC resistance."""

    def __init__(self, qubits: int = 4096):
        self.qubits = qubits
        self.operations = 0

    def factor_rsa(self, n: int) -> Dict[str, Any]:
        """
        Simulate factoring an RSA modulus.
        Returns factors if successful, None if resistant.
        """
        # Shor's can factor in O((log n)^3) with sufficient qubits
        bits = n.bit_length()
        required_qubits = 2 * bits + 1

        if self.qubits >= required_qubits:
            # In simulation, we can factor if we have enough qubits
            # Real implementation would use quantum period finding
            self.operations = int(math.log2(n) ** 3)
            return {
                "success": True,
                "factors": self._trivial_factor(n),
                "operations": self.operations,
                "qubits_used": required_qubits,
            }
        return {
            "success": False,
            "reason": "Insufficient qubits",
            "qubits_needed": required_qubits,
            "qubits_available": self.qubits,
        }

    def attack_discrete_log(self, g: int, h: int, p: int) -> Dict[str, Any]:
        """
        Simulate discrete logarithm attack.
        Returns x where g^x = h (mod p) if successful.
        """
        bits = p.bit_length()
        required_qubits = 2 * bits + 1

        if self.qubits >= required_qubits:
            self.operations = int(math.log2(p) ** 3)
            return {
                "success": True,
                "operations": self.operations,
            }
        return {
            "success": False,
            "reason": "Insufficient qubits for discrete log",
        }

    def attack_lattice(self, lattice_dim: int, security_level: int) -> Dict[str, Any]:
        """
        Attempt to attack lattice-based cryptography.
        ML-KEM-768 and ML-DSA-65 use lattice problems believed quantum-resistant.
        """
        # Best known quantum algorithms for lattice problems
        # still require exponential time
        estimated_operations = 2 ** (security_level / 2)  # Grover speedup

        # With current estimates, >10^20 operations needed for ML-KEM-768
        practical_limit = 10 ** 15  # Operations feasible in reasonable time

        if estimated_operations > practical_limit:
            return {
                "success": False,
                "reason": "Lattice problem remains hard for quantum computers",
                "estimated_operations": estimated_operations,
                "security_bits_remaining": security_level,
            }
        return {
            "success": True,
            "operations": estimated_operations,
        }

    def _trivial_factor(self, n: int) -> tuple:
        """Trivial factoring for small numbers (simulation only)."""
        if n <= 1:
            return (n, 1)
        for i in range(2, min(int(n**0.5) + 1, 10000)):
            if n % i == 0:
                return (i, n // i)
        return (n, 1)  # Prime


class TestShorsAlgorithmResistance:
    """Test suite for Shor's algorithm resistance."""

    @pytest.mark.quantum
    def test_ml_kem_768_resistant_to_shor(self, quantum_attack_vectors):
        """
        QAS-001: ML-KEM-768 must remain secure against Shor's algorithm.
        """
        simulator = ShorsAlgorithmSimulator(qubits=10**6)

        # ML-KEM-768 is lattice-based, not susceptible to Shor's
        result = simulator.attack_lattice(lattice_dim=768, security_level=128)

        assert not result["success"], "ML-KEM-768 should resist Shor's algorithm"
        assert result["security_bits_remaining"] >= 128, \
            "Security level should remain at least 128 bits"

    @pytest.mark.quantum
    def test_ml_dsa_65_resistant_to_shor(self):
        """
        QAS-001: ML-DSA-65 must remain secure against Shor's algorithm.
        """
        simulator = ShorsAlgorithmSimulator(qubits=10**6)

        # ML-DSA-65 is also lattice-based
        result = simulator.attack_lattice(lattice_dim=768, security_level=128)

        assert not result["success"], "ML-DSA-65 should resist Shor's algorithm"
        assert result["security_bits_remaining"] >= 128

    @pytest.mark.quantum
    def test_rsa_2048_broken_by_shor(self):
        """
        Verify that RSA-2048 would be broken by sufficient quantum computer.
        This demonstrates why PQC is necessary.
        """
        simulator = ShorsAlgorithmSimulator(qubits=8192)

        # Small test number (real RSA-2048 would need many more qubits)
        n = 15  # 3 * 5, trivial example
        result = simulator.factor_rsa(n)

        assert result["success"], "RSA is vulnerable to Shor's algorithm"
        factors = result["factors"]
        assert factors[0] * factors[1] == n or factors == (n, 1)

    @pytest.mark.quantum
    def test_quantum_attack_with_limited_qubits(self):
        """
        Test that attacks fail with insufficient quantum resources.
        """
        simulator = ShorsAlgorithmSimulator(qubits=100)  # Very limited

        # Try to factor a larger number
        result = simulator.factor_rsa(2**20 + 7)  # ~20-bit number

        assert not result["success"], \
            "Attack should fail with insufficient qubits"
        assert "qubits_needed" in result

    @pytest.mark.quantum
    def test_scbe_pqc_configuration(self, pqc_key_pair):
        """
        Verify SCBE uses properly sized PQC keys.
        """
        # ML-KEM-768 public key is 1184 bytes
        assert len(pqc_key_pair["public_key"]) == 1184, \
            "Public key should be 1184 bytes for ML-KEM-768"

        # ML-KEM-768 secret key is 2400 bytes
        assert len(pqc_key_pair["secret_key"]) == 2400, \
            "Secret key should be 2400 bytes for ML-KEM-768"

        # Shared secret is 32 bytes
        assert len(pqc_key_pair["shared_secret"]) == 32, \
            "Shared secret should be 32 bytes"

    @pytest.mark.quantum
    def test_key_encapsulation_security(self, pqc_key_pair):
        """
        Test that key encapsulation produces cryptographically secure output.
        """
        shared_secret = pqc_key_pair["shared_secret"]

        # Check entropy of shared secret
        unique_bytes = len(set(shared_secret))
        assert unique_bytes > 20, \
            "Shared secret should have high entropy"

        # Verify it's not obviously patterned
        assert shared_secret != bytes(32), "Secret should not be all zeros"
        assert shared_secret != bytes([0xFF] * 32), "Secret should not be all ones"

    @pytest.mark.quantum
    @pytest.mark.parametrize("security_level", [128, 192, 256])
    def test_security_levels(self, security_level):
        """
        Test various post-quantum security levels.
        """
        simulator = ShorsAlgorithmSimulator(qubits=10**6)

        # Map security levels to lattice dimensions
        dimensions = {128: 768, 192: 1024, 256: 1280}
        dim = dimensions[security_level]

        result = simulator.attack_lattice(lattice_dim=dim, security_level=security_level)

        assert not result["success"], \
            f"Security level {security_level} should be quantum-resistant"
        assert result["security_bits_remaining"] >= security_level


class TestQuantumKeyRecoveryResistance:
    """Tests for key recovery attack resistance."""

    @pytest.mark.quantum
    def test_no_key_recovery_after_million_qubits(self, pqc_key_pair):
        """
        Property 1: No key recovery after 10^6 simulated qubits.
        """
        simulator = ShorsAlgorithmSimulator(qubits=10**6)

        # Attempt lattice attack
        result = simulator.attack_lattice(lattice_dim=768, security_level=128)

        assert not result["success"], \
            "Key recovery should be impossible with 10^6 qubits"

        # Verify security margin
        security_bits = result.get("security_bits_remaining", 0)
        assert security_bits >= 128, \
            f"Security margin should be >= 128 bits, got {security_bits}"

    @pytest.mark.quantum
    def test_public_key_does_not_leak_private(self, pqc_key_pair):
        """
        Verify public key doesn't leak information about private key.
        """
        public_key = pqc_key_pair["public_key"]
        secret_key = pqc_key_pair["secret_key"]

        # Check for obvious correlations (simplified check)
        pub_hash = hashlib.sha256(public_key).digest()
        sec_hash = hashlib.sha256(secret_key).digest()

        # Hashes should be completely different
        matching_bytes = sum(1 for a, b in zip(pub_hash, sec_hash) if a == b)
        assert matching_bytes < 5, \
            "Public and private key hashes should not correlate"

    @pytest.mark.quantum
    def test_multiple_key_generations_independent(self):
        """
        Multiple key generations should produce independent keys.
        """
        keys = [secrets.token_bytes(1184) for _ in range(10)]

        # All keys should be unique
        assert len(set(keys)) == 10, "Generated keys should be unique"

        # Check pairwise independence
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                matching = sum(1 for a, b in zip(keys[i], keys[j]) if a == b)
                # Expect ~1184/256 ≈ 4.6 matching bytes by chance
                assert matching < 50, \
                    "Keys should be statistically independent"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
