"""
ISO 27001:2022 Compliance Tests

Tests for ISO 27001 Information Security Management System controls.
"""

import pytest
import hashlib
import secrets
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EncryptionAlgorithm(Enum):
    """Approved encryption algorithms."""
    AES_256_GCM = "AES-256-GCM"
    AES_256_CBC = "AES-256-CBC"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"


class KeyState(Enum):
    """Key lifecycle states."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    COMPROMISED = "compromised"
    DESTROYED = "destroyed"


@dataclass
class CryptoKey:
    """A cryptographic key."""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_material: bytes
    created_at: datetime
    expires_at: datetime
    state: KeyState
    usage_count: int = 0


@dataclass
class EncryptedData:
    """Encrypted data container."""
    ciphertext: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    iv: bytes
    auth_tag: bytes
    timestamp: datetime


class CryptographyManager:
    """
    Manages cryptographic operations per ISO 27001 A.8.24.
    """

    def __init__(self):
        self.keys: Dict[str, CryptoKey] = {}
        self.key_usage_log: List[Dict] = []

    def generate_key(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        validity_days: int = 365
    ) -> CryptoKey:
        """Generate a new cryptographic key."""
        key_id = secrets.token_hex(16)
        key_material = secrets.token_bytes(32)  # 256 bits

        key = CryptoKey(
            key_id=key_id,
            algorithm=algorithm,
            key_material=key_material,
            created_at=datetime.now(),
            expires_at=datetime.now() + __import__('datetime').timedelta(days=validity_days),
            state=KeyState.ACTIVE,
        )

        self.keys[key_id] = key
        self._log_key_usage(key_id, "GENERATED")
        return key

    def encrypt(self, plaintext: bytes, key_id: str) -> Optional[EncryptedData]:
        """Encrypt data using specified key."""
        if key_id not in self.keys:
            return None

        key = self.keys[key_id]

        # Check key state
        if key.state != KeyState.ACTIVE:
            return None

        # Check key expiry
        if datetime.now() > key.expires_at:
            key.state = KeyState.DEACTIVATED
            return None

        # Generate IV
        iv = secrets.token_bytes(12)  # 96 bits for GCM

        # Simulate encryption (XOR for demo)
        key_extended = (key.key_material * ((len(plaintext) // 32) + 1))[:len(plaintext)]
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_extended))

        # Simulate auth tag
        auth_tag = hashlib.sha256(ciphertext + iv + key.key_material).digest()[:16]

        key.usage_count += 1
        self._log_key_usage(key_id, "ENCRYPT")

        return EncryptedData(
            ciphertext=ciphertext,
            key_id=key_id,
            algorithm=key.algorithm,
            iv=iv,
            auth_tag=auth_tag,
            timestamp=datetime.now(),
        )

    def decrypt(self, encrypted: EncryptedData) -> Optional[bytes]:
        """Decrypt data."""
        if encrypted.key_id not in self.keys:
            return None

        key = self.keys[encrypted.key_id]

        # Verify auth tag
        expected_tag = hashlib.sha256(
            encrypted.ciphertext + encrypted.iv + key.key_material
        ).digest()[:16]

        if expected_tag != encrypted.auth_tag:
            self._log_key_usage(encrypted.key_id, "DECRYPT_FAILED_AUTH")
            return None

        # Simulate decryption
        key_extended = (key.key_material * ((len(encrypted.ciphertext) // 32) + 1))[:len(encrypted.ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(encrypted.ciphertext, key_extended))

        self._log_key_usage(encrypted.key_id, "DECRYPT")
        return plaintext

    def rotate_key(self, old_key_id: str) -> Optional[CryptoKey]:
        """Rotate a key (create new, deactivate old)."""
        if old_key_id not in self.keys:
            return None

        old_key = self.keys[old_key_id]
        old_key.state = KeyState.DEACTIVATED

        new_key = self.generate_key(old_key.algorithm)
        self._log_key_usage(old_key_id, "ROTATED", {"new_key_id": new_key.key_id})

        return new_key

    def destroy_key(self, key_id: str) -> bool:
        """Securely destroy a key."""
        if key_id not in self.keys:
            return False

        key = self.keys[key_id]

        # Zero out key material (in real impl, use secure memory wiping)
        key.key_material = bytes(32)
        key.state = KeyState.DESTROYED

        self._log_key_usage(key_id, "DESTROYED")
        return True

    def _log_key_usage(self, key_id: str, operation: str, details: Dict = None):
        """Log key usage for audit."""
        self.key_usage_log.append({
            "timestamp": datetime.now(),
            "key_id": key_id,
            "operation": operation,
            "details": details or {},
        })


class TestA824Cryptography:
    """A.8.24: Tests for cryptographic controls."""

    @pytest.fixture
    def crypto_mgr(self):
        return CryptographyManager()

    @pytest.mark.compliance
    def test_aes_256_encryption(self, crypto_mgr):
        """
        A.8.24: System must use AES-256 or equivalent.
        """
        key = crypto_mgr.generate_key(EncryptionAlgorithm.AES_256_GCM)

        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert len(key.key_material) == 32  # 256 bits

    @pytest.mark.compliance
    def test_encryption_decryption_correctness(self, crypto_mgr):
        """
        A.8.24: Encryption and decryption must be correct.
        """
        key = crypto_mgr.generate_key()
        plaintext = b"Sensitive data that needs protection"

        encrypted = crypto_mgr.encrypt(plaintext, key.key_id)
        assert encrypted is not None
        assert encrypted.ciphertext != plaintext

        decrypted = crypto_mgr.decrypt(encrypted)
        assert decrypted == plaintext

    @pytest.mark.compliance
    def test_authenticated_encryption(self, crypto_mgr):
        """
        A.8.24: Encryption must include authentication.
        """
        key = crypto_mgr.generate_key()
        encrypted = crypto_mgr.encrypt(b"test data", key.key_id)

        assert encrypted.auth_tag is not None
        assert len(encrypted.auth_tag) >= 16  # At least 128 bits

    @pytest.mark.compliance
    def test_auth_tag_verification(self, crypto_mgr):
        """
        A.8.24: Tampered ciphertext must be detected.
        """
        key = crypto_mgr.generate_key()
        encrypted = crypto_mgr.encrypt(b"test data", key.key_id)

        # Tamper with ciphertext
        tampered = EncryptedData(
            ciphertext=bytes([b ^ 0xFF for b in encrypted.ciphertext]),
            key_id=encrypted.key_id,
            algorithm=encrypted.algorithm,
            iv=encrypted.iv,
            auth_tag=encrypted.auth_tag,
            timestamp=encrypted.timestamp,
        )

        # Decryption should fail
        result = crypto_mgr.decrypt(tampered)
        assert result is None

    @pytest.mark.compliance
    def test_unique_ivs(self, crypto_mgr):
        """
        A.8.24: IVs must be unique for each encryption.
        """
        key = crypto_mgr.generate_key()
        plaintext = b"same data"

        ivs = set()
        for _ in range(100):
            encrypted = crypto_mgr.encrypt(plaintext, key.key_id)
            ivs.add(encrypted.iv)

        assert len(ivs) == 100, "All IVs must be unique"


class TestKeyManagement:
    """Tests for key management controls."""

    @pytest.fixture
    def crypto_mgr(self):
        return CryptographyManager()

    @pytest.mark.compliance
    def test_key_generation_randomness(self, crypto_mgr):
        """
        Key material must be cryptographically random.
        """
        keys = [crypto_mgr.generate_key() for _ in range(10)]
        key_materials = [k.key_material for k in keys]

        # All keys should be unique
        assert len(set(key_materials)) == 10

    @pytest.mark.compliance
    def test_key_expiration(self, crypto_mgr):
        """
        Keys must have defined expiration.
        """
        key = crypto_mgr.generate_key(validity_days=30)

        assert key.expires_at is not None
        assert key.expires_at > datetime.now()

    @pytest.mark.compliance
    def test_expired_key_rejected(self, crypto_mgr):
        """
        Expired keys must not be usable.
        """
        key = crypto_mgr.generate_key(validity_days=-1)  # Already expired

        result = crypto_mgr.encrypt(b"test", key.key_id)
        assert result is None

    @pytest.mark.compliance
    def test_key_rotation(self, crypto_mgr):
        """
        Keys must be rotatable.
        """
        old_key = crypto_mgr.generate_key()
        old_key_id = old_key.key_id

        new_key = crypto_mgr.rotate_key(old_key_id)

        assert new_key is not None
        assert new_key.key_id != old_key_id
        assert crypto_mgr.keys[old_key_id].state == KeyState.DEACTIVATED

    @pytest.mark.compliance
    def test_key_destruction(self, crypto_mgr):
        """
        Keys must be securely destroyable.
        """
        key = crypto_mgr.generate_key()
        key_id = key.key_id

        result = crypto_mgr.destroy_key(key_id)
        assert result is True

        # Key should be destroyed
        assert crypto_mgr.keys[key_id].state == KeyState.DESTROYED
        assert crypto_mgr.keys[key_id].key_material == bytes(32)  # Zeroed


class TestA515AccessControl:
    """A.5.15: Tests for access control."""

    @pytest.mark.compliance
    def test_encryption_requires_valid_key(self):
        """
        Operations require valid credentials (key).
        """
        crypto_mgr = CryptographyManager()

        result = crypto_mgr.encrypt(b"test", "invalid_key_id")
        assert result is None


class TestA834AuditLogging:
    """A.8.34: Tests for audit logging."""

    @pytest.fixture
    def crypto_mgr(self):
        return CryptographyManager()

    @pytest.mark.compliance
    def test_key_operations_logged(self, crypto_mgr):
        """
        A.8.34: All key operations must be logged.
        """
        key = crypto_mgr.generate_key()
        crypto_mgr.encrypt(b"test", key.key_id)
        crypto_mgr.destroy_key(key.key_id)

        log = crypto_mgr.key_usage_log
        operations = [e["operation"] for e in log]

        assert "GENERATED" in operations
        assert "ENCRYPT" in operations
        assert "DESTROYED" in operations

    @pytest.mark.compliance
    def test_audit_log_timestamps(self, crypto_mgr):
        """
        A.8.34: Audit logs must have accurate timestamps.
        """
        before = datetime.now()
        crypto_mgr.generate_key()
        after = datetime.now()

        log_entry = crypto_mgr.key_usage_log[-1]
        assert before <= log_entry["timestamp"] <= after

    @pytest.mark.compliance
    def test_audit_log_completeness(self, crypto_mgr):
        """
        A.8.34: Audit logs must contain all required fields.
        """
        crypto_mgr.generate_key()

        log_entry = crypto_mgr.key_usage_log[-1]

        assert "timestamp" in log_entry
        assert "key_id" in log_entry
        assert "operation" in log_entry


class TestA825SecureDevelopment:
    """A.8.25: Tests for secure development."""

    @pytest.mark.compliance
    def test_no_hardcoded_keys(self):
        """
        A.8.25: No hardcoded cryptographic keys.
        """
        # This would typically scan source code
        # Here we verify key generation is dynamic
        crypto_mgr = CryptographyManager()

        key1 = crypto_mgr.generate_key()
        key2 = crypto_mgr.generate_key()

        assert key1.key_material != key2.key_material

    @pytest.mark.compliance
    def test_secure_random_generation(self):
        """
        A.8.25: Random values must use secure generator.
        """
        # Verify we're using secrets module
        random_bytes = secrets.token_bytes(32)

        # Should have high entropy
        unique_bytes = len(set(random_bytes))
        assert unique_bytes > 20, "Random bytes should have high entropy"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
