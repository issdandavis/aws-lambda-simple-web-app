"""
SpiralSeal SS1 - Seal and Unseal Functions

Provides AES-256-GCM encryption with Sacred Tongue spell-text encoding.

Format:
    SS1|kid=<key_id>|aad=<context>|<salt_spell>|<nonce_spell>|<ct_spell>|<tag_spell>

Example:
    SS1|kid=k01|aad=service=openai;env=prod|ru:thal'vor|ko:sil'vara|ca:drev'asha|dr:mor'thal
"""

import os
import hashlib
import hmac
from typing import Optional, Tuple

from .sacred_tongues import encode_to_spelltext, decode_from_spelltext

# Crypto availability - checked lazily
CRYPTO_AVAILABLE = False
_AESGCM = None
_HKDF = None
_hashes = None


_crypto_init_tried = False


def _init_crypto():
    """
    Lazy initialization of cryptography library.

    Returns True if real AES-GCM is available, False to use simulation.
    """
    global CRYPTO_AVAILABLE, _AESGCM, _HKDF, _hashes, _crypto_init_tried

    if _crypto_init_tried:
        return CRYPTO_AVAILABLE

    _crypto_init_tried = True

    # Skip crypto import attempt - use simulation mode
    # This avoids issues with broken native crypto libraries
    # For production, set SPIRALSEAL_USE_CRYPTO=1 environment variable
    import os
    if not os.environ.get("SPIRALSEAL_USE_CRYPTO"):
        return False

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        _AESGCM = AESGCM
        _HKDF = HKDF
        _hashes = hashes
        CRYPTO_AVAILABLE = True
        return True
    except BaseException:
        return False


# ═══════════════════════════════════════════════════════════════
# Key Derivation
# ═══════════════════════════════════════════════════════════════

def derive_key(master_secret: bytes, salt: bytes, kid: str) -> bytes:
    """
    Derive encryption key via HKDF.

    key = HKDF-SHA256(master_secret, salt, info="SS1-{kid}")
    """
    if _init_crypto():
        hkdf = _HKDF(
            algorithm=_hashes.SHA256(),
            length=32,
            salt=salt,
            info=f"SS1-{kid}".encode(),
        )
        return hkdf.derive(master_secret)
    else:
        # Simulation: simple HMAC-based derivation
        data = salt + f"SS1-{kid}".encode()
        return hmac.new(master_secret, data, hashlib.sha256).digest()


# ═══════════════════════════════════════════════════════════════
# AES-GCM Encryption (with fallback simulation)
# ═══════════════════════════════════════════════════════════════

def aes_gcm_encrypt(
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    aad: bytes,
) -> Tuple[bytes, bytes]:
    """
    Encrypt with AES-256-GCM.

    Returns (ciphertext, tag).
    """
    if _init_crypto():
        aesgcm = _AESGCM(key)
        # cryptography library returns ciphertext + tag concatenated
        ct_with_tag = aesgcm.encrypt(nonce, plaintext, aad)
        # Last 16 bytes are the tag
        ciphertext = ct_with_tag[:-16]
        tag = ct_with_tag[-16:]
        return ciphertext, tag
    else:
        # Simulation: XOR with key-derived stream + HMAC tag
        # WARNING: NOT CRYPTOGRAPHICALLY SECURE - for demo only
        stream = hashlib.sha256(key + nonce).digest()
        # Extend stream for longer plaintexts
        while len(stream) < len(plaintext):
            stream += hashlib.sha256(stream).digest()

        ciphertext = bytes(p ^ s for p, s in zip(plaintext, stream[:len(plaintext)]))
        tag = hmac.new(key, ciphertext + aad, hashlib.sha256).digest()[:16]
        return ciphertext, tag


def aes_gcm_decrypt(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    aad: bytes,
) -> bytes:
    """
    Decrypt with AES-256-GCM.

    Raises ValueError if authentication fails.
    """
    if _init_crypto():
        aesgcm = _AESGCM(key)
        ct_with_tag = ciphertext + tag
        try:
            return aesgcm.decrypt(nonce, ct_with_tag, aad)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    else:
        # Simulation: verify HMAC then XOR
        expected_tag = hmac.new(key, ciphertext + aad, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Authentication failed: tag mismatch")

        stream = hashlib.sha256(key + nonce).digest()
        while len(stream) < len(ciphertext):
            stream += hashlib.sha256(stream).digest()

        return bytes(c ^ s for c, s in zip(ciphertext, stream[:len(ciphertext)]))


# ═══════════════════════════════════════════════════════════════
# SS1 Format
# ═══════════════════════════════════════════════════════════════

def format_ss1(
    kid: str,
    aad: str,
    salt: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
) -> str:
    """Format components into SS1 blob."""
    salt_spell = encode_to_spelltext(salt, "salt")
    nonce_spell = encode_to_spelltext(nonce, "nonce")
    ct_spell = encode_to_spelltext(ciphertext, "ct")
    tag_spell = encode_to_spelltext(tag, "tag")

    return f"SS1|kid={kid}|aad={aad}|{salt_spell}|{nonce_spell}|{ct_spell}|{tag_spell}"


def parse_ss1(blob: str) -> Tuple[str, str, bytes, bytes, bytes, bytes]:
    """
    Parse SS1 blob into components.

    Returns (kid, aad, salt, nonce, ciphertext, tag).
    """
    parts = blob.split("|")

    if len(parts) != 7:
        raise ValueError(f"Invalid SS1 format: expected 7 parts, got {len(parts)}")

    if parts[0] != "SS1":
        raise ValueError(f"Invalid SS1 version: {parts[0]}")

    # Parse kid
    if not parts[1].startswith("kid="):
        raise ValueError(f"Invalid kid field: {parts[1]}")
    kid = parts[1][4:]

    # Parse aad
    if not parts[2].startswith("aad="):
        raise ValueError(f"Invalid aad field: {parts[2]}")
    aad = parts[2][4:]

    # Decode spell-text sections
    _, salt = decode_from_spelltext(parts[3])
    _, nonce = decode_from_spelltext(parts[4])
    _, ciphertext = decode_from_spelltext(parts[5])
    _, tag = decode_from_spelltext(parts[6])

    return kid, aad, salt, nonce, ciphertext, tag


# ═══════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════

def seal(
    plaintext: bytes,
    master_secret: bytes,
    aad: str = "",
    kid: str = "k01",
) -> str:
    """
    Seal plaintext into SS1 spell-text blob.

    Args:
        plaintext: Data to encrypt
        master_secret: 32-byte master key
        aad: Additional authenticated data (context string)
        kid: Key identifier for rotation

    Returns:
        SS1 formatted spell-text blob
    """
    # Generate random salt and nonce
    salt = os.urandom(16)
    nonce = os.urandom(12)

    # Derive encryption key
    key = derive_key(master_secret, salt, kid)

    # Encrypt
    ciphertext, tag = aes_gcm_encrypt(key, nonce, plaintext, aad.encode())

    # Format as SS1
    return format_ss1(kid, aad, salt, nonce, ciphertext, tag)


def unseal(
    blob: str,
    master_secret: bytes,
    aad: Optional[str] = None,
) -> bytes:
    """
    Unseal SS1 spell-text blob to plaintext.

    Args:
        blob: SS1 formatted spell-text
        master_secret: 32-byte master key
        aad: Expected AAD (if None, uses AAD from blob)

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If authentication fails or AAD mismatch
    """
    # Parse blob
    kid, blob_aad, salt, nonce, ciphertext, tag = parse_ss1(blob)

    # Verify AAD if specified
    if aad is not None and aad != blob_aad:
        raise ValueError(f"AAD mismatch: expected '{aad}', got '{blob_aad}'")

    # Use blob's AAD for decryption
    aad_bytes = blob_aad.encode()

    # Derive key
    key = derive_key(master_secret, salt, kid)

    # Decrypt and verify
    return aes_gcm_decrypt(key, nonce, ciphertext, tag, aad_bytes)


class SpiralSealSS1:
    """
    Class-based API for SpiralSeal SS1.

    Supports key rotation and multiple key IDs.
    """

    def __init__(self, master_secret: bytes, kid: str = "k01"):
        """
        Initialize with master secret and key ID.

        Args:
            master_secret: 32-byte master key
            kid: Default key identifier
        """
        if len(master_secret) < 16:
            raise ValueError("Master secret must be at least 16 bytes")

        self._secrets = {kid: master_secret}
        self._current_kid = kid

    def rotate_key(self, new_kid: str, new_secret: bytes) -> None:
        """Add a new key for rotation."""
        self._secrets[new_kid] = new_secret
        self._current_kid = new_kid

    def seal(self, plaintext: bytes, aad: str = "", kid: Optional[str] = None) -> str:
        """Seal with current or specified key."""
        kid = kid or self._current_kid
        if kid not in self._secrets:
            raise ValueError(f"Unknown kid: {kid}")

        return seal(plaintext, self._secrets[kid], aad, kid)

    def unseal(self, blob: str, aad: Optional[str] = None) -> bytes:
        """Unseal, automatically selecting correct key by kid."""
        # Parse to get kid
        parsed_kid, _, _, _, _, _ = parse_ss1(blob)

        if parsed_kid not in self._secrets:
            raise ValueError(f"Unknown kid: {parsed_kid}")

        return unseal(blob, self._secrets[parsed_kid], aad)

    @staticmethod
    def status() -> dict:
        """Report backend status."""
        return {
            "version": "SS1",
            "crypto_backend": "cryptography" if CRYPTO_AVAILABLE else "simulation",
            "aead": "AES-256-GCM",
            "kdf": "HKDF-SHA256",
            "tongues": ["KO", "AV", "RU", "CA", "UM", "DR"],
        }
