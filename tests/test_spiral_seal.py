#!/usr/bin/env python3
"""
Tests for SpiralSeal SS1 - Sacred Tongue Encryption Envelope.

Tests cover:
- Sacred Tongue tokenization (6 tongues × 256 tokens)
- SpiralSeal encryption/decryption round-trip
- SS1 wire format serialization
- VeiledSeal redaction wrapper
- PQCSpiralSeal with Dilithium3 signatures
- Edge cases and error handling
"""

import pytest
import os
import time
import hashlib

from symphonic_cipher.scbe_aethermoore.spiral_seal import (
    # Sacred Tongues
    SacredTongue,
    SacredTongueTokenizer,
    Token,
    TONGUE_WORDLISTS,
    DOMAIN_TONGUE_MAP,
    get_tongue_for_domain,
    get_tokenizer,
    get_combined_alphabet,
    get_magical_signature,

    # SpiralSeal
    SpiralSeal,
    VeiledSeal,
    PQCSpiralSeal,
    SpiralSealResult,
    VeiledSealResult,
    KDFType,
    AEADType,
    quick_seal,
    quick_unseal,
    get_crypto_backend_info,

    # Constants
    SS1_VERSION,
    SALT_SIZE,
    TAG_SIZE,
)


# =============================================================================
# SACRED TONGUE TOKENIZER TESTS
# =============================================================================

class TestSacredTongue:
    """Test the SacredTongue enum and wordlists."""

    def test_six_tongues_exist(self):
        """Verify all six sacred tongues are defined."""
        tongues = list(SacredTongue)
        assert len(tongues) == 6
        assert SacredTongue.KORAELIN in tongues
        assert SacredTongue.AVALI in tongues
        assert SacredTongue.RUNETHIC in tongues
        assert SacredTongue.CASSISIVADAN in tongues
        assert SacredTongue.UMBROTH in tongues
        assert SacredTongue.DRAUMRIC in tongues

    def test_tongue_values(self):
        """Verify tongue short codes."""
        assert SacredTongue.KORAELIN.value == "ko"
        assert SacredTongue.AVALI.value == "av"
        assert SacredTongue.RUNETHIC.value == "ru"
        assert SacredTongue.CASSISIVADAN.value == "ca"
        assert SacredTongue.UMBROTH.value == "um"
        assert SacredTongue.DRAUMRIC.value == "dr"

    def test_all_wordlists_have_256_tokens(self):
        """Each tongue must have exactly 16×16=256 unique tokens."""
        for tongue in SacredTongue:
            prefixes, suffixes = TONGUE_WORDLISTS[tongue]
            assert len(prefixes) == 16, f"{tongue.name} has {len(prefixes)} prefixes"
            assert len(suffixes) == 16, f"{tongue.name} has {len(suffixes)} suffixes"
            # Verify all combinations are unique
            tokens = set()
            for p in prefixes:
                for s in suffixes:
                    token = f"{p}'{s}"
                    assert token not in tokens, f"Duplicate token {token} in {tongue.name}"
                    tokens.add(token)
            assert len(tokens) == 256

    def test_domain_mappings(self):
        """Verify domain-to-tongue mappings."""
        assert get_tongue_for_domain("aad") == SacredTongue.AVALI
        assert get_tongue_for_domain("salt") == SacredTongue.RUNETHIC
        assert get_tongue_for_domain("nonce") == SacredTongue.KORAELIN
        assert get_tongue_for_domain("ct") == SacredTongue.CASSISIVADAN
        assert get_tongue_for_domain("tag") == SacredTongue.DRAUMRIC
        assert get_tongue_for_domain("veil") == SacredTongue.UMBROTH

    def test_invalid_domain_raises(self):
        """Unknown domain should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            get_tongue_for_domain("invalid")


class TestSacredTongueTokenizer:
    """Test the SacredTongueTokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        return SacredTongueTokenizer()

    def test_encode_single_byte(self, tokenizer):
        """Encode a single byte and verify token structure."""
        tokens = tokenizer.encode(b"\x00", SacredTongue.KORAELIN)
        assert len(tokens) == 1
        token = tokens[0]
        assert token.tongue == SacredTongue.KORAELIN
        assert token.byte_value == 0

    def test_encode_all_bytes(self, tokenizer):
        """Encode all 256 bytes and verify round-trip."""
        all_bytes = bytes(range(256))
        for tongue in SacredTongue:
            tokens = tokenizer.encode(all_bytes, tongue)
            assert len(tokens) == 256
            # Decode back
            decoded = tokenizer.decode(tokens)
            assert decoded == all_bytes

    def test_encode_to_string_format(self, tokenizer):
        """Verify string format: tongue:prefix'suffix."""
        result = tokenizer.encode_to_string(b"\x00", SacredTongue.KORAELIN)
        assert result.startswith("ko:")
        assert "'" in result

    def test_decode_from_string_roundtrip(self, tokenizer):
        """String encoding round-trip."""
        original = b"Hello, World!"
        for tongue in SacredTongue:
            encoded = tokenizer.encode_to_string(original, tongue, " ")
            decoded = tokenizer.decode_from_string(encoded, tongue, " ")
            assert decoded == original

    def test_token_to_byte_mapping(self, tokenizer):
        """Verify byte = prefix_index * 16 + suffix_index."""
        for tongue in SacredTongue:
            prefixes, suffixes = TONGUE_WORDLISTS[tongue]
            for byte_val in range(256):
                expected_prefix_idx = byte_val >> 4
                expected_suffix_idx = byte_val & 0x0F
                token = tokenizer.get_token_for_byte(byte_val, tongue)
                assert token.prefix == prefixes[expected_prefix_idx]
                assert token.suffix == suffixes[expected_suffix_idx]
                assert token.byte_value == byte_val

    def test_decode_unknown_token_raises(self, tokenizer):
        """Unknown token should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown token"):
            tokenizer.decode_from_string("invalid'token", SacredTongue.KORAELIN)

    def test_get_tokenizer_singleton(self):
        """get_tokenizer() should return same instance."""
        t1 = get_tokenizer()
        t2 = get_tokenizer()
        assert t1 is t2

    def test_magical_signature_unique_per_tongue(self):
        """Each tongue should have a unique magical signature."""
        signatures = set()
        for tongue in SacredTongue:
            sig = get_magical_signature(tongue)
            assert len(sig) == 16  # 16 hex chars
            assert sig not in signatures
            signatures.add(sig)

    def test_combined_alphabet(self):
        """Verify combined alphabet statistics."""
        alphabet = get_combined_alphabet()
        assert "prefixes" in alphabet
        assert "suffixes" in alphabet
        assert "total_unique_prefixes" in alphabet
        assert "total_unique_suffixes" in alphabet
        # Should have many unique prefixes across all tongues
        assert alphabet["total_unique_prefixes"] > 50


# =============================================================================
# SPIRAL SEAL ENCRYPTION TESTS
# =============================================================================

class TestSpiralSeal:
    """Test the SpiralSeal encryption class."""

    @pytest.fixture
    def seal(self):
        return SpiralSeal(master_password=b"test_password_123")

    def test_seal_unseal_roundtrip(self, seal):
        """Basic encryption/decryption round-trip."""
        plaintext = b"Hello, Spiralverse!"
        result = seal.seal(plaintext)

        # Verify result structure
        assert isinstance(result, SpiralSealResult)
        assert result.ciphertext != plaintext
        assert len(result.salt) == SALT_SIZE
        assert len(result.tag) == TAG_SIZE

        # Decrypt
        decrypted = seal.unseal(
            result.salt, result.nonce, result.ciphertext, result.tag
        )
        assert decrypted == plaintext

    def test_seal_with_aad(self, seal):
        """Encryption with associated authenticated data."""
        plaintext = b"sensitive data"
        aad = b"user:admin|action:read"

        result = seal.seal(plaintext, aad=aad)
        assert result.aad == aad
        assert result.aad_tokens != ""

        # Decrypt with correct AAD
        decrypted = seal.unseal(
            result.salt, result.nonce, result.ciphertext, result.tag, aad
        )
        assert decrypted == plaintext

    def test_seal_wrong_aad_fails(self, seal):
        """Decryption with wrong AAD should fail."""
        plaintext = b"sensitive data"
        result = seal.seal(plaintext, aad=b"correct_aad")

        with pytest.raises(Exception):  # Could be ValueError or authentication error
            seal.unseal(
                result.salt, result.nonce, result.ciphertext, result.tag, b"wrong_aad"
            )

    def test_tokens_use_correct_tongues(self, seal):
        """Verify each component uses its assigned Sacred Tongue."""
        result = seal.seal(b"test", aad=b"context")

        # Check token prefixes
        assert result.salt_tokens.startswith("ru:")  # Runethic
        assert result.nonce_tokens.startswith("ko:")  # Kor'aelin
        assert result.ct_tokens.startswith("ca:")  # Cassisivadan
        assert result.tag_tokens.startswith("dr:")  # Draumric
        assert result.aad_tokens.startswith("av:")  # Avali

    def test_unseal_from_tokens(self, seal):
        """Decrypt from token strings."""
        plaintext = b"token-based decryption test"
        result = seal.seal(plaintext)

        # Strip tongue prefixes for decode
        def strip_prefix(s):
            return s.split(":", 1)[1] if ":" in s else s

        decrypted = seal.unseal_tokens(
            strip_prefix(result.salt_tokens),
            strip_prefix(result.nonce_tokens),
            strip_prefix(result.ct_tokens),
            strip_prefix(result.tag_tokens),
        )
        assert decrypted == plaintext


class TestSS1Format:
    """Test SS1 wire format serialization."""

    @pytest.fixture
    def seal(self):
        return SpiralSeal(master_password=b"format_test")

    def test_ss1_string_format(self, seal):
        """Verify SS1 string format structure."""
        result = seal.seal(b"test", aad=b"ctx")
        ss1_string = result.to_ss1_string()

        assert ss1_string.startswith("SS1|")
        assert "kid=" in ss1_string
        assert "aad=av:" in ss1_string
        assert "salt=ru:" in ss1_string
        assert "nonce=ko:" in ss1_string
        assert "ct=ca:" in ss1_string
        assert "tag=dr:" in ss1_string

    def test_ss1_string_roundtrip(self, seal):
        """SS1 string serialization round-trip."""
        plaintext = b"Round-trip test data!"
        aad = b"test_context"

        result = seal.seal(plaintext, aad=aad)
        ss1_string = result.to_ss1_string()

        # Decrypt from SS1 string
        decrypted = seal.unseal_string(ss1_string)
        assert decrypted == plaintext

    def test_ss1_string_invalid_version_fails(self, seal):
        """Invalid version should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown version"):
            seal.unseal_string("SS2|kid=abc|salt=ru:test")

    def test_to_dict(self, seal):
        """Test dictionary serialization."""
        result = seal.seal(b"dict test")
        d = result.to_dict()

        assert d["version"] == "SS1"
        assert "key_id" in d
        assert "salt" in d
        assert "nonce" in d
        assert "ciphertext" in d
        assert "tag" in d
        assert "salt_tokens" in d
        assert "nonce_tokens" in d


class TestQuickFunctions:
    """Test convenience functions."""

    def test_quick_seal_unseal(self):
        """quick_seal and quick_unseal round-trip."""
        plaintext = b"Quick test!"
        password = b"quick_password"

        ss1_string = quick_seal(plaintext, password)
        assert ss1_string.startswith("SS1|")

        decrypted = quick_unseal(ss1_string, password)
        assert decrypted == plaintext

    def test_quick_seal_with_aad(self):
        """quick_seal with AAD."""
        plaintext = b"Data with context"
        password = b"password123"
        aad = b"important_context"

        ss1_string = quick_seal(plaintext, password, aad)
        assert "aad=av:" in ss1_string


class TestVeiledSeal:
    """Test VeiledSeal with Umbroth redaction wrapper."""

    @pytest.fixture
    def seal(self):
        return VeiledSeal(master_password=b"veiled_test")

    def test_veiled_seal_creates_veil_marker(self, seal):
        """VeiledSeal should create an Umbroth veil marker."""
        result = seal.seal_veiled(b"secret data")

        assert isinstance(result, VeiledSealResult)
        assert result.veil_marker != ""
        assert "'" in result.veil_marker  # Umbroth token format

    def test_veiled_seal_log_safe(self, seal):
        """Log-safe representation hides content."""
        result = seal.seal_veiled(b"sensitive")
        log_safe = result.to_log_safe()

        assert "VEILED" in log_safe
        assert result.veil_marker in log_safe
        assert "sensitive" not in log_safe

    def test_veiled_seal_full_ss1_available(self, seal):
        """Full SS1 string is still accessible."""
        result = seal.seal_veiled(b"data")
        ss1 = result.to_ss1_string()

        assert ss1.startswith("SS1|")

    def test_veiled_seal_custom_veil_id(self, seal):
        """Custom veil ID produces consistent marker."""
        result1 = seal.seal_veiled(b"test", veil_id="my_veil")
        result2 = seal.seal_veiled(b"test", veil_id="my_veil")

        assert result1.veil_marker == result2.veil_marker


class TestPQCSpiralSeal:
    """Test PQC-enhanced SpiralSeal."""

    def test_pqc_seal_basic(self):
        """Basic PQCSpiralSeal without keys."""
        seal = PQCSpiralSeal(master_key=os.urandom(32))
        result = seal.seal(b"pqc test")

        assert isinstance(result, SpiralSealResult)

    def test_pqc_seal_with_signing(self):
        """PQCSpiralSeal with Dilithium3 signing."""
        # Try to get Dilithium3
        try:
            from symphonic_cipher.scbe_aethermoore.pqc import Dilithium3
            sig_keys = Dilithium3.generate_keypair()

            seal = PQCSpiralSeal(
                master_key=os.urandom(32),
                signing_secret_key=sig_keys.secret_key
            )

            result, signature = seal.seal_signed(b"signed data")

            assert result is not None
            if seal.pqc_available:
                assert signature is not None
                # Verify signature
                sign_data = (
                    result.key_id +
                    result.salt +
                    result.nonce +
                    result.ciphertext +
                    result.tag
                )
                assert Dilithium3.verify(sig_keys.public_key, sign_data, signature)

        except ImportError:
            pytest.skip("PQC module not available")

    def test_pqc_seal_with_kyber(self):
        """PQCSpiralSeal with Kyber768 key exchange."""
        try:
            from symphonic_cipher.scbe_aethermoore.pqc import Kyber768

            # Generate recipient keypair
            recipient = Kyber768.generate_keypair()

            # Create seal with recipient's public key
            seal = PQCSpiralSeal(recipient_public_key=recipient.public_key)

            result = seal.seal(b"for recipient")
            assert result is not None

            # Verify encapsulated ciphertext is available
            if seal.pqc_available:
                assert seal.encapsulated_ciphertext is not None

        except ImportError:
            pytest.skip("PQC module not available")


class TestKDFAndAEADTypes:
    """Test different KDF and AEAD configurations."""

    def test_scrypt_kdf(self):
        """Test with scrypt KDF."""
        seal = SpiralSeal(
            master_password=b"scrypt_test",
            kdf_type=KDFType.SCRYPT
        )
        result = seal.seal(b"scrypt data")
        assert result.kdf_type == KDFType.SCRYPT

        decrypted = seal.unseal(
            result.salt, result.nonce, result.ciphertext, result.tag
        )
        assert decrypted == b"scrypt data"

    def test_master_key_instead_of_password(self):
        """Use direct master key instead of password."""
        key = os.urandom(32)
        seal = SpiralSeal(master_key=key)

        result = seal.seal(b"key data")
        decrypted = seal.unseal(
            result.salt, result.nonce, result.ciphertext, result.tag
        )
        assert decrypted == b"key data"

    def test_crypto_backend_info(self):
        """get_crypto_backend_info returns valid structure."""
        info = get_crypto_backend_info()

        assert "nacl_available" in info
        assert "argon2_available" in info
        assert "cryptography_available" in info
        assert "recommended_kdf" in info
        assert "recommended_aead" in info


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_plaintext(self):
        """Encrypt empty plaintext."""
        seal = SpiralSeal(master_password=b"empty_test")
        result = seal.seal(b"")

        decrypted = seal.unseal(
            result.salt, result.nonce, result.ciphertext, result.tag
        )
        assert decrypted == b""

    def test_large_plaintext(self):
        """Encrypt large plaintext (1MB)."""
        seal = SpiralSeal(master_password=b"large_test")
        plaintext = os.urandom(1024 * 1024)  # 1MB

        result = seal.seal(plaintext)
        decrypted = seal.unseal(
            result.salt, result.nonce, result.ciphertext, result.tag
        )
        assert decrypted == plaintext

    def test_no_password_or_key_raises(self):
        """Must provide password or key."""
        with pytest.raises(ValueError, match="Must provide"):
            SpiralSeal()

    def test_custom_key_id(self):
        """Custom key ID is preserved."""
        key_id = b"mykey123"
        seal = SpiralSeal(master_password=b"test", key_id=key_id)
        assert seal.key_id == key_id

        result = seal.seal(b"data")
        assert result.key_id == key_id

    def test_deterministic_with_same_inputs(self):
        """Same salt/nonce produces same ciphertext."""
        seal = SpiralSeal(master_password=b"deterministic")
        salt = b"fixed_salt_here!"
        nonce = os.urandom(24)  # XChaCha uses 24-byte nonce

        result1 = seal.seal(b"data", salt=salt, nonce=nonce)
        result2 = seal.seal(b"data", salt=salt, nonce=nonce)

        assert result1.ciphertext == result2.ciphertext
        assert result1.tag == result2.tag


class TestTokenSpecificBehavior:
    """Test Sacred Tongue token-specific behavior."""

    def test_koraelin_for_nonce(self):
        """Nonce uses Kor'aelin (flow/intent)."""
        seal = SpiralSeal(master_password=b"tongue_test")
        result = seal.seal(b"test")

        # Verify Kor'aelin prefixes are used
        prefixes, _ = TONGUE_WORDLISTS[SacredTongue.KORAELIN]
        nonce_text = result.nonce_tokens

        # At least one known prefix should appear
        assert any(p in nonce_text for p in prefixes)

    def test_runethic_for_salt(self):
        """Salt uses Runethic (binding/constraints)."""
        seal = SpiralSeal(master_password=b"tongue_test")
        result = seal.seal(b"test")

        prefixes, _ = TONGUE_WORDLISTS[SacredTongue.RUNETHIC]
        salt_text = result.salt_tokens
        assert any(p in salt_text for p in prefixes)

    def test_cassisivadan_for_ciphertext(self):
        """Ciphertext uses Cassisivadan (bitcraft)."""
        seal = SpiralSeal(master_password=b"tongue_test")
        result = seal.seal(b"test")

        prefixes, _ = TONGUE_WORDLISTS[SacredTongue.CASSISIVADAN]
        ct_text = result.ct_tokens
        assert any(p in ct_text for p in prefixes)

    def test_draumric_for_tag(self):
        """Auth tag uses Draumric (structure/forge)."""
        seal = SpiralSeal(master_password=b"tongue_test")
        result = seal.seal(b"test")

        prefixes, _ = TONGUE_WORDLISTS[SacredTongue.DRAUMRIC]
        tag_text = result.tag_tokens
        assert any(p in tag_text for p in prefixes)

    def test_avali_for_aad(self):
        """AAD uses Avali (diplomacy/context)."""
        seal = SpiralSeal(master_password=b"tongue_test")
        result = seal.seal(b"test", aad=b"context_data")

        prefixes, _ = TONGUE_WORDLISTS[SacredTongue.AVALI]
        aad_text = result.aad_tokens
        assert any(p in aad_text for p in prefixes)


# =============================================================================
# INTEGRATION WITH PQC MODULE TESTS
# =============================================================================

class TestPQCIntegration:
    """Test integration with PQC module."""

    def test_pqc_available_check(self):
        """Check PQC availability detection."""
        seal = PQCSpiralSeal(master_key=os.urandom(32))
        # Should have pqc_available property
        assert hasattr(seal, "pqc_available")

    def test_full_pqc_workflow(self):
        """Full PQC workflow: Kyber key exchange + Dilithium signing."""
        try:
            from symphonic_cipher.scbe_aethermoore.pqc import Kyber768, Dilithium3

            # Alice generates signing keys
            alice_sig = Dilithium3.generate_keypair()

            # Bob generates encryption keys
            bob_kem = Kyber768.generate_keypair()

            # Alice creates seal for Bob
            seal = PQCSpiralSeal(
                recipient_public_key=bob_kem.public_key,
                signing_secret_key=alice_sig.secret_key
            )

            # Encrypt and sign
            plaintext = b"Hello Bob, from Alice!"
            result, signature = seal.seal_signed(plaintext)

            # Bob decapsulates to get shared secret
            encap_ct = seal.encapsulated_ciphertext
            if encap_ct:
                shared_secret = Kyber768.decapsulate(bob_kem.secret_key, encap_ct)

                # Bob creates seal with shared secret
                bob_seal = SpiralSeal(master_key=shared_secret)

                # Decrypt
                decrypted = bob_seal.unseal(
                    result.salt, result.nonce, result.ciphertext, result.tag
                )
                assert decrypted == plaintext

                # Verify Alice's signature
                sign_data = (
                    result.key_id +
                    result.salt +
                    result.nonce +
                    result.ciphertext +
                    result.tag
                )
                assert Dilithium3.verify(alice_sig.public_key, sign_data, signature)

        except ImportError:
            pytest.skip("PQC module not available")


# =============================================================================
# STAGGERED AUTH TESTS
# =============================================================================

class TestSpiralKeyDerivation:
    """Test Spiral Key Derivation (SKD) system."""

    def test_derive_tongue_key_unique_per_tongue(self):
        """Each tongue should produce a unique key from same master."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            derive_tongue_key, TONGUE_SPIRAL_ORDER
        )

        master_key = os.urandom(32)
        keys = {}

        for tongue in TONGUE_SPIRAL_ORDER:
            keys[tongue] = derive_tongue_key(master_key, tongue)

        # All keys should be unique
        key_set = set(keys.values())
        assert len(key_set) == 6, "All 6 tongue keys should be unique"

    def test_derive_tongue_key_deterministic(self):
        """Same inputs should produce same key."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import derive_tongue_key

        master_key = b"test_master_key_32_bytes_long!!!"
        key1 = derive_tongue_key(master_key, "ko")
        key2 = derive_tongue_key(master_key, "ko")

        assert key1 == key2

    def test_derive_spiral_key_set(self):
        """Derive complete key set for all tongues."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            derive_spiral_key_set, TONGUE_SPIRAL_ORDER
        )

        master_key = os.urandom(32)
        keys = derive_spiral_key_set(master_key)

        assert len(keys) == 6
        for tongue in TONGUE_SPIRAL_ORDER:
            assert tongue in keys
            assert len(keys[tongue]) == 32

    def test_spiral_key_combine(self):
        """Combine multiple tongue keys."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            derive_spiral_key_set, spiral_key_combine
        )

        master_key = os.urandom(32)
        keys = derive_spiral_key_set(master_key)

        # Combine triad
        combined = spiral_key_combine(keys, ['ko', 'ru', 'um'])
        assert len(combined) == 32

        # Different combinations should produce different keys
        combined2 = spiral_key_combine(keys, ['av', 'ca', 'dr'])
        assert combined != combined2


class TestRefsGrid:
    """Test 6×6 cross-reference grid."""

    def test_build_refs_grid_ring_pattern(self):
        """Ring pattern: each tongue refs self + next."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, RefsPattern, TONGUE_SPIRAL_ORDER
        )

        sections = {
            'ko': b"nonce data",
            'ru': b"salt data",
            'ca': b"ciphertext",
        }

        refs = build_refs_grid(sections, RefsPattern.RING)

        # Ring pattern: each tongue has 2 refs (self + next)
        for tongue in TONGUE_SPIRAL_ORDER:
            assert tongue in refs
            assert len(refs[tongue]) == 2

    def test_build_refs_grid_full_pattern(self):
        """Full pattern: each tongue refs all others."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, RefsPattern, TONGUE_SPIRAL_ORDER
        )

        sections = {'ko': b"test", 'ca': b"data"}
        refs = build_refs_grid(sections, RefsPattern.FULL)

        # Full pattern: 6×6 = 36 refs total
        total_refs = sum(len(refs[t]) for t in refs)
        assert total_refs == 36

    def test_verify_refs_grid_success(self):
        """Refs grid verification with matching data."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, verify_refs_grid, RefsPattern
        )

        sections = {'ko': b"nonce", 'ru': b"salt", 'ca': b"ct"}
        refs = build_refs_grid(sections, RefsPattern.RING)

        valid, failures = verify_refs_grid(sections, refs, RefsPattern.RING)
        assert valid is True
        assert len(failures) == 0

    def test_verify_refs_grid_failure(self):
        """Refs grid verification with tampered data."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, verify_refs_grid, RefsPattern
        )

        sections = {'ko': b"nonce", 'ru': b"salt"}
        refs = build_refs_grid(sections, RefsPattern.RING)

        # Tamper with data
        sections['ko'] = b"TAMPERED"

        valid, failures = verify_refs_grid(sections, refs, RefsPattern.RING)
        assert valid is False
        assert len(failures) > 0


class TestAuthSidecar:
    """Test authentication sidecar."""

    def test_auth_sidecar_compute_verify(self):
        """Compute and verify auth sidecar."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            AuthSidecar, AuthConfig, derive_spiral_key_set, build_refs_grid, RefsPattern
        )

        master_key = os.urandom(32)
        keys = derive_spiral_key_set(master_key, b"test")

        payload = b"test payload data"
        sections = {'ca': payload}
        refs = build_refs_grid(sections, RefsPattern.RING)

        config = AuthConfig(tongues=('ko', 'ru', 'um'), threshold=2)
        sidecar = AuthSidecar.compute(payload, keys, config, refs)

        # Verify
        valid, count, valid_tongues = sidecar.verify(payload, keys, refs)
        assert valid is True
        assert count == 3
        assert set(valid_tongues) == {'ko', 'ru', 'um'}

    def test_auth_sidecar_threshold(self):
        """Threshold verification with partial keys."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            AuthSidecar, AuthConfig, derive_spiral_key_set, build_refs_grid, RefsPattern
        )

        master_key = os.urandom(32)
        keys = derive_spiral_key_set(master_key, b"test")

        payload = b"secret"
        refs = build_refs_grid({'ca': payload}, RefsPattern.RING)

        config = AuthConfig(tongues=('ko', 'ru', 'um'), threshold=2)
        sidecar = AuthSidecar.compute(payload, keys, config, refs)

        # Corrupt one key
        bad_keys = keys.copy()
        bad_keys['ko'] = os.urandom(32)

        # Should still pass with 2/3
        valid, count, valid_tongues = sidecar.verify(payload, bad_keys, refs)
        assert valid is True
        assert count == 2
        assert 'ko' not in valid_tongues

    def test_auth_sidecar_to_tokens(self):
        """Render sidecar as Draumric tokens."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            AuthSidecar, AuthConfig, derive_spiral_key_set, build_refs_grid, RefsPattern
        )

        master_key = os.urandom(32)
        keys = derive_spiral_key_set(master_key)

        config = AuthConfig(tongues=('ko', 'ru', 'um'), threshold=2)
        refs = build_refs_grid({'ca': b"test"}, RefsPattern.RING)
        sidecar = AuthSidecar.compute(b"test", keys, config, refs)

        tokens = sidecar.to_tokens()
        assert "ko:" in tokens
        assert "ru:" in tokens
        assert "um:" in tokens
        assert "dr:" in tokens  # Draumric encoding


class TestStaggeredAuthPacket:
    """Test complete staggered auth packet."""

    def test_pack_verify_roundtrip(self):
        """Pack and verify staggered auth packet."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            StaggeredAuthPacket, SSConfig, AuthConfig, RefsPattern
        )

        master_key = os.urandom(32)
        sections = {'ca': b"ciphertext data", 'ko': b"nonce"}

        config = SSConfig(
            refs=True,
            refs_pattern=RefsPattern.RING,
            auth=AuthConfig(tongues=('ko', 'ru', 'um'), threshold=2)
        )

        packet = StaggeredAuthPacket.pack(sections, master_key, config)

        # Verify
        valid, details = packet.verify(master_key)
        assert valid is True
        assert details["stage1_lengths"] is True
        assert details["stage2_checksum"] is True
        assert details["stage3_refs"] is True
        assert details["stage3_auth"]["valid"] is True

    def test_pack_tamper_detection(self):
        """Detect tampering in staggered auth packet."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            StaggeredAuthPacket, SSConfig, AuthConfig, RefsPattern
        )

        master_key = os.urandom(32)
        sections = {'ca': b"original data"}

        config = SSConfig(
            refs=True,
            auth=AuthConfig(tongues=('ko', 'ru', 'um'), threshold=2)
        )

        packet = StaggeredAuthPacket.pack(sections, master_key, config)

        # Tamper with sections
        packet.sections['ca'] = b"TAMPERED"

        valid, details = packet.verify(master_key)
        assert valid is False

    def test_packet_to_tokens(self):
        """Render packet as spell-text."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            StaggeredAuthPacket, SSConfig, AuthConfig
        )

        master_key = os.urandom(32)
        sections = {'ca': b"test"}

        config = SSConfig(auth=AuthConfig())
        packet = StaggeredAuthPacket.pack(sections, master_key, config)

        tokens = packet.to_tokens()
        assert "[ca]" in tokens
        assert "[checksum]" in tokens
        assert "[auth]" in tokens

    def test_quick_staggered_pack_verify(self):
        """Quick pack and verify functions."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            quick_staggered_pack, quick_staggered_verify
        )

        master_key = os.urandom(32)
        data = b"Hello, Spiral World!"

        packet = quick_staggered_pack(data, master_key)
        assert quick_staggered_verify(packet, master_key) is True

        # Wrong key should fail
        assert quick_staggered_verify(packet, os.urandom(32)) is False


class TestRefsPatternVariations:
    """Test different refs pattern configurations."""

    def test_ring_pattern_refs_count(self):
        """Ring pattern: 6 tongues × 2 refs each = 12 total."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, RefsPattern
        )

        sections = {'ko': b"a", 'av': b"b", 'ru': b"c", 'ca': b"d", 'um': b"e", 'dr': b"f"}
        refs = build_refs_grid(sections, RefsPattern.RING)

        total = sum(len(refs[t]) for t in refs)
        assert total == 12

    def test_two_pattern_refs_count(self):
        """Two pattern: 6 tongues × 3 refs each = 18 total."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, RefsPattern
        )

        sections = {'ko': b"a", 'av': b"b", 'ru': b"c", 'ca': b"d", 'um': b"e", 'dr': b"f"}
        refs = build_refs_grid(sections, RefsPattern.TWO)

        total = sum(len(refs[t]) for t in refs)
        assert total == 18

    def test_full_pattern_refs_count(self):
        """Full pattern: 6 tongues × 6 refs each = 36 total."""
        from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
            build_refs_grid, RefsPattern
        )

        sections = {'ko': b"a"}
        refs = build_refs_grid(sections, RefsPattern.FULL)

        total = sum(len(refs[t]) for t in refs)
        assert total == 36


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
