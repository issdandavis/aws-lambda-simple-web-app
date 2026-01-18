"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  AXIOM 7: SPIRAL SEAL (SACRED TONGUES)                       ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Sacred Tongues Tokenizer with Staggered Authentication                      ║
║                                                                              ║
║  The Six Sacred Tongues (6×256 = 1,536 total tokens):                        ║
║    KO - Koraelin    (nonce/randomness)                                       ║
║    AV - Avali       (additional authenticated data)                          ║
║    RU - Runethic    (salt/key derivation)                                    ║
║    CA - Cassisivadan (ciphertext)                                            ║
║    UM - Umbraic     (metadata/headers)                                       ║
║    DR - Draumric    (authentication tags)                                    ║
║                                                                              ║
║  Staggered Auth: Three-stage verification                                    ║
║    Stage 1: Length checksums                                                 ║
║    Stage 2: Cross-reference grid (RING/TWO/FULL patterns)                    ║
║    Stage 3: Triad authentication (threshold verification)                    ║
║                                                                              ║
║  Spiral Key Derivation:                                                      ║
║    derive_tongue_key(master, tongue) = HKDF(master, "SPIRAL:" + tongue)      ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 7 (Spiral Seal)                                                    ║
║  Author: Issac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from spiral_seal modules
from ..spiral_seal.sacred_tongues import (
    # Tongue Classes
    SacredTongue,
    SacredTongueTokenizer,
    SacredTongueTokenizerCompat,
    TongueSpec,
    TongueInfo,
    Token,
    # Core Constants
    TONGUES,
    TONGUE_SPIRAL_ORDER,
    TONGUE_WORDLISTS,
    SPIRALSCRIPT_KEYWORDS,
    SECTION_TONGUES,
    DOMAIN_TONGUE_MAP,
    # Individual Tongues
    KOR_AELIN,
    AVALI,
    RUNETHIC,
    CASSISIVADAN,
    UMBROTH,
    DRAUMRIC,
    # Token Functions
    get_tokenizer,
    get_tongue_for_domain,
    get_tongue_keywords,
    get_tongue_signature,
    get_combined_alphabet,
    get_magical_signature,
    # Spiral Key Derivation
    derive_tongue_key,
    derive_spiral_key_set,
    spiral_key_combine,
    # Staggered Auth
    RefsPattern, AuthConfig, SSConfig,
    build_refs_grid, verify_refs_grid,
    AuthSidecar, StaggeredAuthPacket,
    quick_staggered_pack, quick_staggered_verify,
    # LWS
    compute_lws_weights,
    # Serialization
    encode_to_spelltext,
    decode_from_spelltext,
    encode_tokens_only,
    decode_tokens_only,
    format_ss1_blob,
    parse_ss1_blob,
)

from ..spiral_seal.spiral_seal import (
    # Core Classes
    SpiralSeal,
    SpiralSealSS1,
    SpiralSealResult,
    VeiledSeal,
    VeiledSealResult,
    PQCSpiralSeal,
    # Functions
    quick_seal,
    quick_unseal,
    derive_key,
    get_crypto_backend_info,
)

__all__ = [
    # Tongue Classes
    'SacredTongue', 'SacredTongueTokenizer', 'SacredTongueTokenizerCompat',
    'TongueSpec', 'TongueInfo', 'Token',
    # Constants
    'TONGUES', 'TONGUE_SPIRAL_ORDER', 'TONGUE_WORDLISTS',
    'SPIRALSCRIPT_KEYWORDS', 'SECTION_TONGUES', 'DOMAIN_TONGUE_MAP',
    # Individual Tongues
    'KOR_AELIN', 'AVALI', 'RUNETHIC', 'CASSISIVADAN', 'UMBROTH', 'DRAUMRIC',
    # Token Functions
    'get_tokenizer', 'get_tongue_for_domain', 'get_tongue_keywords',
    'get_tongue_signature', 'get_combined_alphabet', 'get_magical_signature',
    # Spiral Key Derivation
    'derive_tongue_key', 'derive_spiral_key_set', 'spiral_key_combine',
    # Staggered Auth
    'RefsPattern', 'AuthConfig', 'SSConfig',
    'build_refs_grid', 'verify_refs_grid',
    'AuthSidecar', 'StaggeredAuthPacket',
    'quick_staggered_pack', 'quick_staggered_verify',
    # LWS
    'compute_lws_weights',
    # Serialization
    'encode_to_spelltext', 'decode_from_spelltext',
    'encode_tokens_only', 'decode_tokens_only',
    'format_ss1_blob', 'parse_ss1_blob',
    # Spiral Seal
    'SpiralSeal', 'SpiralSealSS1', 'SpiralSealResult',
    'VeiledSeal', 'VeiledSealResult', 'PQCSpiralSeal',
    'quick_seal', 'quick_unseal', 'derive_key', 'get_crypto_backend_info',
]

AXIOM_ID = "7"
AXIOM_TITLE = "Spiral Seal (Sacred Tongues Tokenizer)"
AXIOM_FORMULA = "6 tongues × 256 tokens = 1,536 symbolic vocabulary"
