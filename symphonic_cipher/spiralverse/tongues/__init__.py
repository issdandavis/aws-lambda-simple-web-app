"""
Sacred Tongues - Cryptolinguistic Encoding
==========================================
Re-exports from the canonical implementation.
"""

from ...scbe_aethermoore.spiral_seal.sacred_tongues import (
    SacredTongueTokenizer,
    SacredTongue,
    Token,
    TongueSpec,
    encode_to_spelltext,
    decode_from_spelltext,
    format_ss1_blob,
    parse_ss1_blob,
    encode_tokens_only,
    decode_tokens_only,
    get_tokenizer,
    TONGUES,
    SECTION_TONGUES,
    KOR_AELIN,
    AVALI,
    RUNETHIC,
    CASSISIVADAN,
    UMBROTH,
    DRAUMRIC,
)

# Alias for Grok's demo compatibility
tokenizer = SacredTongueTokenizer

__all__ = [
    "SacredTongueTokenizer",
    "SacredTongue",
    "Token",
    "TongueSpec",
    "tokenizer",
    "encode_to_spelltext",
    "decode_from_spelltext",
    "format_ss1_blob",
    "parse_ss1_blob",
    "encode_tokens_only",
    "decode_tokens_only",
    "get_tokenizer",
    "TONGUES",
    "SECTION_TONGUES",
    "KOR_AELIN",
    "AVALI",
    "RUNETHIC",
    "CASSISIVADAN",
    "UMBROTH",
    "DRAUMRIC",
]
