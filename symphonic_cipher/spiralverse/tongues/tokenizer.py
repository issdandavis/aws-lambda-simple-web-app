"""
Sacred Tongue Tokenizer - Compatibility Module
==============================================
This module provides compatibility with Grok's demo script imports.
All functionality is re-exported from the canonical sacred_tongues.py.
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
)

__all__ = [
    "SacredTongueTokenizer",
    "SacredTongue",
    "Token",
    "TongueSpec",
    "encode_to_spelltext",
    "decode_from_spelltext",
    "format_ss1_blob",
    "parse_ss1_blob",
    "encode_tokens_only",
    "decode_tokens_only",
    "get_tokenizer",
    "TONGUES",
    "SECTION_TONGUES",
]
