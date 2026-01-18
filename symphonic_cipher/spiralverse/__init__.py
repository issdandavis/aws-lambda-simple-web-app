"""
Spiralverse Protocol
====================
Semantic cryptography with Six Sacred Tongues for cryptolinguistic encoding.
"""

from ..scbe_aethermoore.spiral_seal.sacred_tongues import (
    SacredTongueTokenizer,
    SacredTongue,
    Token,
    encode_to_spelltext,
    decode_from_spelltext,
    format_ss1_blob,
    parse_ss1_blob,
    TONGUES,
    SECTION_TONGUES,
)

__all__ = [
    "SacredTongueTokenizer",
    "SacredTongue",
    "Token",
    "encode_to_spelltext",
    "decode_from_spelltext",
    "format_ss1_blob",
    "parse_ss1_blob",
    "TONGUES",
    "SECTION_TONGUES",
]
