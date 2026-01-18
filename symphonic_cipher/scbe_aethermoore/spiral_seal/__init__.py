"""
SpiralSeal SS1 - Sacred Tongue Cryptographic Encoding

Transforms binary ciphertext into Sacred Tongue spell-text,
making encrypted data look like fantasy language incantations.

Instead of:
    AES-GCM-256:MTIzNDU2Nzg5MGFiY2RlZg==:aGVsbG8gd29ybGQ=

You get:
    SS1|kid=k01|aad=service=prod|ru:thal'vor kreth'an|ko:sil'vara meth'el|ca:drev'asha|dr:mor'thal
"""

from .seal import seal, unseal, SpiralSealSS1
from .sacred_tongues import (
    SacredTongueTokenizer,
    TONGUE_SPECS,
    encode_to_spelltext,
    decode_from_spelltext,
)

__all__ = [
    "seal",
    "unseal",
    "SpiralSealSS1",
    "SacredTongueTokenizer",
    "TONGUE_SPECS",
    "encode_to_spelltext",
    "decode_from_spelltext",
]
