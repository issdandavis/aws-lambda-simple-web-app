"""
Sacred Tongue Tokenizer

Each Tongue encodes bytes 0x00-0xFF using 16 prefixes × 16 suffixes = 256 unique tokens.
Token format: prefix'suffix (e.g., "sil'vara", "thal'kor")

Tongues:
    KO (Kor'aelin)     - Nonce encoding
    AV (Avali)         - Reserved
    RU (Runethic)      - Salt encoding
    CA (Cassisivadan)  - Ciphertext encoding
    UM (Umbroth)       - Reserved
    DR (Draumric)      - Tag encoding
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TongueSpec:
    """Specification for a Sacred Tongue's vocabulary."""
    code: str
    name: str
    prefixes: Tuple[str, ...]
    suffixes: Tuple[str, ...]

    def __post_init__(self):
        assert len(self.prefixes) == 16, f"{self.code} needs 16 prefixes"
        assert len(self.suffixes) == 16, f"{self.code} needs 16 suffixes"


# ═══════════════════════════════════════════════════════════════
# Tongue Vocabularies (16 prefixes × 16 suffixes = 256 tokens each)
# ═══════════════════════════════════════════════════════════════

TONGUE_SPECS: Dict[str, TongueSpec] = {
    "KO": TongueSpec(
        code="KO",
        name="Kor'aelin",
        prefixes=(
            "sil", "vel", "kor", "thal", "meth", "drav", "quel", "zar",
            "fel", "mor", "var", "neth", "sol", "kren", "val", "vara",
        ),
        suffixes=(
            "a", "el", "or", "an", "eth", "ir", "on", "ul",
            "as", "en", "is", "um", "ar", "oth", "in", "esh",
        ),
    ),
    "AV": TongueSpec(
        code="AV",
        name="Avali",
        prefixes=(
            "lum", "aer", "ven", "cir", "plu", "zeph", "nub", "alt",
            "vox", "ton", "har", "mel", "rhy", "can", "son", "ech",
        ),
        suffixes=(
            "a", "is", "os", "us", "ae", "ix", "ox", "ax",
            "em", "um", "am", "im", "en", "on", "an", "yn",
        ),
    ),
    "RU": TongueSpec(
        code="RU",
        name="Runethic",
        prefixes=(
            "thal", "kreth", "vor", "mund", "gral", "steen", "holm", "berg",
            "wald", "fen", "moor", "dal", "glen", "crag", "tor", "fell",
        ),
        suffixes=(
            "or", "an", "eth", "orn", "ald", "ung", "art", "heim",
            "gard", "vik", "fjord", "dal", "ness", "wick", "by", "thorp",
        ),
    ),
    "CA": TongueSpec(
        code="CA",
        name="Cassisivadan",
        prefixes=(
            "drev", "asha", "kelth", "von", "pyr", "ign", "flam", "ard",
            "cal", "fer", "braz", "emb", "cind", "sear", "blaz", "scor",
        ),
        suffixes=(
            "a", "or", "ix", "on", "us", "al", "ar", "en",
            "is", "um", "eth", "ian", "ius", "eon", "ax", "ox",
        ),
    ),
    "UM": TongueSpec(
        code="UM",
        name="Umbroth",
        prefixes=(
            "nyx", "vex", "shad", "murk", "void", "nul", "aeth", "cryp",
            "obs", "tene", "gloom", "dusk", "twi", "dim", "phan", "wraith",
        ),
        suffixes=(
            "os", "is", "ax", "ex", "ix", "ox", "ux", "yx",
            "al", "el", "il", "ol", "ul", "ar", "or", "ur",
        ),
    ),
    "DR": TongueSpec(
        code="DR",
        name="Draumric",
        prefixes=(
            "mor", "vex", "uth", "dral", "xen", "cryth", "zol", "qar",
            "prax", "syn", "arch", "meta", "hyper", "proto", "neo", "omni",
        ),
        suffixes=(
            "thal", "kor", "ven", "dex", "plex", "form", "morph", "gen",
            "type", "struct", "schema", "graph", "node", "link", "web", "net",
        ),
    ),
}


# Section to tongue mapping
SECTION_TONGUES = {
    "salt": "RU",
    "nonce": "KO",
    "ct": "CA",
    "ciphertext": "CA",
    "tag": "DR",
}


class SacredTongueTokenizer:
    """
    Encodes/decodes bytes to Sacred Tongue spell-text.

    Each byte maps to a unique token: prefix'suffix
    - High nibble (0-15) selects prefix
    - Low nibble (0-15) selects suffix
    """

    def __init__(self, tongue_code: str):
        if tongue_code not in TONGUE_SPECS:
            raise ValueError(f"Unknown tongue: {tongue_code}")

        self.spec = TONGUE_SPECS[tongue_code]
        self.prefixes = self.spec.prefixes
        self.suffixes = self.spec.suffixes

        # Build reverse lookup tables
        self._prefix_to_idx = {p: i for i, p in enumerate(self.prefixes)}
        self._suffix_to_idx = {s: i for i, s in enumerate(self.suffixes)}

    def encode_byte(self, byte: int) -> str:
        """Encode single byte to token: prefix'suffix"""
        if not 0 <= byte <= 255:
            raise ValueError(f"Byte out of range: {byte}")

        prefix_idx = byte >> 4       # High nibble
        suffix_idx = byte & 0x0F     # Low nibble
        return f"{self.prefixes[prefix_idx]}'{self.suffixes[suffix_idx]}"

    def decode_token(self, token: str) -> int:
        """Decode token back to byte."""
        if "'" not in token:
            raise ValueError(f"Invalid token format: {token}")

        prefix, suffix = token.split("'", 1)

        if prefix not in self._prefix_to_idx:
            raise ValueError(f"Unknown prefix: {prefix}")
        if suffix not in self._suffix_to_idx:
            raise ValueError(f"Unknown suffix: {suffix}")

        prefix_idx = self._prefix_to_idx[prefix]
        suffix_idx = self._suffix_to_idx[suffix]
        return (prefix_idx << 4) | suffix_idx

    def encode_bytes(self, data: bytes) -> str:
        """Encode bytes to space-separated spell-text."""
        tokens = [self.encode_byte(b) for b in data]
        return " ".join(tokens)

    def decode_bytes(self, spelltext: str) -> bytes:
        """Decode spell-text back to bytes."""
        if not spelltext.strip():
            return b""

        tokens = spelltext.strip().split()
        return bytes(self.decode_token(t) for t in tokens)


def encode_to_spelltext(data: bytes, section: str) -> str:
    """
    Encode bytes to spell-text with tongue prefix.

    Args:
        data: Binary data to encode
        section: Section name (salt, nonce, ct, tag)

    Returns:
        String like "ru:thal'or kreth'an vor'eth"
    """
    tongue_code = SECTION_TONGUES.get(section, "CA")
    tokenizer = SacredTongueTokenizer(tongue_code)
    spelltext = tokenizer.encode_bytes(data)
    return f"{tongue_code.lower()}:{spelltext}"


def decode_from_spelltext(encoded: str) -> Tuple[str, bytes]:
    """
    Decode spell-text with tongue prefix.

    Args:
        encoded: String like "ru:thal'or kreth'an"

    Returns:
        Tuple of (tongue_code, decoded_bytes)
    """
    if ":" not in encoded:
        raise ValueError(f"Missing tongue prefix: {encoded}")

    tongue_code, spelltext = encoded.split(":", 1)
    tongue_code = tongue_code.upper()

    tokenizer = SacredTongueTokenizer(tongue_code)
    data = tokenizer.decode_bytes(spelltext)
    return tongue_code, data
