"""
Six Sacred Tongues - Semantic Encoding System

The Tongues classify data into ontological categories, each with:
- Domain-specific cryptographic keys
- Security levels (Kyber-512/768/1024)
- Multi-signature consensus rules (Roundtable)

Tongue  | Domain          | Role                | Security
--------|-----------------|---------------------|----------
KO      | Light/Logic     | Conductor (Init)    | Level 1
AV      | Air/Abstract    | Courier (Transit)   | Level 2
RU      | Earth/Organic   | Warden (Validate)   | Level 1
CA      | Fire/Emotional  | Engine (Compute)    | Level 3
UM      | Cosmos/Wisdom   | Vault (Encrypt)     | Level 2
DR      | Water/Hidden    | Architect (Schema)  | Level 3
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import hashlib
import hmac
import time
import json
import base64


# ═══════════════════════════════════════════════════════════════
# Tongue Definitions
# ═══════════════════════════════════════════════════════════════

class Tongue(Enum):
    """The Six Sacred Tongues."""
    KO = "KO"  # Kor'aelin - Light/Logic - Conductor
    AV = "AV"  # Avali - Air/Abstract - Courier
    RU = "RU"  # Runethic - Earth/Organic - Warden
    CA = "CA"  # Cassisivadan - Fire/Emotional - Engine
    UM = "UM"  # Umbroth - Cosmos/Wisdom - Vault
    DR = "DR"  # Draumric - Water/Hidden - Architect


class TongueRole(Enum):
    """Functional roles of each Tongue."""
    CONDUCTOR = auto()   # KO - Initiates actions
    COURIER = auto()     # AV - Transports messages
    WARDEN = auto()      # RU - Validates policy
    ENGINE = auto()      # CA - Computes outputs
    VAULT = auto()       # UM - Encrypts secrets
    ARCHITECT = auto()   # DR - Defines schemas


class SecurityLevel(Enum):
    """Kyber security levels."""
    LEVEL_1 = 1  # Kyber-512 (128-bit)
    LEVEL_2 = 2  # Kyber-768 (192-bit)
    LEVEL_3 = 3  # Kyber-1024 (256-bit)


@dataclass
class TongueSpec:
    """Specification for a Sacred Tongue."""
    code: Tongue
    name: str
    domain: str
    element: str
    role: TongueRole
    security_level: SecurityLevel
    symbols: Tuple[str, ...]

    @property
    def kyber_variant(self) -> str:
        """Get Kyber variant name."""
        return {
            SecurityLevel.LEVEL_1: "Kyber-512",
            SecurityLevel.LEVEL_2: "Kyber-768",
            SecurityLevel.LEVEL_3: "Kyber-1024",
        }[self.security_level]


# Tongue Registry
TONGUES: Dict[Tongue, TongueSpec] = {
    Tongue.KO: TongueSpec(
        code=Tongue.KO,
        name="Kor'aelin",
        domain="Control & Orchestration",
        element="Light/Logic",
        role=TongueRole.CONDUCTOR,
        security_level=SecurityLevel.LEVEL_1,
        symbols=("◇", "◆", "◈", "⬖"),
    ),
    Tongue.AV: TongueSpec(
        code=Tongue.AV,
        name="Avali",
        domain="I/O & Messaging",
        element="Air/Abstract",
        role=TongueRole.COURIER,
        security_level=SecurityLevel.LEVEL_2,
        symbols=("◎", "◉", "○", "●"),
    ),
    Tongue.RU: TongueSpec(
        code=Tongue.RU,
        name="Runethic",
        domain="Policy & Constraints",
        element="Earth/Organic",
        role=TongueRole.WARDEN,
        security_level=SecurityLevel.LEVEL_1,
        symbols=("▲", "▽", "◄", "►"),
    ),
    Tongue.CA: TongueSpec(
        code=Tongue.CA,
        name="Cassisivadan",
        domain="Logic & Computation",
        element="Fire/Emotional",
        role=TongueRole.ENGINE,
        security_level=SecurityLevel.LEVEL_3,
        symbols=("★", "☆", "✦", "✧"),
    ),
    Tongue.UM: TongueSpec(
        code=Tongue.UM,
        name="Umbroth",
        domain="Security & Privacy",
        element="Cosmos/Wisdom",
        role=TongueRole.VAULT,
        security_level=SecurityLevel.LEVEL_2,
        symbols=("✴", "✵", "✶", "✷"),
    ),
    Tongue.DR: TongueSpec(
        code=Tongue.DR,
        name="Draumric",
        domain="Types & Structures",
        element="Water/Hidden",
        role=TongueRole.ARCHITECT,
        security_level=SecurityLevel.LEVEL_3,
        symbols=("◈", "◊", "⬥", "⬦"),
    ),
}


# ═══════════════════════════════════════════════════════════════
# Key Derivation
# ═══════════════════════════════════════════════════════════════

def derive_tongue_key(
    master_key: bytes,
    tongue: Tongue,
    context: str = "",
) -> bytes:
    """
    Derive a domain-separated sub-key for a Tongue.

    K_tongue = HMAC-SHA256(K_master, "spiralverse" || TongueCode || Context)
    """
    data = b"spiralverse" + tongue.value.encode() + context.encode()
    return hmac.new(master_key, data, hashlib.sha256).digest()


def derive_composite_key(
    master_key: bytes,
    tongues: List[Tongue],
    context: str = "",
) -> bytes:
    """
    Derive a composite key from multiple Tongues.

    Used for Roundtable consensus requiring multiple signatures.
    """
    # Sort tongues for deterministic ordering
    sorted_tongues = sorted(tongues, key=lambda t: t.value)

    # Chain derivations
    current_key = master_key
    for tongue in sorted_tongues:
        current_key = derive_tongue_key(current_key, tongue, context)

    return current_key


# ═══════════════════════════════════════════════════════════════
# RWP v2 Envelope
# ═══════════════════════════════════════════════════════════════

class Phase(Enum):
    """Processing phases in RWP."""
    SCHEMA = "schema"
    FRACTAL = "fractal"
    INTENT = "intent"
    TRAJECTORY = "trajectory"
    PHASE = "phase"
    NEURAL = "neural"
    SWARM = "swarm"
    CRYPTO = "crypto"


@dataclass
class RWPEnvelope:
    """
    Rosetta Weave Protocol v2 Envelope.

    Hybrid envelope encapsulating semantic intent in a secure wrapper.
    """
    # Control Plane
    ver: str = "2.1"
    tongue: Tongue = Tongue.KO
    origin: str = ""

    # Temporal Binding
    ts: float = field(default_factory=time.time)
    seq: int = 0

    # Processing Phase
    phase: Phase = Phase.INTENT

    # Additional Authenticated Data
    aad: Dict[str, str] = field(default_factory=dict)

    # Data Plane
    payload: bytes = b""

    # Security Layer
    enc: str = "aes-256-gcm"
    kid: str = "rwp2:keyring:v1"
    nonce: bytes = field(default_factory=lambda: b"\x00" * 12)

    # Signatures (for Roundtable consensus)
    signatures: Dict[str, bytes] = field(default_factory=dict)

    def canonical_aad(self) -> bytes:
        """Get canonical AAD for signing."""
        aad_str = ";".join(f"{k}={v}" for k, v in sorted(self.aad.items()))
        return f"context={aad_str};ts={self.ts};seq={self.seq}".encode()

    def sign(self, tongue: Tongue, key: bytes) -> None:
        """Add a Tongue signature."""
        data = self.canonical_aad() + self.payload
        sig = hmac.new(key, data, hashlib.sha256).digest()
        self.signatures[tongue.value] = sig

    def verify(self, tongue: Tongue, key: bytes) -> bool:
        """Verify a Tongue signature."""
        if tongue.value not in self.signatures:
            return False

        data = self.canonical_aad() + self.payload
        expected = hmac.new(key, data, hashlib.sha256).digest()
        return hmac.compare_digest(self.signatures[tongue.value], expected)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "ver": self.ver,
            "tongue": self.tongue.value,
            "origin": self.origin,
            "ts": self.ts,
            "seq": self.seq,
            "phase": self.phase.value,
            "aad": self.aad,
            "payload": base64.urlsafe_b64encode(self.payload).decode(),
            "enc": self.enc,
            "kid": self.kid,
            "nonce": base64.urlsafe_b64encode(self.nonce).decode(),
            "sigs": {
                k: base64.urlsafe_b64encode(v).decode()
                for k, v in self.signatures.items()
            },
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> "RWPEnvelope":
        """Deserialize from dictionary."""
        return cls(
            ver=data.get("ver", "2.1"),
            tongue=Tongue(data["tongue"]),
            origin=data.get("origin", ""),
            ts=data.get("ts", time.time()),
            seq=data.get("seq", 0),
            phase=Phase(data.get("phase", "intent")),
            aad=data.get("aad", {}),
            payload=base64.urlsafe_b64decode(data.get("payload", "")),
            enc=data.get("enc", "aes-256-gcm"),
            kid=data.get("kid", "rwp2:keyring:v1"),
            nonce=base64.urlsafe_b64decode(data.get("nonce", "AAAAAAAAAAAAAAAA")),
            signatures={
                k: base64.urlsafe_b64decode(v)
                for k, v in data.get("sigs", {}).items()
            },
        )

    @classmethod
    def from_json(cls, json_str: str) -> "RWPEnvelope":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


# ═══════════════════════════════════════════════════════════════
# Roundtable Consensus
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoundtableRule:
    """A rule requiring multiple Tongue signatures."""
    name: str
    required_tongues: Set[Tongue]
    description: str


# Standard Roundtable rules
ROUNDTABLE_RULES: Dict[str, RoundtableRule] = {
    "critical_action": RoundtableRule(
        name="critical_action",
        required_tongues={Tongue.KO, Tongue.RU, Tongue.UM},
        description="KO initiates, RU validates policy, UM authenticates",
    ),
    "data_export": RoundtableRule(
        name="data_export",
        required_tongues={Tongue.RU, Tongue.UM, Tongue.DR},
        description="RU validates, UM encrypts, DR verifies schema",
    ),
    "schema_change": RoundtableRule(
        name="schema_change",
        required_tongues={Tongue.DR, Tongue.RU},
        description="DR architects, RU validates",
    ),
    "compute_task": RoundtableRule(
        name="compute_task",
        required_tongues={Tongue.KO, Tongue.CA},
        description="KO initiates, CA computes",
    ),
}


class Roundtable:
    """
    Multi-signature consensus engine.

    Ensures critical actions require signatures from multiple Tongues,
    preventing hallucinations and unauthorized actions.
    """

    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self._tongue_keys: Dict[Tongue, bytes] = {}

    def get_tongue_key(self, tongue: Tongue, context: str = "") -> bytes:
        """Get or derive a Tongue key."""
        cache_key = (tongue, context)
        if cache_key not in self._tongue_keys:
            self._tongue_keys[cache_key] = derive_tongue_key(
                self.master_key, tongue, context
            )
        return self._tongue_keys[cache_key]

    def sign_envelope(
        self,
        envelope: RWPEnvelope,
        tongues: List[Tongue],
        context: str = "",
    ) -> None:
        """Sign an envelope with multiple Tongues."""
        for tongue in tongues:
            key = self.get_tongue_key(tongue, context)
            envelope.sign(tongue, key)

    def verify_envelope(
        self,
        envelope: RWPEnvelope,
        rule_name: str,
        context: str = "",
    ) -> Tuple[bool, List[Tongue]]:
        """
        Verify an envelope against a Roundtable rule.

        Returns (valid, missing_tongues).
        """
        if rule_name not in ROUNDTABLE_RULES:
            raise ValueError(f"Unknown rule: {rule_name}")

        rule = ROUNDTABLE_RULES[rule_name]
        missing = []

        for tongue in rule.required_tongues:
            key = self.get_tongue_key(tongue, context)
            if not envelope.verify(tongue, key):
                missing.append(tongue)

        return len(missing) == 0, missing

    def create_signed_envelope(
        self,
        payload: bytes,
        primary_tongue: Tongue,
        rule_name: str,
        origin: str = "",
        context: str = "",
        aad: Optional[Dict[str, str]] = None,
    ) -> RWPEnvelope:
        """Create and sign an envelope according to a rule."""
        rule = ROUNDTABLE_RULES[rule_name]

        envelope = RWPEnvelope(
            tongue=primary_tongue,
            origin=origin,
            payload=payload,
            aad=aad or {},
        )

        # Sign with all required tongues
        self.sign_envelope(envelope, list(rule.required_tongues), context)

        return envelope


# ═══════════════════════════════════════════════════════════════
# Semantic Router
# ═══════════════════════════════════════════════════════════════

class SemanticRouter:
    """
    Routes messages based on Tongue classification.

    Determines which processing pipeline handles a message
    based on its semantic category.
    """

    def __init__(self):
        self.handlers: Dict[Tongue, callable] = {}

    def register(self, tongue: Tongue, handler: callable) -> None:
        """Register a handler for a Tongue."""
        self.handlers[tongue] = handler

    def route(self, envelope: RWPEnvelope) -> Optional[any]:
        """Route an envelope to its handler."""
        if envelope.tongue in self.handlers:
            return self.handlers[envelope.tongue](envelope)
        return None

    def classify_intent(self, text: str) -> Tongue:
        """
        Classify natural language intent to a Tongue.

        Simple keyword-based classification (production would use ML).
        """
        text_lower = text.lower()

        # Control/orchestration keywords -> KO
        if any(kw in text_lower for kw in ["start", "stop", "init", "orchestrate", "control"]):
            return Tongue.KO

        # Messaging keywords -> AV
        if any(kw in text_lower for kw in ["send", "receive", "message", "notify", "broadcast"]):
            return Tongue.AV

        # Policy keywords -> RU
        if any(kw in text_lower for kw in ["policy", "rule", "validate", "constraint", "require"]):
            return Tongue.RU

        # Computation keywords -> CA
        if any(kw in text_lower for kw in ["compute", "calculate", "process", "execute", "run"]):
            return Tongue.CA

        # Security keywords -> UM
        if any(kw in text_lower for kw in ["encrypt", "decrypt", "secure", "secret", "private"]):
            return Tongue.UM

        # Schema keywords -> DR
        if any(kw in text_lower for kw in ["schema", "type", "structure", "define", "model"]):
            return Tongue.DR

        # Default to KO (conductor)
        return Tongue.KO
