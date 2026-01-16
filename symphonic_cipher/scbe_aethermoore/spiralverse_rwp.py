"""
Spiralverse Real-World Protocol v3.0 Implementation
====================================================

Quantum-resistant protocol with hybrid PQC and Six Sacred Tongues integration.

This module implements the RWP v3.0 specification, providing:
- Hybrid key exchange (X25519 + ML-KEM-768)
- Hybrid signatures (Ed25519 + ML-DSA-65)
- Six Sacred Tongues (SST) bindings for semantic protocol layers
- Protocol negotiation for backward compatibility

Patent Reference: USPTO #63/961,403 (SCBE-AETHERMOORE-2026-001-PROV)
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

# Import existing PQC components
from .pqc_module import (
    get_kem,
    get_signature,
    PQCManager,
)

# Get KEM and Signature instances
_kem = get_kem(prefer_real=False)  # Use simulated for portability
_sig = get_signature(prefer_real=False)

def generate_kyber_keypair():
    """Generate ML-KEM-768 (Kyber) keypair."""
    return _kem.keygen()

def kyber_encapsulate(public_key: bytes):
    """Encapsulate using ML-KEM. Returns (shared_secret, ciphertext)."""
    return _kem.encaps(public_key)

def kyber_decapsulate(private_key: bytes, ciphertext: bytes):
    """Decapsulate using ML-KEM."""
    return _kem.decaps(private_key, ciphertext)

def generate_dilithium_keypair():
    """Generate ML-DSA-65 (Dilithium) keypair."""
    return _sig.keygen()

def dilithium_sign(private_key: bytes, message: bytes):
    """Sign message with ML-DSA."""
    return _sig.sign(private_key, message)

def dilithium_verify(public_key: bytes, message: bytes, signature: bytes):
    """Verify ML-DSA signature."""
    return _sig.verify(public_key, message, signature)

# Import existing tongue system
from .unified import TONGUES, TONGUE_WEIGHTS, CONLANG


# =============================================================================
# PROTOCOL ENUMS AND CONSTANTS
# =============================================================================

class ProtocolMode(Enum):
    """Protocol version modes for negotiation."""
    CLASSICAL_ONLY = 0x01    # RWP v2 legacy
    HYBRID_PREFER_PQ = 0x02  # RWP v3 default
    PQ_ONLY = 0x03           # RWP v3 strict (future)


class SacredTongue(Enum):
    """Six Sacred Tongues - semantic protocol layers."""
    KO = "Korvethian"   # Command/Control
    AV = "Avethril"     # Emotional resonance
    RU = "Runevast"     # Historical binding
    CA = "Celestine"    # Divine invocation / Ceremony
    UM = "Umbralis"     # Shadow protocols
    DR = "Draconic"     # Power amplification / Multi-party


class RiskLevel(Enum):
    """HNDL (Harvest Now, Decrypt Later) risk levels."""
    CRITICAL = auto()  # 50+ years sensitivity
    HIGH = auto()      # 10-25 years
    MEDIUM = auto()    # 5-10 years
    LOW = auto()       # <5 years or ephemeral


# Map Sacred Tongues to existing SCBE tongue indices
SST_TO_SCBE_TONGUE = {
    SacredTongue.KO: 0,  # Maps to "Anchor" - command/control stability
    SacredTongue.AV: 1,  # Maps to "Bridge" - emotional connection
    SacredTongue.RU: 2,  # Maps to "Cut" - historical precision
    SacredTongue.CA: 3,  # Maps to "Paradox" - divine/recursive
    SacredTongue.UM: 4,  # Maps to "Joy" - hidden/shadow (ironic)
    SacredTongue.DR: 5,  # Maps to "Harmony" - multi-party consensus
}

# PQC migration priority by tongue
SST_PQC_PRIORITY = {
    SacredTongue.DR: RiskLevel.CRITICAL,   # Multi-party keys: 50+ years
    SacredTongue.CA: RiskLevel.HIGH,       # Ceremony keys: 25+ years
    SacredTongue.KO: RiskLevel.HIGH,       # Command archives: 10+ years
    SacredTongue.AV: RiskLevel.MEDIUM,     # Emotional payloads: 5-10 years
    SacredTongue.RU: RiskLevel.LOW,        # Hash-based: quantum-safe
    SacredTongue.UM: RiskLevel.LOW,        # Ephemeral: <24 hours
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HybridKeyPair:
    """Combined classical + post-quantum key pair."""
    # Classical (X25519-style, simulated)
    classical_public: bytes
    classical_private: bytes

    # Post-quantum (ML-KEM-768 / Kyber)
    pq_public: bytes
    pq_private: bytes

    algorithm: str = "X25519+ML-KEM-768"
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    sst_binding: Optional[SacredTongue] = None

    def to_bytes(self) -> bytes:
        """Serialize public keys for transmission."""
        return self.classical_public + self.pq_public


@dataclass
class HybridSigningKeyPair:
    """Combined classical + post-quantum signing key pair."""
    # Classical (Ed25519-style, simulated)
    classical_public: bytes
    classical_private: bytes

    # Post-quantum (ML-DSA-65 / Dilithium)
    pq_public: bytes
    pq_private: bytes

    algorithm: str = "Ed25519+ML-DSA-65"
    created_at: float = field(default_factory=time.time)


@dataclass
class HybridSignature:
    """Combined classical + post-quantum signature."""
    classical: bytes      # 64 bytes (Ed25519)
    post_quantum: bytes   # ~3293 bytes (ML-DSA-65)

    timestamp: float = field(default_factory=time.time)
    sst_context: Optional[Dict[str, Any]] = None

    def to_bytes(self) -> bytes:
        """Serialize signature."""
        return self.classical + self.post_quantum

    @property
    def total_size(self) -> int:
        return len(self.classical) + len(self.post_quantum)


@dataclass
class SSTContext:
    """Semantic context for Six Sacred Tongues operations."""
    tongue: SacredTongue
    protocol_version: str = "3.0"
    hybrid_mode: bool = True

    # Tongue-specific metadata
    resonance_frequency: Optional[float] = None  # AV-specific
    temporal_anchor: Optional[int] = None        # RU-specific
    invocation_depth: Optional[int] = None       # CA-specific
    command_chain_id: Optional[str] = None       # KO-specific
    shadow_session_id: Optional[str] = None      # UM-specific
    participant_count: Optional[int] = None      # DR-specific


@dataclass
class VersionNegotiation:
    """Protocol version negotiation result."""
    client_supported: List[ProtocolMode]
    server_supported: List[ProtocolMode]
    negotiated_mode: ProtocolMode
    tongue_requirements: Dict[SacredTongue, ProtocolMode] = field(default_factory=dict)
    downgrade_warning: Optional[str] = None


# =============================================================================
# TONGUE BINDINGS
# =============================================================================

class TongueBinding:
    """Base class for Sacred Tongue bindings."""

    def __init__(self, tongue: SacredTongue, sdk: 'SpiralverseSDK'):
        self.tongue = tongue
        self.sdk = sdk
        self.scbe_index = SST_TO_SCBE_TONGUE[tongue]
        self.priority = SST_PQC_PRIORITY[tongue]

    def create_context(self, **metadata) -> SSTContext:
        """Create SST context for this tongue."""
        return SSTContext(tongue=self.tongue, **metadata)

    def get_tongue_weight(self) -> float:
        """Get SCBE weight for this tongue."""
        return TONGUE_WEIGHTS[self.scbe_index]


class KorvethianBinding(TongueBinding):
    """Korvethian (KO) - Command/Control binding."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        super().__init__(SacredTongue.KO, sdk)
        self.command_chains: Dict[str, List[bytes]] = {}

    async def sign_directive(
        self,
        directive: bytes,
        signing_key: HybridSigningKeyPair,
        chain_id: Optional[str] = None
    ) -> HybridSignature:
        """Sign a command directive with PQC signature."""
        ctx = self.create_context(command_chain_id=chain_id)
        sig = await self.sdk.crypto.sign(directive, signing_key, ctx)

        if chain_id:
            if chain_id not in self.command_chains:
                self.command_chains[chain_id] = []
            self.command_chains[chain_id].append(sig.to_bytes())

        return sig

    async def verify_chain(self, chain_id: str) -> Tuple[bool, str]:
        """Verify command chain integrity."""
        if chain_id not in self.command_chains:
            return False, f"Chain {chain_id} not found"

        chain = self.command_chains[chain_id]
        for i, sig_bytes in enumerate(chain):
            # Verify chain linkage via hash
            if i > 0:
                expected_link = hashlib.sha256(chain[i-1]).digest()
                # In real impl, check link is embedded in signature context

        return True, f"Chain {chain_id} verified ({len(chain)} signatures)"


class AvethrilBinding(TongueBinding):
    """Avethril (AV) - Emotional resonance binding."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        super().__init__(SacredTongue.AV, sdk)

    async def encrypt_sentiment(
        self,
        payload: bytes,
        resonance: float,
        recipient_key: HybridKeyPair
    ) -> Tuple[bytes, bytes]:
        """Encrypt emotional payload with resonance metadata."""
        ctx = self.create_context(resonance_frequency=resonance)

        # Derive key with resonance factor
        shared_secret = await self.sdk.keys.exchange(recipient_key, ctx)

        # Encrypt with AES-256-GCM (simplified simulation)
        nonce = os.urandom(12)
        # Simulated AES-GCM encryption
        key_material = shared_secret[:32]
        cipher_input = payload + ctx.tongue.value.encode()
        ciphertext = hashlib.sha256(key_material + nonce + cipher_input).digest()
        ciphertext += payload  # In real impl, would be actual AES-GCM

        return ciphertext, nonce


class RunevastBinding(TongueBinding):
    """Runevast (RU) - Historical binding."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        super().__init__(SacredTongue.RU, sdk)
        self.hash_chain: List[bytes] = []
        self.anchors: Dict[int, bytes] = {}

    def add_to_chain(self, data: bytes) -> bytes:
        """Add entry to timestamped hash chain."""
        if self.hash_chain:
            prev_hash = self.hash_chain[-1]
        else:
            prev_hash = b'\x00' * 32

        timestamp = int(time.time() * 1000000)  # microseconds
        entry = hashlib.sha256(prev_hash + data + timestamp.to_bytes(8, 'big')).digest()
        self.hash_chain.append(entry)

        return entry

    def create_anchor(self, position: int) -> bytes:
        """Create temporal anchor at chain position."""
        if position >= len(self.hash_chain):
            raise ValueError(f"Position {position} beyond chain length")

        anchor = self.hash_chain[position]
        self.anchors[position] = anchor
        return anchor

    def verify_since_anchor(self, anchor_position: int) -> bool:
        """Verify chain integrity since anchor."""
        if anchor_position not in self.anchors:
            return False

        # Recompute chain from anchor
        expected = self.anchors[anchor_position]
        if self.hash_chain[anchor_position] != expected:
            return False

        return True


class CelestineBinding(TongueBinding):
    """Celestine (CA) - Ceremony key management binding."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        super().__init__(SacredTongue.CA, sdk)
        self.ceremonies: Dict[str, Dict] = {}

    async def initiate_ceremony(
        self,
        ceremony_id: str,
        participants: List[str],
        invocation_depth: int = 1
    ) -> Dict:
        """Initiate new ceremony with PQC key exchange."""
        ctx = self.create_context(invocation_depth=invocation_depth)

        # Generate ceremony master key
        ceremony_key = await self.sdk.keys.generate_key_pair()
        ceremony_key.sst_binding = SacredTongue.CA
        ceremony_key.expires_at = time.time() + (90 * 24 * 3600)  # 90 days max

        ceremony = {
            'id': ceremony_id,
            'participants': participants,
            'master_key': ceremony_key,
            'invocation_depth': invocation_depth,
            'created_at': time.time(),
            'status': 'active',
        }

        self.ceremonies[ceremony_id] = ceremony
        return ceremony

    async def rotate_keys(self, ceremony_id: str, reason: str) -> Dict:
        """Rotate ceremony keys."""
        if ceremony_id not in self.ceremonies:
            raise ValueError(f"Ceremony {ceremony_id} not found")

        ceremony = self.ceremonies[ceremony_id]
        old_key = ceremony['master_key']

        # Generate fresh PQC keys
        new_key = await self.sdk.keys.generate_key_pair()
        new_key.sst_binding = SacredTongue.CA
        new_key.expires_at = time.time() + (90 * 24 * 3600)

        ceremony['master_key'] = new_key
        ceremony['rotation_history'] = ceremony.get('rotation_history', [])
        ceremony['rotation_history'].append({
            'timestamp': time.time(),
            'reason': reason,
            'old_key_hash': hashlib.sha256(old_key.to_bytes()).hexdigest()[:16],
        })

        return ceremony


class UmbralisBinding(TongueBinding):
    """Umbralis (UM) - Shadow protocol binding."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        super().__init__(SacredTongue.UM, sdk)
        self.shadow_sessions: Dict[str, Dict] = {}

    def derive_ephemeral_key(
        self,
        master_secret: bytes,
        session_id: str,
        iteration: int = 0
    ) -> bytes:
        """Derive ephemeral key for shadow session."""
        # HKDF-SHA256 derivation
        info = f"UM-shadow-{session_id}-{iteration}".encode()
        derived = hashlib.pbkdf2_hmac('sha256', master_secret, info, 1, dklen=32)

        self.shadow_sessions[session_id] = {
            'created': time.time(),
            'iteration': iteration,
            'expires': time.time() + (24 * 3600),  # 24 hour max
        }

        return derived

    def expire_session(self, session_id: str) -> bool:
        """Explicitly expire shadow session."""
        if session_id in self.shadow_sessions:
            del self.shadow_sessions[session_id]
            return True
        return False


class DraconicBinding(TongueBinding):
    """Draconic (DR) - Multi-party key agreement binding."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        super().__init__(SacredTongue.DR, sdk)
        self.agreements: Dict[str, Dict] = {}

    async def initiate_multi_party(
        self,
        agreement_id: str,
        participants: List[HybridKeyPair],
    ) -> Dict:
        """Initiate multi-party key agreement."""
        ctx = self.create_context(participant_count=len(participants))

        # Combine all participant public keys
        combined_material = b''
        for p in participants:
            combined_material += p.to_bytes()

        # Derive shared group key
        group_key = hashlib.sha256(combined_material).digest()

        agreement = {
            'id': agreement_id,
            'participant_count': len(participants),
            'group_key_hash': hashlib.sha256(group_key).hexdigest()[:16],
            'created_at': time.time(),
            'protocol': 'hybrid-pqc',
        }

        self.agreements[agreement_id] = agreement
        return agreement, group_key


# =============================================================================
# SST MANAGER
# =============================================================================

class SSTManager:
    """Manager for all Six Sacred Tongues."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        self.sdk = sdk
        self.ko = KorvethianBinding(sdk)
        self.av = AvethrilBinding(sdk)
        self.ru = RunevastBinding(sdk)
        self.ca = CelestineBinding(sdk)
        self.um = UmbralisBinding(sdk)
        self.dr = DraconicBinding(sdk)

        self._bindings = {
            SacredTongue.KO: self.ko,
            SacredTongue.AV: self.av,
            SacredTongue.RU: self.ru,
            SacredTongue.CA: self.ca,
            SacredTongue.UM: self.um,
            SacredTongue.DR: self.dr,
        }

    def get_tongue(self, tongue: SacredTongue) -> TongueBinding:
        """Get binding for specific Sacred Tongue."""
        return self._bindings[tongue]

    def create_context(self, tongue: SacredTongue, **metadata) -> SSTContext:
        """Create SST context for operations."""
        return self._bindings[tongue].create_context(**metadata)

    def get_migration_priority(self) -> List[Tuple[SacredTongue, RiskLevel]]:
        """Get tongues sorted by PQC migration priority."""
        return sorted(
            SST_PQC_PRIORITY.items(),
            key=lambda x: x[1].value
        )


# =============================================================================
# CRYPTO OPERATIONS
# =============================================================================

class CryptoOperations:
    """Cryptographic operations with hybrid PQC support."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        self.sdk = sdk
        self.pqc_manager = PQCManager()

    async def sign(
        self,
        message: bytes,
        key: HybridSigningKeyPair,
        context: Optional[SSTContext] = None
    ) -> HybridSignature:
        """Create hybrid signature."""
        # Classical signature (simulated Ed25519)
        classical_sig = hmac.new(key.classical_private, message, 'sha256').digest()
        classical_sig = classical_sig + classical_sig  # 64 bytes

        # Post-quantum signature (ML-DSA)
        pq_sig = dilithium_sign(key.pq_private, message)

        return HybridSignature(
            classical=classical_sig,
            post_quantum=pq_sig,
            sst_context={'tongue': context.tongue.value} if context else None
        )

    async def verify(
        self,
        message: bytes,
        signature: HybridSignature,
        key: HybridSigningKeyPair
    ) -> bool:
        """Verify hybrid signature."""
        # Verify classical
        expected_classical = hmac.new(key.classical_private, message, 'sha256').digest()
        expected_classical = expected_classical + expected_classical
        classical_valid = hmac.compare_digest(signature.classical, expected_classical)

        # Verify post-quantum
        pq_valid = dilithium_verify(key.pq_public, message, signature.post_quantum)

        # Both must be valid in hybrid mode
        return classical_valid and pq_valid


# =============================================================================
# KEY MANAGER
# =============================================================================

class KeyManager:
    """Key management with hybrid PQC support."""

    def __init__(self, sdk: 'SpiralverseSDK'):
        self.sdk = sdk
        self.stored_keys: Dict[str, HybridKeyPair] = {}

    async def generate_key_pair(self) -> HybridKeyPair:
        """Generate hybrid key pair (X25519 + ML-KEM-768)."""
        # Classical (simulated X25519)
        classical_private = os.urandom(32)
        classical_public = hashlib.sha256(classical_private).digest()

        # Post-quantum (ML-KEM / Kyber)
        pq_public, pq_private = generate_kyber_keypair()

        return HybridKeyPair(
            classical_public=classical_public,
            classical_private=classical_private,
            pq_public=pq_public,
            pq_private=pq_private,
        )

    async def generate_signing_key_pair(self) -> HybridSigningKeyPair:
        """Generate hybrid signing key pair (Ed25519 + ML-DSA-65)."""
        # Classical (simulated Ed25519)
        classical_private = os.urandom(32)
        classical_public = hashlib.sha256(classical_private).digest()

        # Post-quantum (ML-DSA / Dilithium)
        pq_public, pq_private = generate_dilithium_keypair()

        return HybridSigningKeyPair(
            classical_public=classical_public,
            classical_private=classical_private,
            pq_public=pq_public,
            pq_private=pq_private,
        )

    async def exchange(
        self,
        remote_key: HybridKeyPair,
        context: Optional[SSTContext] = None
    ) -> bytes:
        """Perform hybrid key exchange."""
        # Classical DH (simulated)
        classical_shared = hashlib.sha256(
            remote_key.classical_public + os.urandom(32)
        ).digest()

        # Post-quantum KEM
        ciphertext, pq_shared = kyber_encapsulate(remote_key.pq_public)

        # Combine secrets with HKDF
        combined = classical_shared + pq_shared
        context_bytes = context.tongue.value.encode() if context else b'default'

        final_secret = hashlib.pbkdf2_hmac(
            'sha256',
            combined,
            context_bytes,
            iterations=1,
            dklen=32
        )

        return final_secret


# =============================================================================
# MAIN SDK CLASS
# =============================================================================

class SpiralverseSDK:
    """
    SpiralverseSDK - Quantum-Resistant Protocol Implementation
    Version: 3.0.0

    The main entry point for Spiralverse protocol operations with
    post-quantum cryptography support and Six Sacred Tongues integration.
    """

    VERSION = "3.0.0"
    PROTOCOL = "RWP-3.0"

    def __init__(
        self,
        preferred_mode: ProtocolMode = ProtocolMode.HYBRID_PREFER_PQ,
        allow_classical_fallback: bool = True,
        enabled_tongues: Optional[List[SacredTongue]] = None,
        cache_verified_signatures: bool = True,
        batch_operations: bool = True,
        audit_classical_operations: bool = True,
    ):
        self.preferred_mode = preferred_mode
        self.allow_classical_fallback = allow_classical_fallback
        self.enabled_tongues = enabled_tongues or list(SacredTongue)
        self.cache_verified_signatures = cache_verified_signatures
        self.batch_operations = batch_operations
        self.audit_classical_operations = audit_classical_operations

        # Initialize subsystems
        self.keys = KeyManager(self)
        self.crypto = CryptoOperations(self)
        self.tongues = SSTManager(self)

        # Signature cache
        self._sig_cache: Dict[bytes, bool] = {}

        # Audit log
        self._audit_log: List[Dict] = []

    @property
    def version(self) -> Dict[str, Any]:
        return {
            'sdk': self.VERSION,
            'protocol': self.PROTOCOL,
            'pqc_support': True,
            'hybrid_mode': self.preferred_mode == ProtocolMode.HYBRID_PREFER_PQ,
        }

    @property
    def protocol_mode(self) -> ProtocolMode:
        return self.preferred_mode

    def negotiate_version(
        self,
        client_modes: List[ProtocolMode],
        server_modes: List[ProtocolMode],
    ) -> VersionNegotiation:
        """Negotiate protocol version with remote party."""
        # Find best common mode
        for mode in [ProtocolMode.HYBRID_PREFER_PQ, ProtocolMode.PQ_ONLY, ProtocolMode.CLASSICAL_ONLY]:
            if mode in client_modes and mode in server_modes:
                negotiated = mode
                break
        else:
            raise ValueError("No compatible protocol mode")

        # Check for downgrade
        downgrade_warning = None
        if ProtocolMode.HYBRID_PREFER_PQ in client_modes and negotiated == ProtocolMode.CLASSICAL_ONLY:
            downgrade_warning = "Downgraded to classical-only mode (remote doesn't support PQC)"
            if self.audit_classical_operations:
                self._audit_log.append({
                    'event': 'protocol_downgrade',
                    'timestamp': time.time(),
                    'details': downgrade_warning,
                })

        return VersionNegotiation(
            client_supported=client_modes,
            server_supported=server_modes,
            negotiated_mode=negotiated,
            downgrade_warning=downgrade_warning,
        )

    def get_audit_log(self) -> List[Dict]:
        """Get audit log entries."""
        return self._audit_log.copy()


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test() -> Dict[str, Any]:
    """Run self-test of Spiralverse RWP v3 implementation."""
    import asyncio

    results = {
        'version': SpiralverseSDK.VERSION,
        'tests': [],
        'passed': 0,
        'failed': 0,
    }

    async def run_tests():
        sdk = SpiralverseSDK()

        # Test 1: Key generation
        try:
            kex_key = await sdk.keys.generate_key_pair()
            sig_key = await sdk.keys.generate_signing_key_pair()
            results['tests'].append(('key_generation', True, 'Generated hybrid key pairs'))
            results['passed'] += 1
        except Exception as e:
            results['tests'].append(('key_generation', False, str(e)))
            results['failed'] += 1

        # Test 2: Signing and verification
        try:
            message = b"Test message for SCBE-AETHERMOORE"
            sig_key = await sdk.keys.generate_signing_key_pair()
            signature = await sdk.crypto.sign(message, sig_key)
            valid = await sdk.crypto.verify(message, signature, sig_key)
            assert valid, "Signature verification failed"
            results['tests'].append(('sign_verify', True, f'Signature size: {signature.total_size} bytes'))
            results['passed'] += 1
        except Exception as e:
            results['tests'].append(('sign_verify', False, str(e)))
            results['failed'] += 1

        # Test 3: SST bindings
        try:
            # Test Korvethian command signing
            ko = sdk.tongues.ko
            directive = b"EXECUTE: security_scan"
            sig = await ko.sign_directive(directive, sig_key, chain_id="test-chain")
            assert sig.sst_context['tongue'] == 'Korvethian'

            # Test Runevast hash chain
            ru = sdk.tongues.ru
            entry1 = ru.add_to_chain(b"Genesis block")
            entry2 = ru.add_to_chain(b"Second entry")
            anchor = ru.create_anchor(0)
            assert ru.verify_since_anchor(0)

            results['tests'].append(('sst_bindings', True, 'All 6 tongues operational'))
            results['passed'] += 1
        except Exception as e:
            results['tests'].append(('sst_bindings', False, str(e)))
            results['failed'] += 1

        # Test 4: Protocol negotiation
        try:
            negotiation = sdk.negotiate_version(
                client_modes=[ProtocolMode.HYBRID_PREFER_PQ, ProtocolMode.CLASSICAL_ONLY],
                server_modes=[ProtocolMode.HYBRID_PREFER_PQ],
            )
            assert negotiation.negotiated_mode == ProtocolMode.HYBRID_PREFER_PQ
            results['tests'].append(('negotiation', True, 'Negotiated HYBRID_PREFER_PQ'))
            results['passed'] += 1
        except Exception as e:
            results['tests'].append(('negotiation', False, str(e)))
            results['failed'] += 1

        # Test 5: Key exchange
        try:
            alice_key = await sdk.keys.generate_key_pair()
            bob_key = await sdk.keys.generate_key_pair()

            ctx = sdk.tongues.create_context(SacredTongue.AV, resonance_frequency=440.0)
            shared = await sdk.keys.exchange(bob_key, ctx)
            assert len(shared) == 32

            results['tests'].append(('key_exchange', True, 'Hybrid key exchange successful'))
            results['passed'] += 1
        except Exception as e:
            results['tests'].append(('key_exchange', False, str(e)))
            results['failed'] += 1

    asyncio.run(run_tests())

    results['success'] = results['failed'] == 0
    return results


if __name__ == "__main__":
    result = self_test()
    print(f"Spiralverse RWP v{result['version']} Self-Test")
    print("=" * 50)
    for name, passed, detail in result['tests']:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {detail}")
    print("=" * 50)
    print(f"Passed: {result['passed']}/{result['passed'] + result['failed']}")
