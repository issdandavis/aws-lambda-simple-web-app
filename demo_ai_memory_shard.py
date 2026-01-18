#!/usr/bin/env python3
"""
AI Memory Shard Demo
====================

End-to-end demonstration of the Spiralverse Protocol stack:

1. SEAL   - Encrypt memory with SpiralSeal SS1 (Sacred Tongue encoding)
2. STORE  - Place in harmonic slot (6D coordinate + cymatic position)
3. GOVERN - Check governance layers before retrieval
4. UNSEAL - Retrieve and decrypt if authorized

This ties together:
- SpiralSeal SS1 (spell-text crypto)
- GeoSeal (dual-manifold authorization)
- Governance Engine (snap detection, causality)
- Post-Quantum Crypto (Kyber/Dilithium simulation)
- Harmonic Scaling (risk amplification)

Usage:
    python demo_ai_memory_shard.py
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Imports from our stack
# ═══════════════════════════════════════════════════════════════

from symphonic_cipher.scbe_aethermoore.spiral_seal import (
    seal, unseal, SpiralSealSS1
)
from symphonic_cipher.scbe_aethermoore.geoseal import (
    GeoSealEngine, GeoSealState, SphericalCoord, HypercubeCoord,
    compute_time_dilation, IntersectionType
)
from symphonic_cipher.scbe_aethermoore.governance import (
    GovernanceEngine, SnapProtocol, CausalityVerifier,
    harmonic_scaling, evaluate_gue, GovernanceDecision
)
from symphonic_cipher.scbe_aethermoore.quantum import (
    PQCryptoSystem, PQContextCommitment
)
from symphonic_cipher.qasi_core import (
    realm_distance, hyperbolic_distance
)


# ═══════════════════════════════════════════════════════════════
# Memory Shard Storage (Harmonic Slot)
# ═══════════════════════════════════════════════════════════════

@dataclass
class HarmonicSlot:
    """A slot in 6D harmonic space for storing memory shards."""
    coord_6d: Tuple[float, float, float, float, float, float]
    sealed_blob: str
    agent_id: str
    topic: str
    timestamp: float = field(default_factory=time.time)
    mode: str = "standard"  # standard, sensitive, critical

    def slot_hash(self) -> str:
        """Unique identifier for this slot."""
        data = f"{self.coord_6d}:{self.agent_id}:{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class HarmonicMemoryStore:
    """
    Simple in-memory store for memory shards.

    Maps 6D coordinates to sealed blobs with governance metadata.
    """

    def __init__(self):
        self.slots: Dict[str, HarmonicSlot] = {}
        self.realm_centers: List[np.ndarray] = [
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # "home" realm
        ]

    def store(self, slot: HarmonicSlot) -> str:
        """Store a memory shard, return slot hash."""
        slot_id = slot.slot_hash()
        self.slots[slot_id] = slot
        return slot_id

    def retrieve(self, slot_id: str) -> Optional[HarmonicSlot]:
        """Retrieve a slot by ID."""
        return self.slots.get(slot_id)

    def compute_d_star(self, coord_6d: Tuple) -> float:
        """Compute distance to nearest realm center."""
        u = np.array(coord_6d)
        centers = np.array(self.realm_centers)
        return realm_distance(u, centers)


# ═══════════════════════════════════════════════════════════════
# Governance Gate
# ═══════════════════════════════════════════════════════════════

@dataclass
class GovernanceResult:
    """Result of governance check."""
    allowed: bool
    decision: str
    risk_prime: float
    d_star: float
    harmonic_scale: float
    geoseal_authorized: bool
    pq_valid: bool
    reason: str


def check_governance(
    coord_6d: Tuple[float, ...],
    agent_id: str,
    store: HarmonicMemoryStore,
    master_secret: bytes,
) -> GovernanceResult:
    """
    Full governance check for memory retrieval.

    Layers:
    1. Harmonic risk scaling (d* distance)
    2. GeoSeal dual-manifold authorization
    3. Post-quantum signature verification
    """

    # 1. Compute hyperbolic distance to realm
    d_star = store.compute_d_star(coord_6d)

    # 2. Harmonic scaling: H(d*, R) = R^(d*²)
    R = 1.5  # Perfect fifth
    try:
        H = harmonic_scaling(d_star, R)
    except (OverflowError, ValueError):
        H = 11.0  # Bounded fallback

    # Base risk from position (farther = riskier)
    risk_base = min(0.1 + d_star * 0.2, 1.0)
    risk_prime = risk_base * H

    # 3. GeoSeal check
    geoseal = GeoSealEngine(master_secret)
    # Map 6D to sphere (theta, phi) and hypercube
    theta = coord_6d[0] * np.pi
    phi = coord_6d[1] * 2 * np.pi
    policy_coords = tuple(coord_6d[2:5])  # Use dims 2-4 for policy

    state = geoseal.create_state(theta, phi, policy_coords)
    intersection, keys, dilation = geoseal.authorize(state)

    geoseal_authorized = intersection.authorized

    # 4. PQ signature check (simulated)
    pq = PQCryptoSystem()
    test_message = f"{agent_id}:{coord_6d}".encode()
    sig = pq.sign(test_message)
    pq_valid = pq.verify_signature(test_message, sig)

    # 5. Final decision
    if risk_prime < 0.30 and geoseal_authorized and pq_valid:
        decision = "ALLOW"
        allowed = True
        reason = "All checks passed"
    elif risk_prime > 0.70:
        decision = "DENY"
        allowed = False
        reason = f"Risk too high: {risk_prime:.2f}"
    elif not geoseal_authorized:
        decision = "DENY"
        allowed = False
        reason = f"GeoSeal: outside authorized manifold ({intersection.type.value})"
    elif not pq_valid:
        decision = "DENY"
        allowed = False
        reason = "Post-quantum signature invalid"
    else:
        decision = "QUARANTINE"
        allowed = False
        reason = f"Risk in gray zone: {risk_prime:.2f}"

    return GovernanceResult(
        allowed=allowed,
        decision=decision,
        risk_prime=risk_prime,
        d_star=d_star,
        harmonic_scale=H,
        geoseal_authorized=geoseal_authorized,
        pq_valid=pq_valid,
        reason=reason,
    )


# ═══════════════════════════════════════════════════════════════
# Main Demo Flow
# ═══════════════════════════════════════════════════════════════

def demo_seal_memory(
    payload: str,
    agent_id: str,
    topic: str,
    coord_6d: Tuple[float, ...],
    master_secret: bytes,
) -> Tuple[str, HarmonicSlot]:
    """
    SEAL: Encrypt a memory payload and store in harmonic slot.
    """
    print("\n" + "═" * 60)
    print("PHASE 1: SEAL MEMORY")
    print("═" * 60)

    # Create SpiralSeal instance
    ss = SpiralSealSS1(master_secret)

    # AAD encodes context
    aad = f"agent={agent_id};topic={topic}"

    # Seal the payload
    sealed_blob = ss.seal(payload.encode(), aad=aad)

    print(f"  Agent:    {agent_id}")
    print(f"  Topic:    {topic}")
    print(f"  Payload:  {payload[:50]}{'...' if len(payload) > 50 else ''}")
    print(f"  AAD:      {aad}")
    print(f"  Coord 6D: {coord_6d}")
    print()
    print(f"  Sealed blob (spell-text):")
    # Show first part of each tongue section
    parts = sealed_blob.split("|")
    for part in parts[:4]:
        print(f"    {part[:60]}{'...' if len(part) > 60 else ''}")
    print(f"    ... ({len(sealed_blob)} chars total)")

    # Create slot
    slot = HarmonicSlot(
        coord_6d=coord_6d,
        sealed_blob=sealed_blob,
        agent_id=agent_id,
        topic=topic,
    )

    return sealed_blob, slot


def demo_store_memory(
    slot: HarmonicSlot,
    store: HarmonicMemoryStore,
) -> str:
    """
    STORE: Place sealed shard in harmonic memory store.
    """
    print("\n" + "═" * 60)
    print("PHASE 2: STORE IN HARMONIC SLOT")
    print("═" * 60)

    slot_id = store.store(slot)
    d_star = store.compute_d_star(slot.coord_6d)

    print(f"  Slot ID:  {slot_id}")
    print(f"  d* (realm distance): {d_star:.4f}")
    print(f"  Mode:     {slot.mode}")
    print(f"  Stored at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(slot.timestamp))}")

    return slot_id


def demo_govern_retrieval(
    slot_id: str,
    store: HarmonicMemoryStore,
    master_secret: bytes,
) -> GovernanceResult:
    """
    GOVERN: Check all governance layers before allowing retrieval.
    """
    print("\n" + "═" * 60)
    print("PHASE 3: GOVERNED RETRIEVAL")
    print("═" * 60)

    slot = store.retrieve(slot_id)
    if not slot:
        print("  ERROR: Slot not found")
        return None

    print(f"  Checking governance for slot {slot_id}...")
    print()

    result = check_governance(
        slot.coord_6d,
        slot.agent_id,
        store,
        master_secret,
    )

    # Print trace
    print("  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ GOVERNANCE TRACE                                        │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ d* (hyperbolic distance):  {result.d_star:>8.4f}                   │")
    print(f"  │ H(d*, R) harmonic scale:   {result.harmonic_scale:>8.2f}                   │")
    print(f"  │ Risk' (amplified):         {result.risk_prime:>8.4f}                   │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ GeoSeal (manifold check):  {'✓ PASS' if result.geoseal_authorized else '✗ FAIL':>10}                   │")
    print(f"  │ Post-Quantum (signature):  {'✓ PASS' if result.pq_valid else '✗ FAIL':>10}                   │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ DECISION: {result.decision:<15}                           │")
    print(f"  │ Reason: {result.reason:<40}    │")
    print("  └─────────────────────────────────────────────────────────┘")

    return result


def demo_unseal_memory(
    slot_id: str,
    store: HarmonicMemoryStore,
    governance: GovernanceResult,
    master_secret: bytes,
) -> Optional[str]:
    """
    UNSEAL: Decrypt and return memory if governance allows.
    """
    print("\n" + "═" * 60)
    print("PHASE 4: UNSEAL MEMORY")
    print("═" * 60)

    if not governance.allowed:
        print(f"  ✗ BLOCKED by governance: {governance.decision}")
        print(f"  Reason: {governance.reason}")
        print("  Memory remains sealed.")
        return None

    slot = store.retrieve(slot_id)
    if not slot:
        print("  ERROR: Slot not found")
        return None

    # Unseal
    ss = SpiralSealSS1(master_secret)
    try:
        plaintext = ss.unseal(slot.sealed_blob)
        payload = plaintext.decode()

        print(f"  ✓ UNSEALED successfully")
        print(f"  Agent:   {slot.agent_id}")
        print(f"  Topic:   {slot.topic}")
        print()
        print(f"  Recovered payload:")
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        for line in payload.split('\n'):
            print(f"  │ {line:<55} │")
        print(f"  └─────────────────────────────────────────────────────────┘")

        return payload

    except ValueError as e:
        print(f"  ✗ UNSEAL FAILED: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# Run Demo
# ═══════════════════════════════════════════════════════════════

def run_demo():
    """Run the full AI Memory Shard demo."""

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║         AI MEMORY SHARD DEMO - Spiralverse Protocol          ║")
    print("║                                                               ║")
    print("║  SpiralSeal + GeoSeal + Governance + Post-Quantum            ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # Setup
    master_secret = b"spiralverse-demo-key-32b!"  # 32 bytes for AES-256
    master_secret = master_secret.ljust(32, b'\x00')

    store = HarmonicMemoryStore()

    # Demo payload (an AI memory/conversation snippet)
    memory_payload = """User asked: "What is the Spiralverse Protocol?"
Agent response: It's a quantum-resistant semantic encoding
framework for AI coordination, using harmonic scaling and
dual-manifold geometry for trust verification."""

    # ─────────────────────────────────────────────────────────
    # SCENARIO 1: Normal access (should ALLOW)
    # ─────────────────────────────────────────────────────────
    print("\n" + "▓" * 63)
    print("▓  SCENARIO 1: Normal Access (close to realm center)          ▓")
    print("▓" * 63)

    # Coordinate near realm center (low risk)
    coord_safe = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    sealed, slot = demo_seal_memory(
        payload=memory_payload,
        agent_id="agent-alpha-7",
        topic="protocol-explanation",
        coord_6d=coord_safe,
        master_secret=master_secret,
    )

    slot_id = demo_store_memory(slot, store)
    governance = demo_govern_retrieval(slot_id, store, master_secret)
    recovered = demo_unseal_memory(slot_id, store, governance, master_secret)

    # ─────────────────────────────────────────────────────────
    # SCENARIO 2: Suspicious access (should DENY)
    # ─────────────────────────────────────────────────────────
    print("\n" + "▓" * 63)
    print("▓  SCENARIO 2: Suspicious Access (far from realm center)      ▓")
    print("▓" * 63)

    # Coordinate far from realm center (high risk)
    coord_suspicious = (0.95, 0.95, 0.1, 0.1, 0.9, 0.9)

    sealed2, slot2 = demo_seal_memory(
        payload="Secret: The backdoor code is XYZ123",
        agent_id="agent-unknown",
        topic="sensitive-data",
        coord_6d=coord_suspicious,
        master_secret=master_secret,
    )

    slot_id2 = demo_store_memory(slot2, store)
    governance2 = demo_govern_retrieval(slot_id2, store, master_secret)
    recovered2 = demo_unseal_memory(slot_id2, store, governance2, master_secret)

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    print("\n" + "═" * 63)
    print("DEMO COMPLETE")
    print("═" * 63)
    print(f"""
  This demo showed:

  1. SEAL: Memory encrypted with SpiralSeal SS1
     - Payload → Sacred Tongue spell-text
     - AES-256-GCM with HKDF key derivation

  2. STORE: Placed in 6D harmonic coordinate space
     - Position determines risk via d* distance

  3. GOVERN: Multi-layer authorization
     - Harmonic scaling: H(d*, R) = R^(d*²)
     - GeoSeal dual-manifold intersection
     - Post-quantum signature verification

  4. UNSEAL: Conditional retrieval
     - Only if ALL governance layers approve

  Scenario 1 (safe):       {governance.decision}
  Scenario 2 (suspicious): {governance2.decision}
""")


if __name__ == "__main__":
    run_demo()
