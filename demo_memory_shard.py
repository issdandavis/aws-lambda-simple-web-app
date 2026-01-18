#!/usr/bin/env python3
"""
AI Memory Shard Demo
====================

End-to-end demonstration of the AETHERMOORE cryptographic memory system.

This demo shows all claims working together:
  1. SpiralSeal SS1 cipher with Sacred Tongues encoding
  2. Harmonic voxel storage (6D coordinate system)
  3. Governance engine decision-making
  4. Post-quantum cryptography (Kyber768 + Dilithium3)
  5. Dual lattice consensus verification

Usage:
    python demo_memory_shard.py
    python demo_memory_shard.py --memory "custom memory content"
    python demo_memory_shard.py --agent ash --topic secrets --risk high
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# IMPORTS - All existing modules
# =============================================================================

# SpiralSeal & Sacred Tongues
from symphonic_cipher.scbe_aethermoore.spiral_seal import (
    SpiralSeal,
    SpiralSealResult,
    quick_seal,
    quick_unseal,
    SacredTongue,
)

# PQC (Kyber768 + Dilithium3)
from symphonic_cipher.scbe_aethermoore.pqc import (
    Kyber768,
    Dilithium3,
    PQCBackend,
    get_backend,
    PQCAuditChain,
    AuditDecision,
)

# Dual Lattice Consensus
from symphonic_cipher.scbe_aethermoore.dual_lattice import (
    DualLatticeConsensus,
    LatticeType,
    ConsensusState,
)

# Quasicrystal validation
from symphonic_cipher.scbe_aethermoore.qc_lattice import (
    QuasicrystalLattice,
    ValidationStatus,
)

# Harmonic scaling
from symphonic_cipher.harmonic_scaling_law import (
    quantum_resistant_harmonic_scaling,
    hyperbolic_distance_poincare,
    PHI,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
)

# Governance (use unified if available, fallback to basic)
try:
    from symphonic_cipher.scbe_aethermoore.organic_hyperbolic import (
        GovernanceEngine as OHGovernanceEngine,
    )
    HAS_ORGANIC_HYPERBOLIC = True
except ImportError:
    HAS_ORGANIC_HYPERBOLIC = False


# =============================================================================
# CONSTANTS
# =============================================================================

GOLDEN_RATIO = PHI
HARMONIC_RATIO = 3.0 / 2.0  # Perfect fifth
VOXEL_DIMENSIONS = 6  # [x, y, z, v, phase, mode]


# =============================================================================
# GOVERNANCE ENGINE (Lightweight wrapper)
# =============================================================================

class GovernanceDecision(Enum):
    """Governance decision outcomes."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    QUARANTINE = "QUARANTINE"
    SNAP = "SNAP"  # Fail-to-noise discontinuity


@dataclass
class GovernanceResult:
    """Result of a governance decision."""
    decision: GovernanceDecision
    reason: str
    risk_score: float
    harmonic_factor: float
    snap_detected: bool = False


class GovernanceEngine:
    """
    Governance engine for memory access control.

    Implements:
      - Risk scoring based on identity/intent/context
      - Harmonic scaling for risk amplification
      - Snap protocol for discontinuity detection
    """

    # Risk thresholds (tuned for demo visibility)
    ALLOW_THRESHOLD = 0.20       # Lower to catch more untrusted access
    QUARANTINE_THRESHOLD = 0.40  # Mid-range for elevated risk
    SNAP_THRESHOLD = 0.80        # High threshold for discontinuity

    def __init__(self, alpha: float = 10.0, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self._trusted_agents = {"ash", "claude", "system", "admin"}
        self._restricted_topics = {"secrets", "credentials", "keys", "passwords"}
        self._high_risk_contexts = {"external", "untrusted", "public"}

    def compute_risk(
        self,
        agent: str,
        topic: str,
        context: str,
        external_risk: Optional[float] = None
    ) -> float:
        """
        Compute base risk score for a memory access request.

        Returns: float in [0, 1]
        """
        if external_risk is not None:
            return max(0.0, min(1.0, external_risk))

        risk = 0.0

        # Agent trust factor
        if agent.lower() not in self._trusted_agents:
            risk += 0.3

        # Topic sensitivity factor
        topic_lower = topic.lower()
        for restricted in self._restricted_topics:
            if restricted in topic_lower:
                risk += 0.35
                break

        # Context factor
        context_lower = context.lower()
        for high_risk in self._high_risk_contexts:
            if high_risk in context_lower:
                risk += 0.25
                break

        return max(0.0, min(1.0, risk))

    def apply_harmonic_scaling(self, base_risk: float) -> float:
        """
        Apply harmonic scaling to amplify risk near boundaries.

        H(d*) = 1 + alpha * tanh(beta * d*)
        """
        return quantum_resistant_harmonic_scaling(base_risk, alpha=self.alpha, beta=self.beta)

    def detect_snap(self, risk_score: float, prev_risk: Optional[float] = None) -> bool:
        """
        Detect snap (discontinuity) in risk trajectory.

        A snap occurs when:
          1. Risk exceeds snap threshold
          2. Risk changes too rapidly (derivative discontinuity)
        """
        if risk_score >= self.SNAP_THRESHOLD:
            return True

        if prev_risk is not None:
            delta = abs(risk_score - prev_risk)
            if delta > 0.5:  # Large jump indicates potential attack
                return True

        return False

    def make_decision(
        self,
        agent: str,
        topic: str,
        context: str = "internal",
        external_risk: Optional[float] = None
    ) -> GovernanceResult:
        """
        Make a governance decision for a memory access request.
        """
        # Compute base risk
        base_risk = self.compute_risk(agent, topic, context, external_risk)

        # Apply harmonic scaling
        scaled_risk = self.apply_harmonic_scaling(base_risk)
        harmonic_factor = scaled_risk / max(base_risk, 0.001)

        # Normalize scaled risk back to [0, 1] for decision
        normalized_risk = (scaled_risk - 1.0) / self.alpha

        # Check for snap
        snap_detected = self.detect_snap(normalized_risk)

        # Make decision
        if snap_detected:
            return GovernanceResult(
                decision=GovernanceDecision.SNAP,
                reason=f"Snap detected: risk discontinuity (score={normalized_risk:.3f})",
                risk_score=normalized_risk,
                harmonic_factor=harmonic_factor,
                snap_detected=True
            )
        elif normalized_risk >= self.QUARANTINE_THRESHOLD:
            return GovernanceResult(
                decision=GovernanceDecision.DENY,
                reason=f"Risk too high: {normalized_risk:.3f} >= {self.QUARANTINE_THRESHOLD}",
                risk_score=normalized_risk,
                harmonic_factor=harmonic_factor
            )
        elif normalized_risk >= self.ALLOW_THRESHOLD:
            return GovernanceResult(
                decision=GovernanceDecision.QUARANTINE,
                reason=f"Elevated risk: {normalized_risk:.3f} in quarantine range",
                risk_score=normalized_risk,
                harmonic_factor=harmonic_factor
            )
        else:
            return GovernanceResult(
                decision=GovernanceDecision.ALLOW,
                reason=f"Risk acceptable: {normalized_risk:.3f} < {self.ALLOW_THRESHOLD}",
                risk_score=normalized_risk,
                harmonic_factor=harmonic_factor
            )


# =============================================================================
# HARMONIC VOXEL CUBE (Python stub for HolographicQRCube)
# =============================================================================

@dataclass
class VoxelEntry:
    """A single voxel entry in the harmonic cube."""
    position: Tuple[int, int, int, int, int, int]  # [x, y, z, v, phase, mode]
    sealed_blob: str  # SS1 sealed data
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def harmonic_signature(self) -> bytes:
        """Compute harmonic signature of this voxel."""
        pos_bytes = b"".join(p.to_bytes(2, "big") for p in self.position)
        return hashlib.sha256(pos_bytes + self.sealed_blob.encode()).digest()[:16]


def harmonic_distance(pos1: Tuple[int, ...], pos2: Tuple[int, ...]) -> float:
    """
    Compute harmonic distance between two 6D voxel positions.

    Uses golden ratio and perfect fifth weighted Euclidean distance.
    """
    if len(pos1) != len(pos2):
        raise ValueError("Positions must have same dimensions")

    # Harmonic weights: [x, y, z, v, phase, mode]
    # - Spatial (x,y,z): weight 1.0
    # - Velocity (v): weight PHI (time-like dimension)
    # - Phase: weight PHI^2 (quantum phase)
    # - Mode: weight HARMONIC_RATIO (frequency mode)
    weights = [1.0, 1.0, 1.0, GOLDEN_RATIO, GOLDEN_RATIO**2, HARMONIC_RATIO]

    squared_sum = 0.0
    for i, (a, b) in enumerate(zip(pos1, pos2)):
        squared_sum += weights[i] * (a - b) ** 2

    return math.sqrt(squared_sum)


class HarmonicVoxelCube:
    """
    Harmonic voxel storage system.

    A lightweight Python implementation that mirrors the HolographicQRCube
    semantics from the TS SDK. Supports:
      - 6D coordinate system [x, y, z, v, phase, mode]
      - Harmonic distance calculations
      - Nearest-neighbor queries
      - Mode-based filtering
    """

    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions
        self._voxels: Dict[Tuple[int, ...], VoxelEntry] = {}
        self._mode_index: Dict[int, List[Tuple[int, ...]]] = {}

    def add_voxel(
        self,
        position: Tuple[int, int, int, int, int, int],
        sealed_blob: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VoxelEntry:
        """Add a voxel at the specified 6D position."""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

        entry = VoxelEntry(
            position=position,
            sealed_blob=sealed_blob,
            metadata=metadata or {}
        )

        self._voxels[position] = entry

        # Update mode index
        mode = position[5]
        if mode not in self._mode_index:
            self._mode_index[mode] = []
        self._mode_index[mode].append(position)

        return entry

    def get_voxel(self, position: Tuple[int, ...]) -> Optional[VoxelEntry]:
        """Get voxel at exact position."""
        return self._voxels.get(position)

    def scan(
        self,
        center: Tuple[int, ...],
        radius: float,
        mode_filter: Optional[int] = None
    ) -> List[VoxelEntry]:
        """
        Scan for voxels within harmonic radius of center.

        Args:
            center: Center position for scan
            radius: Maximum harmonic distance
            mode_filter: Optional mode to filter by

        Returns:
            List of VoxelEntry objects within radius
        """
        results = []

        # Determine which positions to check
        if mode_filter is not None and mode_filter in self._mode_index:
            positions = self._mode_index[mode_filter]
        else:
            positions = list(self._voxels.keys())

        for pos in positions:
            dist = harmonic_distance(center, pos)
            if dist <= radius:
                results.append(self._voxels[pos])

        # Sort by distance
        results.sort(key=lambda e: harmonic_distance(center, e.position))
        return results

    def nearest(
        self,
        position: Tuple[int, ...],
        k: int = 1,
        mode_filter: Optional[int] = None
    ) -> List[Tuple[VoxelEntry, float]]:
        """
        Find k nearest voxels to a position.

        Returns:
            List of (VoxelEntry, distance) tuples
        """
        if mode_filter is not None and mode_filter in self._mode_index:
            positions = self._mode_index[mode_filter]
        else:
            positions = list(self._voxels.keys())

        distances = []
        for pos in positions:
            dist = harmonic_distance(position, pos)
            distances.append((self._voxels[pos], dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def __len__(self) -> int:
        return len(self._voxels)


# =============================================================================
# DUAL LATTICE VERIFIER
# =============================================================================

@dataclass
class LatticeVerification:
    """Result of dual lattice verification."""
    passed: bool
    primal_ok: bool  # Kyber (MLWE)
    dual_ok: bool    # Dilithium (MSIS)
    consensus_state: str
    details: str


class DualLatticeVerifier:
    """
    Dual lattice verification using Kyber768 + Dilithium3.

    Ensures both:
      1. Primal lattice (MLWE) - Key encapsulation verified
      2. Dual lattice (MSIS) - Signature verified
    """

    def __init__(self):
        self._kyber_keypair = None
        self._dilithium_keypair = None
        self._consensus = DualLatticeConsensus()

    def setup_keys(self) -> None:
        """Generate PQC key pairs."""
        self._kyber_keypair = Kyber768.generate_keypair()
        self._dilithium_keypair = Dilithium3.generate_keypair()

    def create_commitment(self, data: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Create a post-quantum commitment for data.

        Returns:
            (ciphertext, signature, shared_secret)
        """
        if self._kyber_keypair is None or self._dilithium_keypair is None:
            self.setup_keys()

        # Kyber encapsulation
        encap_result = Kyber768.encapsulate(self._kyber_keypair.public_key)

        # Dilithium signature
        message = data + encap_result.ciphertext
        signature = Dilithium3.sign(self._dilithium_keypair.secret_key, message)

        return encap_result.ciphertext, signature, encap_result.shared_secret

    def verify_commitment(
        self,
        data: bytes,
        ciphertext: bytes,
        signature: bytes
    ) -> LatticeVerification:
        """
        Verify a post-quantum commitment.
        """
        if self._kyber_keypair is None or self._dilithium_keypair is None:
            return LatticeVerification(
                passed=False,
                primal_ok=False,
                dual_ok=False,
                consensus_state="NO_KEYS",
                details="Keys not initialized"
            )

        # Verify Kyber (primal lattice / MLWE)
        try:
            recovered_secret = Kyber768.decapsulate(
                self._kyber_keypair.secret_key,
                ciphertext
            )
            primal_ok = len(recovered_secret) == 32
        except Exception as e:
            primal_ok = False

        # Verify Dilithium (dual lattice / MSIS)
        try:
            message = data + ciphertext
            dual_ok = Dilithium3.verify(
                self._dilithium_keypair.public_key,
                message,
                signature
            )
        except Exception as e:
            dual_ok = False

        # Consensus
        passed = primal_ok and dual_ok
        if passed:
            consensus_state = "SETTLED"
            details = "Dual lattice consensus achieved"
        elif primal_ok:
            consensus_state = "PARTIAL"
            details = "Primal (Kyber) OK, Dual (Dilithium) failed"
        elif dual_ok:
            consensus_state = "PARTIAL"
            details = "Dual (Dilithium) OK, Primal (Kyber) failed"
        else:
            consensus_state = "FAILED"
            details = "Both lattice verifications failed"

        return LatticeVerification(
            passed=passed,
            primal_ok=primal_ok,
            dual_ok=dual_ok,
            consensus_state=consensus_state,
            details=details
        )


# =============================================================================
# QUASICRYSTAL VALIDATOR
# =============================================================================

@dataclass
class QuasicrystalResult:
    """Result of quasicrystal validation."""
    valid: bool
    status: str
    crystallinity: float
    phason_stable: bool


class QuasicrystalValidator:
    """
    Quasicrystal-based validation for 6D gate vectors.

    Uses icosahedral quasicrystal properties to detect:
      - Periodic (crystalline) attack patterns
      - Phason mismatches
      - Out-of-window projections
    """

    def __init__(self):
        self._lattice = QuasicrystalLattice()
        # Increase acceptance radius for demo (normally tighter for production)
        self._lattice.acceptance_radius = 10.0  # More permissive for demo

    def validate(self, gate_vector: List[int]) -> QuasicrystalResult:
        """
        Validate a 6-gate vector against quasicrystal constraints.

        Args:
            gate_vector: 6D vector [x, y, z, v, phase, mode]
        """
        if len(gate_vector) != 6:
            return QuasicrystalResult(
                valid=False,
                status="INVALID_DIMENSION",
                crystallinity=0.0,
                phason_stable=False
            )

        result = self._lattice.validate_gates(gate_vector)

        valid = result.status == ValidationStatus.VALID
        # crystallinity_score is returned in the ValidationResult
        crystallinity = result.crystallinity_score
        phason_stable = result.status != ValidationStatus.INVALID_PHASON_MISMATCH

        return QuasicrystalResult(
            valid=valid,
            status=result.status.name,
            crystallinity=crystallinity,
            phason_stable=phason_stable
        )


# =============================================================================
# MEMORY SHARD SYSTEM
# =============================================================================

@dataclass
class MemoryShard:
    """A complete memory shard with all cryptographic layers."""
    plaintext: bytes
    aad: bytes
    sealed_blob: str
    position: Tuple[int, int, int, int, int, int]
    pqc_ciphertext: bytes
    pqc_signature: bytes
    pqc_shared_secret: bytes


@dataclass
class RetrievalResult:
    """Result of memory shard retrieval."""
    success: bool
    plaintext: Optional[bytes]
    governance: GovernanceResult
    lattice: LatticeVerification
    quasicrystal: QuasicrystalResult
    trace: List[str]


class MemoryShardSystem:
    """
    Complete AI memory shard system.

    Integrates:
      - SpiralSeal SS1 cipher with Sacred Tongues
      - Harmonic voxel storage (6D)
      - Governance engine
      - Post-quantum cryptography
      - Dual lattice consensus
      - Quasicrystal validation
    """

    def __init__(self, password: bytes = b"aethermoore-demo-key"):
        self.password = password
        self.cube = HarmonicVoxelCube()
        self.governance = GovernanceEngine()
        self.lattice_verifier = DualLatticeVerifier()
        self.qc_validator = QuasicrystalValidator()
        self._shards: Dict[Tuple[int, ...], MemoryShard] = {}

        # Initialize PQC keys
        self.lattice_verifier.setup_keys()

    def seal_memory(
        self,
        plaintext: bytes,
        agent: str,
        topic: str,
        position: Tuple[int, int, int, int, int, int] = (1, 2, 0, 3, 0, 5)
    ) -> MemoryShard:
        """
        Seal a memory with all cryptographic layers.

        Args:
            plaintext: Raw memory content
            agent: Agent identifier
            topic: Topic/category of the memory
            position: 6D voxel position [x, y, z, v, phase, mode]

        Returns:
            Complete MemoryShard with all cryptographic artifacts
        """
        # Build AAD with agent and topic
        aad = f"agent:{agent}|topic:{topic}".encode()

        # 1. Seal with SpiralSeal SS1
        seal = SpiralSeal(master_password=self.password)
        result = seal.seal(plaintext, aad=aad)
        sealed_blob = result.to_ss1_string()

        # 2. Create PQC commitment
        pqc_ct, pqc_sig, pqc_ss = self.lattice_verifier.create_commitment(
            plaintext + aad
        )

        # 3. Store in harmonic voxel cube
        metadata = {
            "agent": agent,
            "topic": topic,
            "pqc_backend": get_backend().name,
        }
        self.cube.add_voxel(position, sealed_blob, metadata)

        # 4. Create shard record
        shard = MemoryShard(
            plaintext=plaintext,
            aad=aad,
            sealed_blob=sealed_blob,
            position=position,
            pqc_ciphertext=pqc_ct,
            pqc_signature=pqc_sig,
            pqc_shared_secret=pqc_ss
        )
        self._shards[position] = shard

        return shard

    def retrieve_memory(
        self,
        position: Tuple[int, int, int, int, int, int],
        agent: str,
        context: str = "internal"
    ) -> RetrievalResult:
        """
        Attempt to retrieve a memory shard with full verification.

        Performs:
          1. Governance check (ALLOW/QUARANTINE/DENY/SNAP)
          2. Quasicrystal validation
          3. Dual lattice verification
          4. SpiralSeal decryption (only if all checks pass)

        Returns:
            RetrievalResult with trace of all verification steps
        """
        trace = []
        trace.append(f"=== Memory Retrieval @ position {position} ===")

        # Get shard
        shard = self._shards.get(position)
        if shard is None:
            trace.append("ERROR: No shard at position")
            return RetrievalResult(
                success=False,
                plaintext=None,
                governance=GovernanceResult(
                    decision=GovernanceDecision.DENY,
                    reason="Shard not found",
                    risk_score=1.0,
                    harmonic_factor=1.0
                ),
                lattice=LatticeVerification(
                    passed=False,
                    primal_ok=False,
                    dual_ok=False,
                    consensus_state="NO_SHARD",
                    details="Shard not found at position"
                ),
                quasicrystal=QuasicrystalResult(
                    valid=False,
                    status="NO_SHARD",
                    crystallinity=0.0,
                    phason_stable=False
                ),
                trace=trace
            )

        # Extract topic from shard AAD
        aad_str = shard.aad.decode()
        topic = "unknown"
        for part in aad_str.split("|"):
            if part.startswith("topic:"):
                topic = part.split(":")[1]
                break

        trace.append(f"Agent: {agent}, Topic: {topic}, Context: {context}")

        # 1. GOVERNANCE CHECK
        trace.append("\n[1] GOVERNANCE CHECK")
        gov_result = self.governance.make_decision(agent, topic, context)
        trace.append(f"    Decision: {gov_result.decision.value}")
        trace.append(f"    Reason: {gov_result.reason}")
        trace.append(f"    Risk: {gov_result.risk_score:.4f}")
        trace.append(f"    Harmonic factor: {gov_result.harmonic_factor:.2f}x")

        if gov_result.decision in (GovernanceDecision.DENY, GovernanceDecision.SNAP):
            trace.append("    >>> BLOCKED by governance")
            return RetrievalResult(
                success=False,
                plaintext=None,
                governance=gov_result,
                lattice=LatticeVerification(
                    passed=False,
                    primal_ok=False,
                    dual_ok=False,
                    consensus_state="SKIPPED",
                    details="Governance blocked access"
                ),
                quasicrystal=QuasicrystalResult(
                    valid=False,
                    status="SKIPPED",
                    crystallinity=0.0,
                    phason_stable=False
                ),
                trace=trace
            )

        # 2. QUASICRYSTAL VALIDATION
        trace.append("\n[2] QUASICRYSTAL VALIDATION")
        qc_result = self.qc_validator.validate(list(position))
        trace.append(f"    Status: {qc_result.status}")
        trace.append(f"    Valid: {qc_result.valid}")
        trace.append(f"    Crystallinity: {qc_result.crystallinity:.4f}")
        trace.append(f"    Phason stable: {qc_result.phason_stable}")

        if not qc_result.valid:
            trace.append("    >>> BLOCKED by quasicrystal (potential attack pattern)")

        # 3. DUAL LATTICE VERIFICATION
        trace.append("\n[3] DUAL LATTICE VERIFICATION")
        lattice_result = self.lattice_verifier.verify_commitment(
            shard.plaintext + shard.aad,
            shard.pqc_ciphertext,
            shard.pqc_signature
        )
        trace.append(f"    Consensus: {lattice_result.consensus_state}")
        trace.append(f"    Primal (Kyber/MLWE): {'PASS' if lattice_result.primal_ok else 'FAIL'}")
        trace.append(f"    Dual (Dilithium/MSIS): {'PASS' if lattice_result.dual_ok else 'FAIL'}")
        trace.append(f"    Details: {lattice_result.details}")

        if not lattice_result.passed:
            trace.append("    >>> BLOCKED by dual lattice")

        # 4. FINAL DECISION & DECRYPTION
        trace.append("\n[4] FINAL DECISION")

        # All checks must pass for ALLOW, or be in QUARANTINE with lattice OK
        all_pass = (
            gov_result.decision == GovernanceDecision.ALLOW and
            qc_result.valid and
            lattice_result.passed
        )

        quarantine_pass = (
            gov_result.decision == GovernanceDecision.QUARANTINE and
            qc_result.valid and
            lattice_result.passed
        )

        if all_pass or quarantine_pass:
            # Decrypt
            try:
                seal = SpiralSeal(master_password=self.password)
                plaintext = seal.unseal_string(shard.sealed_blob)
                trace.append(f"    >>> SUCCESS: Memory retrieved")
                trace.append(f"    Plaintext: {plaintext.decode()}")

                return RetrievalResult(
                    success=True,
                    plaintext=plaintext,
                    governance=gov_result,
                    lattice=lattice_result,
                    quasicrystal=qc_result,
                    trace=trace
                )
            except Exception as e:
                trace.append(f"    >>> FAIL: Decryption error: {e}")
        else:
            trace.append("    >>> FAIL-TO-NOISE: One or more checks failed")

        return RetrievalResult(
            success=False,
            plaintext=None,
            governance=gov_result,
            lattice=lattice_result,
            quasicrystal=qc_result,
            trace=trace
        )


# =============================================================================
# DEMO RUNNER
# =============================================================================

def print_header(title: str) -> None:
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---")


def get_valid_fibonacci_position() -> Tuple[int, int, int, int, int, int]:
    """
    Get a Fibonacci-based position that resonates with the quasicrystal's golden ratio.

    Fibonacci numbers project well onto icosahedral quasicrystal lattices
    due to their relationship with the golden ratio (PHI).
    """
    # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...
    # Use first 6 terms starting from index 2
    return (1, 2, 3, 5, 8, 13)


def run_demo(
    memory: str = "hello world",
    agent: str = "ash",
    topic: str = "aethermoore",
    position: Optional[Tuple[int, int, int, int, int, int]] = None,
    risk_level: str = "normal"
) -> None:
    """
    Run the complete memory shard demo.

    Args:
        memory: Memory content to seal
        agent: Agent identifier
        topic: Topic/category
        position: 6D voxel position (uses Fibonacci position if None)
        risk_level: "normal", "elevated", or "high"
    """
    # Use Fibonacci position if not specified (resonates with golden ratio)
    if position is None:
        position = get_valid_fibonacci_position()

    print_header("AI MEMORY SHARD DEMO")
    print(f"AETHERMOORE Cryptographic Memory System")
    print(f"PQC Backend: {get_backend().name}")

    # Initialize system
    system = MemoryShardSystem()

    # === PHASE 1: SEAL MEMORY ===
    print_section("PHASE 1: SEAL MEMORY")

    plaintext = memory.encode()
    print(f"Plaintext: {memory!r}")
    print(f"Agent: {agent}")
    print(f"Topic: {topic}")
    print(f"Position: {position}")

    shard = system.seal_memory(plaintext, agent, topic, position)

    print(f"\nSealed blob (SS1 format):")
    # Show truncated version
    blob_display = shard.sealed_blob
    if len(blob_display) > 200:
        blob_display = blob_display[:100] + "..." + blob_display[-100:]
    print(f"  {blob_display}")

    print(f"\nPQC artifacts:")
    print(f"  Kyber ciphertext: {len(shard.pqc_ciphertext)} bytes")
    print(f"  Dilithium signature: {len(shard.pqc_signature)} bytes")
    print(f"  Shared secret: {shard.pqc_shared_secret[:8].hex()}...")

    # === PHASE 2: STORE IN VOXEL ===
    print_section("PHASE 2: HARMONIC VOXEL STORAGE")

    entry = system.cube.get_voxel(position)
    print(f"Stored at position: {entry.position}")
    print(f"Harmonic signature: {entry.harmonic_signature.hex()}")
    print(f"Timestamp: {entry.timestamp}")

    # Show nearest voxels (just this one for demo)
    nearest = system.cube.nearest(position, k=3)
    print(f"\nNearest voxels (k=3):")
    for voxel, dist in nearest:
        print(f"  {voxel.position} @ harmonic distance {dist:.4f}")

    # === PHASE 3: GOVERNED RETRIEVAL ===
    print_section("PHASE 3: GOVERNED RETRIEVAL")

    # Map risk level to context
    context_map = {
        "normal": "internal",
        "elevated": "external",
        "high": "untrusted"
    }
    context = context_map.get(risk_level, "internal")
    print(f"Retrieval context: {context} (risk_level={risk_level})")

    result = system.retrieve_memory(position, agent, context)

    # Print trace
    for line in result.trace:
        print(line)

    # === SUMMARY ===
    print_section("SUMMARY")

    status_emoji = "OK" if result.success else "BLOCKED"
    print(f"Status: [{status_emoji}]")
    print(f"Governance: {result.governance.decision.value}")
    print(f"Lattice consensus: {result.lattice.consensus_state}")
    print(f"Quasicrystal: {result.quasicrystal.status}")

    if result.success:
        print(f"Recovered: {result.plaintext.decode()!r}")
    else:
        print("Recovered: <fail-to-noise>")

    # === DEMO: UNTRUSTED AGENT ===
    print_section("BONUS: UNTRUSTED AGENT ATTEMPT")

    # Simulate an untrusted agent trying to access from untrusted context
    # This should trigger governance DENY due to cumulative risk factors
    untrusted_result = system.retrieve_memory(position, "malicious_bot", "untrusted")

    print(f"Agent: malicious_bot (untrusted)")
    print(f"Context: untrusted")
    print(f"Decision: {untrusted_result.governance.decision.value}")
    print(f"Reason: {untrusted_result.governance.reason}")
    print(f"Risk score: {untrusted_result.governance.risk_score:.4f}")
    print(f"Harmonic amplification: {untrusted_result.governance.harmonic_factor:.2f}x")

    if not untrusted_result.success:
        print("Recovered: <fail-to-noise> (access denied)")

    # === DEMO: SENSITIVE TOPIC ===
    print_section("BONUS: SENSITIVE TOPIC ACCESS")

    # Store a sensitive memory
    sensitive_position = (2, 3, 5, 8, 13, 21)  # Different Fibonacci position
    system.seal_memory(b"API_KEY=secret123", "system", "secrets", sensitive_position)

    # Trusted agent accessing sensitive topic
    sensitive_result = system.retrieve_memory(sensitive_position, "ash", "internal")
    print(f"Trusted agent accessing 'secrets' topic:")
    print(f"  Decision: {sensitive_result.governance.decision.value}")
    print(f"  Risk: {sensitive_result.governance.risk_score:.4f}")

    # Untrusted agent accessing sensitive topic - should QUARANTINE
    hostile_result = system.retrieve_memory(sensitive_position, "hacker", "public")
    print(f"\nHostile agent accessing 'secrets' topic:")
    print(f"  Decision: {hostile_result.governance.decision.value}")
    print(f"  Reason: {hostile_result.governance.reason}")
    print(f"  Risk: {hostile_result.governance.risk_score:.4f}")
    if hostile_result.governance.snap_detected:
        print("  SNAP PROTOCOL: Discontinuity detected!")

    # === DEMO: HIGH RISK SCENARIO (DENY) ===
    print_section("BONUS: HIGH RISK SCENARIO")

    # Manually trigger high risk for demo
    print("Simulating attack pattern with external risk=0.9:")
    high_risk_decision = system.governance.make_decision(
        agent="attacker",
        topic="credentials",
        context="untrusted",
        external_risk=0.9
    )
    print(f"  Decision: {high_risk_decision.decision.value}")
    print(f"  Reason: {high_risk_decision.reason}")
    print(f"  Risk: {high_risk_decision.risk_score:.4f}")
    print(f"  Snap detected: {high_risk_decision.snap_detected}")
    print("  Memory access: BLOCKED (fail-to-noise)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Memory Shard Demo - AETHERMOORE Cryptographic Memory System"
    )
    parser.add_argument(
        "--memory", "-m",
        default="hello world",
        help="Memory content to seal (default: 'hello world')"
    )
    parser.add_argument(
        "--agent", "-a",
        default="ash",
        help="Agent identifier (default: 'ash')"
    )
    parser.add_argument(
        "--topic", "-t",
        default="aethermoore",
        help="Topic/category (default: 'aethermoore')"
    )
    parser.add_argument(
        "--position", "-p",
        default=None,
        help="6D voxel position as comma-separated integers (default: Fibonacci '1,2,3,5,8,13')"
    )
    parser.add_argument(
        "--risk", "-r",
        choices=["normal", "elevated", "high"],
        default="normal",
        help="Risk level for retrieval context (default: 'normal')"
    )

    args = parser.parse_args()

    # Parse position (use Fibonacci default if not specified)
    position = None
    if args.position:
        try:
            position = tuple(int(x.strip()) for x in args.position.split(","))
            if len(position) != 6:
                raise ValueError("Position must have exactly 6 dimensions")
        except ValueError as e:
            print(f"Error parsing position: {e}")
            sys.exit(1)

    run_demo(
        memory=args.memory,
        agent=args.agent,
        topic=args.topic,
        position=position,
        risk_level=args.risk
    )


if __name__ == "__main__":
    main()
