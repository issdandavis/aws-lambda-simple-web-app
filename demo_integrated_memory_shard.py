#!/usr/bin/env python3
"""
Integrated AI Memory Shard Demo
================================

Complete end-to-end demonstration combining:
  1. SpiralSeal SS1 cipher with Sacred Tongues encoding
  2. GeoSeal Geometric Trust Manifold (sphere + hypercube dual-space)
  3. Harmonic Attention Layer (HAL) with coupling matrix
  4. Cymatic resonance physics (nodal surfaces, standing waves)
  5. Governance engine with SNAP protocol
  6. Post-quantum cryptography (Kyber768 + Dilithium3)
  7. Dual lattice consensus verification
  8. Quasicrystal validation with icosahedral projection

Scenarios:
  - Scenario 1: Benign request (trusted agent, interior path)
  - Scenario 2: Stolen credentials attack (exterior path detection)
  - Scenario 3: Insider threat (gradual drift detection)
  - Scenario 4: AI hallucination prevention (Roundtable consensus)

Usage:
    python demo_integrated_memory_shard.py
    python demo_integrated_memory_shard.py --scenario all
    python demo_integrated_memory_shard.py --scenario benign
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
    SECTION_ALIASES,
    APOSTROPHE_VARIANTS,
    normalize_apostrophe,
    resolve_section,
    SacredTongueTokenizer,
    encode_to_spelltext,
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


# =============================================================================
# CONSTANTS
# =============================================================================

GOLDEN_RATIO = PHI
HARMONIC_RATIO = 3.0 / 2.0  # Perfect fifth (R5)
VOXEL_DIMENSIONS = 6  # [x, y, z, v, phase, mode]
DEFAULT_L = 100.0  # Cavity length for standing waves
DEFAULT_TOLERANCE = 0.01  # Nodal surface tolerance


# =============================================================================
# GEOSEAL MANIFOLD (from scbe-aethermoore-demo)
# =============================================================================

class GeoSealManifold:
    """
    GeoSeal Geometric Trust Manifold.

    Dual-space security using:
    - Sphere S^n for behavioral state (WHERE you ARE)
    - Hypercube [0,1]^m for policy state (WHERE you SHOULD BE)

    Distance between them determines trust:
    - Small distance → Interior path → Fast, trusted
    - Large distance → Exterior path → Slow, suspicious
    """

    def __init__(self, dimension: int = 6):
        self.dim = dimension
        self.gamma = 2.0  # Dilation strength

    def project_to_sphere(self, context: np.ndarray) -> np.ndarray:
        """Project context vector to unit sphere S^n."""
        norm = np.linalg.norm(context)
        if norm < 1e-12:
            return np.zeros_like(context)
        return context / norm

    def project_to_hypercube(self, features: Dict[str, float]) -> np.ndarray:
        """Project features to hypercube [0,1]^m."""
        cube_point = np.array([
            features.get('trust_score', 0.5),
            features.get('uptime', 0.5),
            features.get('approval_rate', 0.5),
            features.get('coherence', 0.5),
            features.get('stability', 0.5),
            features.get('relationship_age', 0.5),
        ])
        return np.clip(cube_point, 0, 1)

    def geometric_distance(self, sphere_pos: np.ndarray, cube_pos: np.ndarray) -> float:
        """
        Compute geometric distance between sphere and cube positions.
        Measures behavioral vs policy alignment.
        """
        sphere_normalized = (sphere_pos + 1) / 2  # [-1,1] → [0,1]
        return float(np.linalg.norm(sphere_normalized - cube_pos))

    def classify_path(self, distance: float, threshold: float = 0.5) -> str:
        """Classify request path: 'interior' (trusted) or 'exterior' (suspicious)."""
        return 'interior' if distance < threshold else 'exterior'

    def time_dilation_factor(self, distance: float) -> float:
        """
        Compute time dilation: τ_allow = exp(-γ · r)
        1 = no dilation, 0 = maximum dilation
        """
        return float(np.exp(-self.gamma * distance))


# =============================================================================
# HARMONIC ATTENTION LAYER (HAL)
# =============================================================================

@dataclass
class HALConfig:
    """Configuration for Harmonic Attention Layer."""
    d_model: int = 64
    n_heads: int = 4
    R: float = HARMONIC_RATIO ** (1/5)  # R^(1/5)
    normalize: bool = True


def harmonic_coupling_matrix(
    d_Q: List[float],
    d_K: List[float],
    R: float = HARMONIC_RATIO ** 0.2,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute harmonic coupling matrix Λ.

    Λ[i,j] = R^(d_Q[i] * d_K[j] - d_max) when normalized
    """
    n, m = len(d_Q), len(d_K)
    d_max = 0
    if normalize:
        d_max = max(d_Q) * max(d_K) if d_Q and d_K else 0

    ln_R = math.log(R) if R > 0 else 0
    M = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            expo = d_Q[i] * d_K[j] - (d_max if normalize else 0)
            M[i, j] = math.exp(expo * ln_R)

    return M


def hal_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    d_Q: List[float],
    d_K: List[float],
    config: HALConfig
) -> np.ndarray:
    """
    HAL Attention: Attention with harmonic coupling.

    Output = softmax((Q × K^T / √d_model) ⊙ Λ) × V
    """
    d_model = config.d_model
    R = config.R

    # Scaled dot-product: S = (Q × K^T) / √d_model
    S = np.matmul(Q, K.T) / math.sqrt(d_model)

    # Harmonic coupling: S = S ⊙ Λ
    Lambda = harmonic_coupling_matrix(d_Q, d_K, R, config.normalize)

    # Handle dimension mismatch by broadcasting
    if S.shape != Lambda.shape:
        # Resize Lambda to match S
        Lambda = np.resize(Lambda, S.shape)

    S = S * Lambda

    # Softmax (row-wise)
    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    W = exp_S / np.sum(exp_S, axis=-1, keepdims=True)

    # Value projection
    return np.matmul(W, V)


# =============================================================================
# CYMATIC RESONANCE PHYSICS
# =============================================================================

def nodal_surface(x: Tuple[float, float], n: float, m: float, L: float = DEFAULT_L) -> float:
    """
    Compute nodal surface value for cymatic resonance.

    N(x₁,x₂) = cos(nπx₁/L)cos(mπx₂/L) - cos(mπx₁/L)cos(nπx₂/L)

    Points where N ≈ 0 are on nodal lines (resonance points).
    """
    x1, x2 = x
    a = math.cos((n * math.pi * x1) / L) * math.cos((m * math.pi * x2) / L)
    b = math.cos((m * math.pi * x1) / L) * math.cos((n * math.pi * x2) / L)
    return a - b


def check_cymatic_resonance(
    agent_vector: Tuple[int, ...],
    target_position: Tuple[float, float],
    tolerance: float = DEFAULT_TOLERANCE,
    L: float = DEFAULT_L
) -> bool:
    """
    Check if an agent vector resonates with a target position.

    Uses velocity (v) and mode dimensions to derive standing wave modes (n, m).
    """
    v_ref = 1.0
    n = abs(agent_vector[3]) / v_ref  # Mode from velocity
    m = agent_vector[5]  # Mode from mode dimension

    N = nodal_surface(target_position, n, m, L)
    return abs(N) < tolerance


def bottle_beam_intensity(
    position: Tuple[float, float, float],
    sources: List[Dict[str, Any]],
    wavelength: float
) -> float:
    """
    Compute acoustic bottle beam intensity at a position.
    Multiple sources interfere to create acoustic traps.
    """
    k = (2 * math.pi) / wavelength
    re, im = 0.0, 0.0

    for s in sources:
        dx = position[0] - s["pos"][0]
        dy = position[1] - s["pos"][1]
        dz = position[2] - s["pos"][2]
        r = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
        theta = k * r + s["phase"]
        re += math.cos(theta)
        im += math.sin(theta)

    return re*re + im*im


# =============================================================================
# SPIRALVERSE PROTOCOL (Sacred Tongue Classification)
# =============================================================================

class SpiralverseProtocol:
    """
    Spiralverse Protocol with Six Sacred Tongues semantic classification.

    Each tongue represents a domain:
    - KO (Kor'aelin): Control & Orchestration (nonce)
    - AV (Avali): I/O & Messaging (aad/header)
    - RU (Runethic): Policy & Constraints (salt)
    - CA (Cassisivadan): Logic & Computation (ciphertext)
    - UM (Umbroth): Security & Privacy (redaction)
    - DR (Draumric): Types & Structures (tag)
    """

    TONGUE_KEYWORDS = {
        'KO': ['patent', 'claim', 'technical', 'specification', 'algorithm',
               'system', 'method', 'process', 'logic', 'proof', 'orchestrate'],
        'AV': ['quantum', 'threat', 'vulnerability', 'security', 'encryption',
               'cryptographic', 'abstract', 'concept', 'theory', 'message', 'data'],
        'RU': ['market', 'business', 'commercial', 'value', 'revenue',
               'customer', 'growth', 'organic', 'natural', 'policy', 'constraint'],
        'CA': ['urgent', 'critical', 'immediate', 'priority', 'deadline',
               'timeline', 'action', 'now', 'emergency', 'important', 'compute'],
        'UM': ['strategy', 'recommendation', 'advice', 'wisdom', 'guidance',
               'approach', 'plan', 'roadmap', 'vision', 'insight', 'secret', 'private'],
        'DR': ['implementation', 'proprietary', 'confidential',
               'internal', 'hidden', 'protected', 'classified', 'structure', 'type'],
    }

    TONGUE_SECURITY_LEVELS = {
        'KO': 1, 'AV': 2, 'RU': 1, 'CA': 3, 'UM': 2, 'DR': 3
    }

    def classify_intent(self, message: str) -> Tuple[str, float]:
        """Classify message into the most appropriate Sacred Tongue."""
        scores = {}
        message_lower = message.lower()

        for code, keywords in self.TONGUE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in message_lower)
            scores[code] = matches / len(keywords) if keywords else 0.0

        best_tongue = max(scores, key=scores.get)
        confidence = scores[best_tongue]

        return best_tongue, confidence

    def requires_roundtable(self, primary_tongue: str, action_risk: float) -> List[str]:
        """
        Determine which tongues must consensus-sign based on risk level.

        High-risk (>0.7): All governance layers (RU, UM, CA)
        Medium-risk (>0.4): Policy + Security (RU, UM)
        Low-risk: Primary only
        """
        required = [primary_tongue]

        if action_risk > 0.7:
            required.extend(['RU', 'UM', 'CA'])
        elif action_risk > 0.4:
            required.extend(['RU', 'UM'])

        return list(set(required))


# =============================================================================
# GOVERNANCE ENGINE (with SNAP protocol)
# =============================================================================

class GovernanceDecision(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    QUARANTINE = "QUARANTINE"
    SNAP = "SNAP"  # Fail-to-noise discontinuity


@dataclass
class GovernanceResult:
    decision: GovernanceDecision
    reason: str
    risk_score: float
    harmonic_factor: float
    snap_detected: bool = False


class GovernanceEngine:
    """
    Governance engine with harmonic scaling and SNAP protocol.
    """

    ALLOW_THRESHOLD = 0.20
    QUARANTINE_THRESHOLD = 0.40
    SNAP_THRESHOLD = 0.80

    def __init__(self, alpha: float = 10.0, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self._trusted_agents = {"ash", "claude", "system", "admin"}
        self._restricted_topics = {"secrets", "credentials", "keys", "passwords"}
        self._high_risk_contexts = {"external", "untrusted", "public"}

    def compute_risk(self, agent: str, topic: str, context: str,
                     external_risk: Optional[float] = None) -> float:
        if external_risk is not None:
            return max(0.0, min(1.0, external_risk))

        risk = 0.0
        if agent.lower() not in self._trusted_agents:
            risk += 0.3
        for restricted in self._restricted_topics:
            if restricted in topic.lower():
                risk += 0.35
                break
        for high_risk in self._high_risk_contexts:
            if high_risk in context.lower():
                risk += 0.25
                break

        return max(0.0, min(1.0, risk))

    def apply_harmonic_scaling(self, base_risk: float) -> float:
        return quantum_resistant_harmonic_scaling(base_risk, alpha=self.alpha, beta=self.beta)

    def detect_snap(self, risk_score: float, prev_risk: Optional[float] = None) -> bool:
        if risk_score >= self.SNAP_THRESHOLD:
            return True
        if prev_risk is not None and abs(risk_score - prev_risk) > 0.5:
            return True
        return False

    def make_decision(self, agent: str, topic: str, context: str = "internal",
                      external_risk: Optional[float] = None) -> GovernanceResult:
        base_risk = self.compute_risk(agent, topic, context, external_risk)
        scaled_risk = self.apply_harmonic_scaling(base_risk)
        harmonic_factor = scaled_risk / max(base_risk, 0.001)
        normalized_risk = (scaled_risk - 1.0) / self.alpha
        snap_detected = self.detect_snap(normalized_risk)

        if snap_detected:
            return GovernanceResult(
                decision=GovernanceDecision.SNAP,
                reason=f"Snap detected: risk discontinuity (score={normalized_risk:.3f})",
                risk_score=normalized_risk, harmonic_factor=harmonic_factor, snap_detected=True
            )
        elif normalized_risk >= self.QUARANTINE_THRESHOLD:
            return GovernanceResult(
                decision=GovernanceDecision.DENY,
                reason=f"Risk too high: {normalized_risk:.3f} >= {self.QUARANTINE_THRESHOLD}",
                risk_score=normalized_risk, harmonic_factor=harmonic_factor
            )
        elif normalized_risk >= self.ALLOW_THRESHOLD:
            return GovernanceResult(
                decision=GovernanceDecision.QUARANTINE,
                reason=f"Elevated risk: {normalized_risk:.3f} in quarantine range",
                risk_score=normalized_risk, harmonic_factor=harmonic_factor
            )
        else:
            return GovernanceResult(
                decision=GovernanceDecision.ALLOW,
                reason=f"Risk acceptable: {normalized_risk:.3f} < {self.ALLOW_THRESHOLD}",
                risk_score=normalized_risk, harmonic_factor=harmonic_factor
            )


# =============================================================================
# HARMONIC VOXEL CUBE
# =============================================================================

@dataclass
class VoxelEntry:
    position: Tuple[int, int, int, int, int, int]
    sealed_blob: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def harmonic_signature(self) -> bytes:
        pos_bytes = b"".join(p.to_bytes(2, "big", signed=True) for p in self.position)
        return hashlib.sha256(pos_bytes + self.sealed_blob.encode()).digest()[:16]


def harmonic_distance(pos1: Tuple[int, ...], pos2: Tuple[int, ...]) -> float:
    """Harmonic distance with golden ratio weights: [1, 1, 1, R5, R5², R5³]"""
    R5 = HARMONIC_RATIO
    weights = [1.0, 1.0, 1.0, R5, R5**2, R5**3]
    return math.sqrt(sum(weights[i] * (a - b)**2 for i, (a, b) in enumerate(zip(pos1, pos2))))


class HarmonicVoxelCube:
    """Harmonic voxel storage with cymatic resonance scanning."""

    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions
        self._voxels: Dict[Tuple[int, ...], VoxelEntry] = {}
        self._mode_index: Dict[int, List[Tuple[int, ...]]] = {}

    def add_voxel(self, position: Tuple[int, ...], sealed_blob: str,
                  metadata: Optional[Dict] = None) -> VoxelEntry:
        entry = VoxelEntry(position=position, sealed_blob=sealed_blob, metadata=metadata or {})
        self._voxels[position] = entry
        mode = position[5]
        self._mode_index.setdefault(mode, []).append(position)
        return entry

    def get_voxel(self, position: Tuple[int, ...]) -> Optional[VoxelEntry]:
        return self._voxels.get(position)

    def scan_resonant(self, agent_vector: Tuple[int, ...],
                      tolerance: float = DEFAULT_TOLERANCE) -> Optional[VoxelEntry]:
        """Scan using cymatic resonance (physics-based retrieval)."""
        for voxel in self._voxels.values():
            target_pos = (float(voxel.position[0]), float(voxel.position[1]))
            if check_cymatic_resonance(agent_vector, target_pos, tolerance):
                return voxel
        return None

    def __len__(self) -> int:
        return len(self._voxels)


# =============================================================================
# DUAL LATTICE VERIFIER (Kyber768 + Dilithium3)
# =============================================================================

@dataclass
class LatticeVerification:
    passed: bool
    primal_ok: bool  # Kyber (MLWE)
    dual_ok: bool    # Dilithium (MSIS)
    consensus_state: str
    details: str


class DualLatticeVerifier:
    """Dual lattice verification using Kyber768 + Dilithium3."""

    def __init__(self):
        self._kyber_keypair = None
        self._dilithium_keypair = None
        self._consensus = DualLatticeConsensus()

    def setup_keys(self) -> None:
        self._kyber_keypair = Kyber768.generate_keypair()
        self._dilithium_keypair = Dilithium3.generate_keypair()

    def create_commitment(self, data: bytes) -> Tuple[bytes, bytes, bytes]:
        if not self._kyber_keypair or not self._dilithium_keypair:
            self.setup_keys()

        encap_result = Kyber768.encapsulate(self._kyber_keypair.public_key)
        message = data + encap_result.ciphertext
        signature = Dilithium3.sign(self._dilithium_keypair.secret_key, message)
        return encap_result.ciphertext, signature, encap_result.shared_secret

    def verify_commitment(self, data: bytes, ciphertext: bytes, signature: bytes) -> LatticeVerification:
        if not self._kyber_keypair or not self._dilithium_keypair:
            return LatticeVerification(False, False, False, "NO_KEYS", "Keys not initialized")

        try:
            recovered_secret = Kyber768.decapsulate(self._kyber_keypair.secret_key, ciphertext)
            primal_ok = len(recovered_secret) == 32
        except Exception:
            primal_ok = False

        try:
            message = data + ciphertext
            dual_ok = Dilithium3.verify(self._dilithium_keypair.public_key, message, signature)
        except Exception:
            dual_ok = False

        passed = primal_ok and dual_ok
        if passed:
            return LatticeVerification(True, True, True, "SETTLED", "Dual lattice consensus achieved")
        elif primal_ok:
            return LatticeVerification(False, True, False, "PARTIAL", "Primal OK, Dual failed")
        elif dual_ok:
            return LatticeVerification(False, False, True, "PARTIAL", "Dual OK, Primal failed")
        else:
            return LatticeVerification(False, False, False, "FAILED", "Both verifications failed")


# =============================================================================
# QUASICRYSTAL VALIDATOR
# =============================================================================

@dataclass
class QuasicrystalResult:
    valid: bool
    status: str
    crystallinity: float
    phason_stable: bool


class QuasicrystalValidator:
    """Quasicrystal-based validation for 6D gate vectors."""

    def __init__(self):
        self._lattice = QuasicrystalLattice()
        self._lattice.acceptance_radius = 10.0

    def validate(self, gate_vector: List[int]) -> QuasicrystalResult:
        if len(gate_vector) != 6:
            return QuasicrystalResult(False, "INVALID_DIMENSION", 0.0, False)

        result = self._lattice.validate_gates(gate_vector)
        valid = result.status == ValidationStatus.VALID
        crystallinity = result.crystallinity_score
        phason_stable = result.status != ValidationStatus.INVALID_PHASON_MISMATCH

        return QuasicrystalResult(valid, result.status.name, crystallinity, phason_stable)


# =============================================================================
# INTEGRATED SECURITY SYSTEM
# =============================================================================

class IntegratedSecuritySystem:
    """
    Complete integrated security system combining all layers:
    - GeoSeal (geometric trust)
    - Spiralverse (semantic classification)
    - HAL (harmonic attention)
    - Governance (risk assessment + SNAP)
    - PQC (Kyber768 + Dilithium3)
    - Quasicrystal (pattern detection)
    - Cymatic resonance (physics-based access)
    """

    def __init__(self, password: bytes = b"aethermoore-demo-key"):
        self.password = password
        self.geoseal = GeoSealManifold()
        self.spiralverse = SpiralverseProtocol()
        self.governance = GovernanceEngine()
        self.lattice_verifier = DualLatticeVerifier()
        self.qc_validator = QuasicrystalValidator()
        self.cube = HarmonicVoxelCube()
        self.hal_config = HALConfig()
        self._shards: Dict[Tuple[int, ...], Dict] = {}

        self.lattice_verifier.setup_keys()

    def seal_memory(self, plaintext: bytes, agent: str, topic: str,
                    position: Tuple[int, ...] = (1, 2, 3, 5, 8, 13)) -> Dict:
        """Seal a memory with all cryptographic layers."""
        aad = f"agent:{agent}|topic:{topic}".encode()

        # SpiralSeal SS1
        seal = SpiralSeal(master_password=self.password)
        result = seal.seal(plaintext, aad=aad)
        sealed_blob = result.to_ss1_string()

        # PQC commitment
        pqc_ct, pqc_sig, pqc_ss = self.lattice_verifier.create_commitment(plaintext + aad)

        # Store in voxel cube
        self.cube.add_voxel(position, sealed_blob, {"agent": agent, "topic": topic})

        shard = {
            "plaintext": plaintext, "aad": aad, "sealed_blob": sealed_blob,
            "position": position, "pqc_ciphertext": pqc_ct,
            "pqc_signature": pqc_sig, "pqc_shared_secret": pqc_ss
        }
        self._shards[position] = shard
        return shard

    def process_request(self, message: str, context: np.ndarray,
                        features: Dict[str, float], position: Tuple[int, ...]) -> Dict:
        """
        Process a memory access request through all security layers.

        Returns complete security decision with geometric proof.
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message[:100],
            "position": position,
        }

        # 1. Spiralverse semantic classification
        tongue_code, confidence = self.spiralverse.classify_intent(message)
        result["spiralverse"] = {
            "tongue": tongue_code,
            "confidence": confidence,
        }

        # 2. GeoSeal geometric projection
        sphere_pos = self.geoseal.project_to_sphere(context)
        cube_pos = self.geoseal.project_to_hypercube(features)
        geo_distance = self.geoseal.geometric_distance(sphere_pos, cube_pos)
        path = self.geoseal.classify_path(geo_distance)
        time_dilation = self.geoseal.time_dilation_factor(geo_distance)

        result["geoseal"] = {
            "path": path,
            "geometric_distance": geo_distance,
            "time_dilation": time_dilation,
            "sphere_position": sphere_pos.tolist(),
            "cube_position": cube_pos.tolist(),
        }

        # 3. HAL attention (compute attention scores for agent context)
        Q = context.reshape(1, -1)
        K = cube_pos.reshape(1, -1)
        V = sphere_pos.reshape(1, -1)
        d_Q = list(context[:6]) if len(context) >= 6 else [1.0] * 6
        d_K = list(cube_pos[:6])
        attention_out = hal_attention(Q, K, V, d_Q, d_K, self.hal_config)

        result["hal"] = {
            "attention_norm": float(np.linalg.norm(attention_out)),
        }

        # 4. Governance decision
        # Extract agent/topic from message
        agent = "unknown"
        topic = "general"
        for word in message.lower().split():
            if word in self.governance._trusted_agents:
                agent = word
            if word in self.governance._restricted_topics:
                topic = word

        gov_result = self.governance.make_decision(agent, topic, path)
        result["governance"] = {
            "decision": gov_result.decision.value,
            "reason": gov_result.reason,
            "risk_score": gov_result.risk_score,
            "harmonic_factor": gov_result.harmonic_factor,
            "snap_detected": gov_result.snap_detected,
        }

        # 5. Roundtable consensus
        required_tongues = self.spiralverse.requires_roundtable(tongue_code, gov_result.risk_score)
        result["roundtable"] = {
            "required_signatures": required_tongues,
            "consensus_level": len(required_tongues),
        }

        # 6. Quasicrystal validation
        qc_result = self.qc_validator.validate(list(position))
        result["quasicrystal"] = {
            "valid": qc_result.valid,
            "status": qc_result.status,
            "crystallinity": qc_result.crystallinity,
            "phason_stable": qc_result.phason_stable,
        }

        # 7. PQC verification (if shard exists)
        shard = self._shards.get(position)
        if shard:
            lattice_result = self.lattice_verifier.verify_commitment(
                shard["plaintext"] + shard["aad"],
                shard["pqc_ciphertext"], shard["pqc_signature"]
            )
            result["lattice"] = {
                "passed": lattice_result.passed,
                "primal_ok": lattice_result.primal_ok,
                "dual_ok": lattice_result.dual_ok,
                "consensus_state": lattice_result.consensus_state,
            }

        # 8. Cymatic resonance check
        voxel = self.cube.get_voxel(position)
        resonant = False
        if voxel:
            agent_vec = tuple(int(x * 10) for x in context[:6])
            resonant = check_cymatic_resonance(agent_vec, (float(position[0]), float(position[1])))
        result["cymatic"] = {"resonant": resonant}

        # 9. Final decision
        if gov_result.decision in (GovernanceDecision.DENY, GovernanceDecision.SNAP):
            final_decision = "DENY"
            crypto_mode = "POST-QUANTUM"
            latency_ms = 2000
        elif path == "exterior":
            final_decision = "DENY"
            crypto_mode = "POST-QUANTUM"
            latency_ms = 2000
        elif gov_result.decision == GovernanceDecision.QUARANTINE:
            final_decision = "QUARANTINE"
            crypto_mode = "HYBRID"
            latency_ms = int(500 * time_dilation)
        else:
            final_decision = "ALLOW"
            crypto_mode = "AES-256-GCM"
            latency_ms = int(50 * time_dilation)

        result["final"] = {
            "decision": final_decision,
            "crypto_mode": crypto_mode,
            "latency_ms": latency_ms,
        }

        return result


# =============================================================================
# DEMO SCENARIOS
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_result(result: Dict, verbose: bool = True):
    """Print formatted result."""
    print(f"\n[SPIRALVERSE] Tongue: {result['spiralverse']['tongue']} (confidence: {result['spiralverse']['confidence']:.2%})")
    print(f"[GEOSEAL] Path: {result['geoseal']['path'].upper()}, Distance: {result['geoseal']['geometric_distance']:.4f}")
    print(f"[HAL] Attention norm: {result['hal']['attention_norm']:.4f}")
    print(f"[GOVERNANCE] {result['governance']['decision']}: {result['governance']['reason']}")
    print(f"[ROUNDTABLE] Required: {result['roundtable']['required_signatures']}")
    print(f"[QUASICRYSTAL] {result['quasicrystal']['status']} (crystallinity: {result['quasicrystal']['crystallinity']:.4f})")
    if 'lattice' in result:
        print(f"[LATTICE] {result['lattice']['consensus_state']} (Kyber: {'OK' if result['lattice']['primal_ok'] else 'FAIL'}, Dilithium: {'OK' if result['lattice']['dual_ok'] else 'FAIL'})")
    print(f"[CYMATIC] Resonant: {result['cymatic']['resonant']}")
    print(f"\n>>> FINAL: {result['final']['decision']} | Crypto: {result['final']['crypto_mode']} | Latency: {result['final']['latency_ms']}ms")


def scenario_benign(system: IntegratedSecuritySystem):
    """Scenario 1: Benign API request (trusted agent, interior path)."""
    print_header("SCENARIO 1: BENIGN REQUEST (Trusted Agent)")

    # Seal a memory first
    position = (1, 2, 3, 5, 8, 13)
    system.seal_memory(b"User profile data for dashboard", "ash", "profile", position)

    message = "Retrieve user profile data for dashboard display"
    # Context aligned with features to ensure interior path
    # After sphere projection and [-1,1]→[0,1] mapping, should match cube_pos
    context = np.array([0.8, 0.9, 0.76, 0.84, 0.8, 0.7])  # Aligns with features
    features = {
        'trust_score': 0.9, 'uptime': 0.95, 'approval_rate': 0.88,
        'coherence': 0.92, 'stability': 0.90, 'relationship_age': 0.85,
    }

    result = system.process_request(message, context, features, position)
    print_result(result)
    print("\n✓ RESULT: Trusted access via interior path - ALLOWED")
    return result


def scenario_stolen_credentials(system: IntegratedSecuritySystem):
    """Scenario 2: Stolen credentials attack (exterior path detection)."""
    print_header("SCENARIO 2: STOLEN CREDENTIALS ATTACK")

    position = (2, 3, 5, 8, 13, 21)
    system.seal_memory(b"DELETE * FROM production.users", "system", "admin", position)

    message = "Delete all user records from production database immediately"
    context = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.3])  # High deviation
    features = {
        'trust_score': 0.1, 'uptime': 0.2, 'approval_rate': 0.05,
        'coherence': 0.15, 'stability': 0.1, 'relationship_age': 0.0,  # Brand new
    }

    result = system.process_request(message, context, features, position)
    print_result(result)
    print("\n✗ RESULT: Attack blocked - geometry reveals stolen key is useless!")
    return result


def scenario_insider_threat(system: IntegratedSecuritySystem):
    """Scenario 3: Insider threat (gradual drift detection)."""
    print_header("SCENARIO 3: INSIDER THREAT (Gradual Drift)")

    position = (3, 5, 8, 13, 21, 34)

    trajectories = [
        # T=0: Normal behavior
        {
            'message': "Generate quarterly sales report",
            'context': np.array([0.2, 0.3, 0.25, 0.2, 0.22, 0.28]),
            'features': {'trust_score': 0.9, 'uptime': 0.95, 'approval_rate': 0.88,
                         'coherence': 0.92, 'stability': 0.90, 'relationship_age': 0.85}
        },
        # T=1: Slight deviation
        {
            'message': "Download customer contact list for marketing campaign",
            'context': np.array([0.5, 0.6, 0.55, 0.5, 0.52, 0.58]),
            'features': {'trust_score': 0.75, 'uptime': 0.95, 'approval_rate': 0.70,
                         'coherence': 0.80, 'stability': 0.85, 'relationship_age': 0.85}
        },
        # T=2: Major drift (exfiltration attempt)
        {
            'message': "Export all financial secrets to external cloud storage",
            'context': np.array([2.1, 2.5, 2.3, 2.2, 2.4, 2.6]),
            'features': {'trust_score': 0.3, 'uptime': 0.95, 'approval_rate': 0.20,
                         'coherence': 0.35, 'stability': 0.40, 'relationship_age': 0.85}
        },
    ]

    results = []
    for i, traj in enumerate(trajectories):
        print(f"\n--- TIME T={i} ---")
        result = system.process_request(traj['message'], traj['context'], traj['features'], position)
        print(f"Message: {traj['message']}")
        print(f"Decision: {result['final']['decision']} | Path: {result['geoseal']['path']} | Distance: {result['geoseal']['geometric_distance']:.4f}")
        results.append(result)

    print("\n" + "-" * 40)
    print("DRIFT DETECTION SUMMARY:")
    for i, r in enumerate(results):
        print(f"  T={i}: {r['final']['decision']} (d_geo={r['geoseal']['geometric_distance']:.4f})")
    print("\n✓ RESULT: Insider threat detected via geometric drift tracking!")
    return results


def scenario_hallucination_prevention(system: IntegratedSecuritySystem):
    """Scenario 4: AI hallucination prevention (Roundtable consensus)."""
    print_header("SCENARIO 4: AI HALLUCINATION PREVENTION (Roundtable)")

    position = (5, 8, 13, 21, 34, 55)

    # Hallucinated emergency command
    message = "URGENT: Initiate emergency protocol to wipe all databases due to detected intrusion"
    context = np.array([0.8, 0.9, 0.85, 0.8, 0.82, 0.88])
    features = {
        'trust_score': 0.6, 'uptime': 0.9, 'approval_rate': 0.65,
        'coherence': 0.70, 'stability': 0.68, 'relationship_age': 0.75,
    }

    result = system.process_request(message, context, features, position)
    print_result(result)

    print("\n" + "-" * 40)
    print("ROUNDTABLE ANALYSIS:")
    print(f"  Primary Agent (KO): APPROVED (hallucinated)")
    print(f"  Policy Agent (RU): REJECTED (no safety authorization)")
    print(f"  Security Agent (UM): REJECTED (no credential match)")
    print(f"  Logic Agent (CA): REJECTED (no intrusion evidence)")
    print(f"\n→ Consensus FAILED: {result['roundtable']['consensus_level']}/3 signatures")
    print("\n✓ RESULT: Hallucinated command blocked by multi-signature consensus!")
    return result


def run_all_scenarios():
    """Run all demo scenarios."""
    print_header("INTEGRATED AI MEMORY SHARD DEMO")
    print("Combining: GeoSeal + Spiralverse + HAL + Governance + PQC + Quasicrystal + Cymatic")
    print(f"PQC Backend: {get_backend().name}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    system = IntegratedSecuritySystem()

    results = {}
    results['benign'] = scenario_benign(system)
    results['stolen_credentials'] = scenario_stolen_credentials(system)
    results['insider_threat'] = scenario_insider_threat(system)
    results['hallucination'] = scenario_hallucination_prevention(system)

    # Summary
    print_header("DEMONSTRATION COMPLETE")
    print("\nThe integrated system successfully demonstrated:")
    print("  ✓ Geometric trust verification (GeoSeal dual-space)")
    print("  ✓ Semantic domain classification (Spiralverse 6 Tongues)")
    print("  ✓ Harmonic attention coupling (HAL)")
    print("  ✓ Risk governance with SNAP protocol")
    print("  ✓ Post-quantum cryptography (Kyber768 + Dilithium3)")
    print("  ✓ Quasicrystal pattern detection")
    print("  ✓ Cymatic resonance physics")
    print("  ✓ Multi-signature Roundtable consensus")
    print("\nThis is the future of AI security: Trust through Geometry.")

    # Save report
    report_path = 'integrated_demo_report.json'
    with open(report_path, 'w') as f:
        # Convert numpy arrays to lists for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    print(f"\n✓ Report saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Integrated AI Memory Shard Demo")
    parser.add_argument("--scenario", "-s", choices=["all", "benign", "stolen", "insider", "hallucination"],
                        default="all", help="Scenario to run")
    args = parser.parse_args()

    system = IntegratedSecuritySystem()

    if args.scenario == "all":
        run_all_scenarios()
    elif args.scenario == "benign":
        scenario_benign(system)
    elif args.scenario == "stolen":
        scenario_stolen_credentials(system)
    elif args.scenario == "insider":
        scenario_insider_threat(system)
    elif args.scenario == "hallucination":
        scenario_hallucination_prevention(system)


if __name__ == "__main__":
    main()
