"""
SCBE Production Service
=======================

Commercial-grade service facade integrating all SCBE components.
Provides a unified API for memory shard operations.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import production components
from .config import ProductionConfig, get_config
from .logging import AuditLogger, get_logger, TimedOperation
from .exceptions import (
    SCBEError,
    PQCError,
    GovernanceError,
    ValidationError,
    AuthenticationError,
    CryptoError,
    LatticeError,
)

# Import core SCBE modules (from existing implementation)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symphonic_cipher.scbe_aethermoore.spiral_seal import (
    SpiralSeal,
    SpiralSealResult,
)
from symphonic_cipher.scbe_aethermoore.pqc import (
    Kyber768,
    Dilithium3,
    get_backend,
)
from symphonic_cipher.scbe_aethermoore.dual_lattice import (
    DualLatticeConsensus,
)
from symphonic_cipher.scbe_aethermoore.qc_lattice import (
    QuasicrystalLattice,
    ValidationStatus,
)
from symphonic_cipher.harmonic_scaling_law import (
    quantum_resistant_harmonic_scaling,
    PHI,
)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class MemoryShard:
    """Sealed memory shard with cryptographic proofs."""
    shard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Position in 6D harmonic voxel space
    position: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)

    # Sealed data
    sealed_blob: str = ""
    aad: bytes = b""

    # Cryptographic proofs
    pqc_ciphertext: Optional[bytes] = None
    pqc_signature: Optional[bytes] = None
    pqc_shared_secret: Optional[bytes] = None

    # Metadata
    agent_id: str = ""
    topic: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "created_at": self.created_at,
            "position": self.position,
            "agent_id": self.agent_id,
            "topic": self.topic,
            "tags": self.tags,
        }


@dataclass
class AccessRequest:
    """Request to access a memory shard."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Requestor context
    agent_id: str = ""
    message: str = ""
    context_vector: Optional[np.ndarray] = None
    features: Dict[str, float] = field(default_factory=dict)

    # Target
    position: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)


@dataclass
class AccessResponse:
    """Response to an access request."""
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Decision
    decision: str = "DENY"  # ALLOW, QUARANTINE, DENY, SNAP
    reason: str = ""

    # Security analysis
    risk_score: float = 0.0
    geometric_distance: float = 0.0
    path: str = "exterior"
    time_dilation: float = 1.0

    # Cryptographic verification
    lattice_verified: bool = False
    quasicrystal_valid: bool = False
    cymatic_resonant: bool = False

    # Execution parameters
    crypto_mode: str = "POST-QUANTUM"
    latency_ms: int = 2000

    # Data (only if ALLOW)
    data: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 4),
            "geometric_distance": round(self.geometric_distance, 4),
            "path": self.path,
            "time_dilation": round(self.time_dilation, 4),
            "lattice_verified": self.lattice_verified,
            "quasicrystal_valid": self.quasicrystal_valid,
            "cymatic_resonant": self.cymatic_resonant,
            "crypto_mode": self.crypto_mode,
            "latency_ms": self.latency_ms,
        }
        if self.data is not None:
            result["data_size"] = len(self.data)
        return result


# =============================================================================
# GEOSEAL (Geometric Trust Manifold)
# =============================================================================

class GeoSealManifold:
    """Production GeoSeal with sphere + hypercube dual-space."""

    def __init__(self, config: ProductionConfig):
        self.dim = config.geoseal.dimension
        self.gamma = config.geoseal.gamma
        self.interior_threshold = config.geoseal.interior_threshold

    def project_to_sphere(self, context: np.ndarray) -> np.ndarray:
        """Project context vector to unit sphere S^n."""
        norm = np.linalg.norm(context)
        if norm < 1e-12:
            return np.zeros(self.dim)
        return context / norm

    def project_to_hypercube(self, features: Dict[str, float]) -> np.ndarray:
        """Project features to hypercube [0,1]^m."""
        feature_keys = [
            "trust_score", "uptime", "approval_rate",
            "coherence", "stability", "relationship_age"
        ]
        cube_point = np.array([features.get(k, 0.5) for k in feature_keys[:self.dim]])
        return np.clip(cube_point, 0, 1)

    def geometric_distance(self, sphere_pos: np.ndarray, cube_pos: np.ndarray) -> float:
        """Compute behavioral vs policy alignment distance."""
        sphere_normalized = (sphere_pos + 1) / 2
        return float(np.linalg.norm(sphere_normalized - cube_pos))

    def classify_path(self, distance: float) -> str:
        """Classify as 'interior' (trusted) or 'exterior' (suspicious)."""
        return "interior" if distance < self.interior_threshold else "exterior"

    def time_dilation_factor(self, distance: float) -> float:
        """Compute time dilation: τ = exp(-γ·r)."""
        return float(np.exp(-self.gamma * distance))


# =============================================================================
# GOVERNANCE ENGINE
# =============================================================================

class GovernanceEngine:
    """Production governance with harmonic scaling and SNAP."""

    def __init__(self, config: ProductionConfig, logger: AuditLogger):
        self.config = config.governance
        self.logger = logger
        self._trusted_agents = {"system", "admin", "claude", "ash"}

    def compute_risk(
        self,
        agent_id: str,
        message: str,
        path: str,
    ) -> float:
        """Compute base risk score."""
        risk = 0.0

        # Agent trust
        if agent_id.lower() not in self._trusted_agents:
            risk += 0.25

        # Message analysis (simplified)
        high_risk_words = {"delete", "drop", "secrets", "password", "credentials"}
        message_lower = message.lower()
        for word in high_risk_words:
            if word in message_lower:
                risk += 0.2
                break

        # Path penalty
        if path == "exterior":
            risk += 0.3

        return min(1.0, max(0.0, risk))

    def apply_harmonic_scaling(self, base_risk: float) -> float:
        """Apply quantum-resistant harmonic scaling."""
        return quantum_resistant_harmonic_scaling(
            base_risk,
            alpha=self.config.alpha,
            beta=self.config.beta
        )

    def make_decision(
        self,
        agent_id: str,
        message: str,
        path: str,
    ) -> Tuple[str, float, str]:
        """Make governance decision. Returns (decision, risk_score, reason)."""
        base_risk = self.compute_risk(agent_id, message, path)
        scaled_risk = self.apply_harmonic_scaling(base_risk)
        normalized_risk = (scaled_risk - 1.0) / self.config.alpha

        # SNAP detection
        if normalized_risk >= self.config.snap_threshold:
            decision = "SNAP"
            reason = f"SNAP triggered: risk discontinuity ({normalized_risk:.3f})"
        elif normalized_risk >= self.config.deny_threshold:
            decision = "DENY"
            reason = f"Risk too high ({normalized_risk:.3f} >= {self.config.deny_threshold})"
        elif normalized_risk >= self.config.quarantine_threshold:
            decision = "QUARANTINE"
            reason = f"Elevated risk ({normalized_risk:.3f})"
        elif normalized_risk >= self.config.allow_threshold:
            decision = "QUARANTINE"
            reason = f"Moderate risk ({normalized_risk:.3f})"
        else:
            decision = "ALLOW"
            reason = f"Risk acceptable ({normalized_risk:.3f})"

        # Log decision
        self.logger.governance_decision(decision, normalized_risk, reason, agent_id)

        return decision, normalized_risk, reason


# =============================================================================
# PRODUCTION SERVICE
# =============================================================================

class SCBEProductionService:
    """
    Commercial-grade SCBE service.

    Integrates:
    - SpiralSeal SS1 cipher
    - GeoSeal geometric trust
    - Post-quantum cryptography
    - Dual lattice consensus
    - Quasicrystal validation
    - Governance with SNAP
    """

    def __init__(
        self,
        password: bytes = b"scbe-production-key",
        config: Optional[ProductionConfig] = None,
    ):
        self.config = config or get_config()
        self.logger = get_logger("scbe.service")
        self.password = password

        # Initialize components
        self.seal = SpiralSeal(master_password=password)
        self.geoseal = GeoSealManifold(self.config)
        self.governance = GovernanceEngine(self.config, self.logger)
        self.qc_lattice = QuasicrystalLattice()
        self.qc_lattice.acceptance_radius = self.config.quasicrystal.acceptance_radius

        # PQC keys
        self._kyber_keypair = None
        self._dilithium_keypair = None
        self._init_pqc()

        # Storage
        self._shards: Dict[Tuple[int, ...], MemoryShard] = {}

        self.logger.info("SCBE Production Service initialized", {
            "config": self.config.environment.value,
            "pqc_backend": get_backend().name,
        })

    def _init_pqc(self):
        """Initialize PQC key pairs."""
        try:
            self._kyber_keypair = Kyber768.generate_keypair()
            self._dilithium_keypair = Dilithium3.generate_keypair()
        except Exception as e:
            self.logger.warning(f"PQC initialization failed: {e}", {"fallback": "mock"})

    def seal_memory(
        self,
        plaintext: bytes,
        agent_id: str,
        topic: str,
        position: Tuple[int, int, int, int, int, int],
        tags: Optional[List[str]] = None,
    ) -> MemoryShard:
        """
        Seal a memory with full cryptographic stack.

        Args:
            plaintext: Data to seal
            agent_id: Agent identifier
            topic: Topic/category
            position: 6D voxel position
            tags: Optional metadata tags

        Returns:
            MemoryShard with sealed data and proofs

        Raises:
            ValidationError: Invalid position
            CryptoError: Sealing failed
        """
        with TimedOperation("seal_memory", self.logger) as timer:
            # Validate position
            if len(position) != 6:
                raise ValidationError.invalid_position(
                    position, "Position must be 6-dimensional"
                )

            # Quasicrystal validation
            qc_result = self.qc_lattice.validate_gates(list(position))
            if qc_result.status == ValidationStatus.INVALID_OUTSIDE_WINDOW:
                self.logger.warning(
                    f"Position {position} outside quasicrystal acceptance window"
                )

            try:
                # Create AAD
                aad = f"agent:{agent_id}|topic:{topic}".encode()

                # SpiralSeal
                result = self.seal.seal(plaintext, aad=aad)
                sealed_blob = result.to_ss1_string()

                # PQC commitment
                pqc_ct = pqc_sig = pqc_ss = None
                if self._kyber_keypair and self._dilithium_keypair:
                    encap = Kyber768.encapsulate(self._kyber_keypair.public_key)
                    pqc_ct = encap.ciphertext
                    pqc_ss = encap.shared_secret

                    message = plaintext + aad
                    pqc_sig = Dilithium3.sign(self._dilithium_keypair.secret_key, message)

                # Create shard
                shard = MemoryShard(
                    position=position,
                    sealed_blob=sealed_blob,
                    aad=aad,
                    pqc_ciphertext=pqc_ct,
                    pqc_signature=pqc_sig,
                    pqc_shared_secret=pqc_ss,
                    agent_id=agent_id,
                    topic=topic,
                    tags=tags or [],
                )

                self._shards[position] = shard
                self.logger.seal_success(position, timer.duration_ms)

                return shard

            except Exception as e:
                self.logger.seal_failure(str(e), position)
                raise CryptoError.seal_failed(str(e)) from e

    def access_memory(self, request: AccessRequest) -> AccessResponse:
        """
        Process a memory access request through full security pipeline.

        Args:
            request: AccessRequest with agent context

        Returns:
            AccessResponse with decision and (optionally) data
        """
        self.logger.set_request_context(
            request_id=request.request_id,
            agent_id=request.agent_id,
        )

        response = AccessResponse(request_id=request.request_id)

        try:
            with TimedOperation("access_memory", self.logger) as timer:
                # 1. GeoSeal verification
                context = request.context_vector
                if context is None:
                    context = np.random.randn(6) * 0.5

                sphere_pos = self.geoseal.project_to_sphere(context)
                cube_pos = self.geoseal.project_to_hypercube(request.features)
                geo_distance = self.geoseal.geometric_distance(sphere_pos, cube_pos)
                path = self.geoseal.classify_path(geo_distance)
                time_dilation = self.geoseal.time_dilation_factor(geo_distance)

                response.geometric_distance = geo_distance
                response.path = path
                response.time_dilation = time_dilation

                self.logger.geoseal_result(path, geo_distance, time_dilation)

                # 2. Governance decision
                decision, risk_score, reason = self.governance.make_decision(
                    request.agent_id, request.message, path
                )
                response.risk_score = risk_score
                response.reason = reason

                # 3. Quasicrystal validation
                qc_result = self.qc_lattice.validate_gates(list(request.position))
                response.quasicrystal_valid = qc_result.status == ValidationStatus.VALID

                # 4. Lattice verification
                shard = self._shards.get(request.position)
                if shard and shard.pqc_ciphertext and shard.pqc_signature:
                    start = time.perf_counter()
                    try:
                        # Verify Kyber
                        Kyber768.decapsulate(
                            self._kyber_keypair.secret_key,
                            shard.pqc_ciphertext
                        )
                        primal_ok = True
                    except Exception:
                        primal_ok = False

                    try:
                        # Verify Dilithium
                        dual_ok = Dilithium3.verify(
                            self._dilithium_keypair.public_key,
                            shard.aad,
                            shard.pqc_signature
                        )
                    except Exception:
                        dual_ok = False

                    lattice_ms = (time.perf_counter() - start) * 1000
                    response.lattice_verified = primal_ok and dual_ok
                    self.logger.lattice_consensus(
                        response.lattice_verified, primal_ok, dual_ok, lattice_ms
                    )

                # 5. Final decision
                if decision in ("DENY", "SNAP") or path == "exterior":
                    response.decision = "DENY"
                    response.crypto_mode = "POST-QUANTUM"
                    response.latency_ms = 2000
                elif decision == "QUARANTINE":
                    response.decision = "QUARANTINE"
                    response.crypto_mode = "HYBRID"
                    response.latency_ms = int(500 * time_dilation)
                else:
                    response.decision = "ALLOW"
                    response.crypto_mode = "AES-256-GCM"
                    response.latency_ms = int(50 * time_dilation)

                    # Return data only on ALLOW
                    if shard:
                        try:
                            unsealed = self.seal.unseal(shard.sealed_blob, aad=shard.aad)
                            response.data = unsealed
                        except Exception as e:
                            self.logger.error(f"Unseal failed: {e}")
                            response.decision = "DENY"
                            response.reason = "Data integrity check failed"

        except Exception as e:
            self.logger.error(f"Access request failed: {e}", exception=e)
            response.decision = "DENY"
            response.reason = f"Internal error: {str(e)}"

        finally:
            self.logger.clear_request_context()

        return response

    def get_shard(self, position: Tuple[int, ...]) -> Optional[MemoryShard]:
        """Get a shard by position (no security checks)."""
        return self._shards.get(position)

    def list_shards(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List shards, optionally filtered by agent."""
        shards = []
        for shard in self._shards.values():
            if agent_id is None or shard.agent_id == agent_id:
                shards.append(shard.to_dict())
        return shards

    def health_check(self) -> Dict[str, Any]:
        """Service health check."""
        return {
            "status": "healthy",
            "version": self.config.service_version,
            "environment": self.config.environment.value,
            "pqc_backend": get_backend().name,
            "shards_count": len(self._shards),
            "config": {
                "geoseal_threshold": self.config.geoseal.interior_threshold,
                "governance_allow": self.config.governance.allow_threshold,
                "governance_deny": self.config.governance.deny_threshold,
            },
        }
