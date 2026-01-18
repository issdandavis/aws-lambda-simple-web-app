"""
Pydantic models for API request/response validation.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════

class Decision(str, Enum):
    ALLOW = "ALLOW"
    QUARANTINE = "QUARANTINE"
    DENY = "DENY"


class ConsensusState(str, Enum):
    PENDING = "pending"
    KYBER_ONLY = "kyber_only"
    DILITHIUM_ONLY = "dilithium_only"
    CONSENSUS = "consensus"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ═══════════════════════════════════════════════════════════════
# Risk Assessment
# ═══════════════════════════════════════════════════════════════

class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""
    context_vector: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="6D context vector [time, id, threat, entropy, load, stability]"
    )
    realm_centers: Optional[List[List[float]]] = Field(
        default=None,
        description="Known realm centers for distance calculation"
    )
    risk_base: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Base behavioral risk score"
    )
    use_bounded: bool = Field(
        default=False,
        description="Use bounded tanh form instead of exponential"
    )


class RiskAssessmentResponse(BaseModel):
    """Response from risk assessment."""
    decision: Decision
    risk_base: float
    risk_prime: float
    harmonic_scale: float
    d_star: float
    security_bits: float
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# Governance
# ═══════════════════════════════════════════════════════════════

class GovernanceVerifyRequest(BaseModel):
    """Request for governance verification."""
    state_trajectory: List[List[float]] = Field(
        ...,
        min_length=2,
        description="Sequence of state vectors to verify"
    )
    timestamps: List[float] = Field(
        ...,
        min_length=2,
        description="Timestamps for each state"
    )
    snap_threshold: float = Field(
        default=0.5,
        description="Threshold for snap detection"
    )


class SnapEvent(BaseModel):
    """A detected discontinuity."""
    index: int
    magnitude: float
    timestamp: float


class GovernanceVerifyResponse(BaseModel):
    """Response from governance verification."""
    valid: bool
    snap_events: List[SnapEvent]
    causality_verified: bool
    total_states: int
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# Consensus
# ═══════════════════════════════════════════════════════════════

class ConsensusSubmitRequest(BaseModel):
    """Request to submit data for consensus."""
    payload: str = Field(
        ...,
        description="Base64-encoded payload data"
    )
    context_id: str = Field(
        ...,
        description="Unique context identifier"
    )


class ConsensusSubmitResponse(BaseModel):
    """Response from consensus submission."""
    state: ConsensusState
    settled: bool
    context_commitment: str
    kyber_valid: bool
    dilithium_valid: bool
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# Harmonic Scaling (Direct)
# ═══════════════════════════════════════════════════════════════

class HarmonicScaleRequest(BaseModel):
    """Request for harmonic scaling calculation."""
    d_star: float = Field(
        ...,
        ge=0.0,
        description="Hyperbolic distance d*"
    )
    R: float = Field(
        default=1.5,
        gt=0.0,
        description="Harmonic ratio (default: 3/2 perfect fifth)"
    )
    base_security_bits: int = Field(
        default=128,
        description="Base security level in bits"
    )


class HarmonicScaleResponse(BaseModel):
    """Response from harmonic scaling."""
    H: float
    d_star: float
    R: float
    security_bits: float
    overflow: bool


# ═══════════════════════════════════════════════════════════════
# SpiralSeal (Sacred Tongues)
# ═══════════════════════════════════════════════════════════════

class SealRequest(BaseModel):
    """Request to seal data with SpiralSeal."""
    plaintext: str = Field(
        ...,
        description="Base64-encoded plaintext to seal"
    )
    aad: str = Field(
        default="",
        description="Additional authenticated data (context)"
    )
    kid: str = Field(
        default="k01",
        description="Key identifier"
    )


class SealResponse(BaseModel):
    """Response from sealing."""
    sealed_blob: str
    tongue_breakdown: dict
    timestamp: float


class UnsealRequest(BaseModel):
    """Request to unseal a SpiralSeal blob."""
    sealed_blob: str = Field(
        ...,
        description="SS1-formatted sealed blob"
    )
    aad: Optional[str] = Field(
        default=None,
        description="Expected AAD (if None, uses blob's AAD)"
    )


class UnsealResponse(BaseModel):
    """Response from unsealing."""
    plaintext: str
    kid: str
    aad: str
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# GeoSeal (Dual-Manifold Authorization)
# ═══════════════════════════════════════════════════════════════

class GeoSealAuthRequest(BaseModel):
    """Request for GeoSeal authorization."""
    theta: float = Field(
        ...,
        ge=0.0,
        le=3.15,
        description="Polar angle on sphere [0, π]"
    )
    phi: float = Field(
        ...,
        ge=0.0,
        le=6.29,
        description="Azimuthal angle on sphere [0, 2π]"
    )
    policy_coords: List[float] = Field(
        ...,
        min_length=3,
        description="Coordinates in policy hypercube [0,1]^m"
    )


class GeoSealAuthResponse(BaseModel):
    """Response from GeoSeal authorization."""
    authorized: bool
    intersection_type: str
    sphere_cell: int
    hypercube_cell: int
    radial_distance: float
    tau_allowed: float
    pow_bits: int
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# Physics Trap
# ═══════════════════════════════════════════════════════════════

class TrapChallengeResponse(BaseModel):
    """A physics trap challenge."""
    trap_id: str
    trap_type: str
    equation: str
    description: str
    given_values: dict
    prompt: str


class TrapSubmitRequest(BaseModel):
    """Submit response to physics trap."""
    trap_id: str = Field(..., description="Challenge ID")
    answer: float = Field(..., description="Computed answer")
    corrections: dict = Field(
        default_factory=dict,
        description="Values that were corrected"
    )
    explanation: str = Field(
        default="",
        description="Explanation of corrections"
    )


class TrapVerifyResponse(BaseModel):
    """Result of trap verification."""
    passed: bool
    is_rogue: bool
    corrections_made: bool
    answer_correct: bool
    explanation: str


# ═══════════════════════════════════════════════════════════════
# Swarm Governance
# ═══════════════════════════════════════════════════════════════

class AgentRegisterRequest(BaseModel):
    """Register a new agent."""
    agent_id: str
    parent_id: Optional[str] = None


class AgentStatusResponse(BaseModel):
    """Agent status."""
    agent_id: str
    state: str
    permission_level: int
    trust_score: float
    risk_score: float
    tasks_completed: int
    tasks_failed: int
    value_generated: float


class SwarmStatusResponse(BaseModel):
    """Overall swarm status."""
    total_agents: int
    by_state: dict
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    modules: dict
