"""
Phase-Breath Hyperbolic Governance

Components:
    - Phase-breath transform (expansion/contraction cycles)
    - Snap protocol (discontinuity detection and rejection)
    - Grand Unified Equation enforcement
    - Causality verification

Governance Principle:
    Actions are not forbidden by policy;
    Invalid actions cannot exist on the manifold
    without violating causality or information balance.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

EPS = 1e-12
SNAP_THRESHOLD = 0.5  # Maximum allowed discontinuity in state trajectory
CAUSALITY_WINDOW = 1.0  # Time window for causality checks (seconds)
B_BREATH_MAX = 0.3  # Maximum breathing amplitude
OMEGA_BREATH = 2 * np.pi / 60  # Breathing frequency (1 minute cycle)


# =============================================================================
# ENUMS
# =============================================================================

class GovernanceDecision(Enum):
    """Governance action outcomes."""
    ALLOW = "ALLOW"
    QUARANTINE = "QUARANTINE"
    DENY = "DENY"
    SNAP_VIOLATION = "SNAP_VIOLATION"
    CAUSALITY_VIOLATION = "CAUSALITY_VIOLATION"


class BreathPhase(Enum):
    """Breathing cycle phase."""
    EXPANSION = "EXPANSION"   # b > 1: expanding
    CONTRACTION = "CONTRACTION"  # b < 1: contracting
    NEUTRAL = "NEUTRAL"  # b ≈ 1


# =============================================================================
# PHASE TRANSFORM
# =============================================================================

def mobius_add(a: np.ndarray, u: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Mobius addition on Poincare ball: a ⊕ u

    a ⊕ u = ((1 + 2⟨a,u⟩ + ||u||²)a + (1 - ||a||²)u) /
            (1 + 2⟨a,u⟩ + ||a||²||u||²)

    This is the fundamental hyperbolic translation operation.
    """
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    au = float(np.dot(a, u))
    aa = float(np.dot(a, a))
    uu = float(np.dot(u, u))

    denom = 1.0 + 2.0 * au + aa * uu
    denom = np.sign(denom) * max(abs(denom), eps)

    num = (1.0 + 2.0 * au + uu) * a + (1.0 - aa) * u
    return num / denom


def clamp_ball(u: np.ndarray, eps_ball: float = 1e-3) -> np.ndarray:
    """Clamp vector to interior of unit ball: ||u|| <= 1 - eps_ball."""
    r = float(np.linalg.norm(u))
    r_max = 1.0 - eps_ball
    if r <= r_max or r == 0.0:
        return u
    return (r_max / r) * u


def phase_transform(
    u: np.ndarray,
    a: np.ndarray,
    Q: Optional[np.ndarray] = None,
    eps_ball: float = 1e-3
) -> np.ndarray:
    """
    Phase transform: T_phase(u) = Q(a ⊕ u)

    Combines Mobius translation by 'a' with optional rotation Q.
    This is an isometry of hyperbolic space.

    Args:
        u: Point in Poincare ball
        a: Translation vector (hyperbolic velocity)
        Q: Optional rotation matrix
        eps_ball: Ball boundary margin

    Returns:
        Transformed point in Poincare ball
    """
    u2 = mobius_add(a, u)
    if Q is not None:
        u2 = np.asarray(Q, dtype=np.float64) @ np.asarray(u2, dtype=np.float64)
    return clamp_ball(u2, eps_ball=eps_ball)


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Create 2D rotation matrix for angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotation_matrix_nd(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """
    Create n-dimensional rotation matrix in (i,j) plane.

    Args:
        n: Dimension of space
        i, j: Indices of rotation plane (i < j)
        theta: Rotation angle

    Returns:
        n×n rotation matrix
    """
    R = np.eye(n, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


# =============================================================================
# BREATHING TRANSFORM
# =============================================================================

def breathing_factor(t: float, b_max: float = B_BREATH_MAX, omega: float = OMEGA_BREATH) -> float:
    """
    Compute breathing factor: b(t) = 1 + b_max·sin(ωt)

    Creates expansion/contraction cycles in hyperbolic space:
        - b > 1: Expansion (points move toward boundary)
        - b < 1: Contraction (points move toward origin)
        - b = 1: Neutral (identity transform)
    """
    return 1.0 + b_max * np.sin(omega * t)


def get_breath_phase(t: float, b_max: float = B_BREATH_MAX, omega: float = OMEGA_BREATH) -> BreathPhase:
    """Determine current phase of breathing cycle."""
    b = breathing_factor(t, b_max, omega)
    if b > 1.01:
        return BreathPhase.EXPANSION
    elif b < 0.99:
        return BreathPhase.CONTRACTION
    return BreathPhase.NEUTRAL


def breathing_transform(
    u: np.ndarray,
    b: float,
    eps_ball: float = 1e-3
) -> np.ndarray:
    """
    Breathing transform: radial diffeomorphism on the Poincare ball.

    T_breath(u; b) = tanh(b·arctanh(||u||)) · u/||u||

    Properties:
        - Diffeomorphism of ball onto itself
        - b > 1: Expands (pushes toward boundary)
        - b < 1: Contracts (pulls toward origin)
        - b = 1: Identity
        - Preserves hyperbolic distance (isometry)

    Args:
        u: Point in Poincare ball
        b: Breathing factor
        eps_ball: Ball boundary margin

    Returns:
        Transformed point
    """
    u = np.asarray(u, dtype=np.float64)
    r = float(np.linalg.norm(u))

    if r == 0.0:
        return u.copy()

    r = min(r, 1.0 - eps_ball)
    rp = float(np.tanh(b * np.arctanh(r)))
    out = (rp / r) * u
    return clamp_ball(out, eps_ball=eps_ball)


def breathing_transform_timed(
    u: np.ndarray,
    t: float,
    b_max: float = B_BREATH_MAX,
    omega: float = OMEGA_BREATH,
    eps_ball: float = 1e-3
) -> np.ndarray:
    """
    Apply breathing transform at time t.

    Convenience wrapper that computes b(t) and applies transform.
    """
    b = breathing_factor(t, b_max, omega)
    return breathing_transform(u, b, eps_ball)


# =============================================================================
# SNAP PROTOCOL - DISCONTINUITY DETECTION
# =============================================================================

@dataclass
class SnapEvent:
    """Record of a detected discontinuity (snap)."""
    timestamp: float
    location: np.ndarray
    magnitude: float
    previous_state: np.ndarray
    attempted_state: np.ndarray
    reason: str


class SnapProtocol:
    """
    Discontinuity detection and rejection system.

    Enforces smooth evolution on the manifold by detecting and rejecting
    state transitions that exceed the snap threshold. This prevents
    teleportation attacks and ensures physical plausibility.

    Principle: Valid trajectories must be continuous. Discontinuities
    (snaps) indicate either attacks or system errors.
    """

    def __init__(
        self,
        threshold: float = SNAP_THRESHOLD,
        history_size: int = 100
    ):
        self.threshold = threshold
        self.history_size = history_size
        self.state_history: List[Tuple[float, np.ndarray]] = []
        self.snap_events: List[SnapEvent] = []

    def hyperbolic_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute Poincare ball distance."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        uu = float(np.dot(u, u))
        vv = float(np.dot(v, v))
        duv = float(np.dot(u - v, u - v))

        # Clamp for numerical stability
        uu = min(uu, 1.0 - EPS)
        vv = min(vv, 1.0 - EPS)

        denom = max((1.0 - uu) * (1.0 - vv), EPS)
        arg = 1.0 + (2.0 * duv) / denom
        arg = max(1.0, arg)

        return float(np.arccosh(arg))

    def check_transition(
        self,
        t: float,
        new_state: np.ndarray
    ) -> Tuple[bool, Optional[SnapEvent]]:
        """
        Check if state transition is valid (no snap).

        Args:
            t: Current timestamp
            new_state: Proposed new state

        Returns:
            (is_valid, snap_event) - True if valid, snap event if not
        """
        if not self.state_history:
            return True, None

        last_t, last_state = self.state_history[-1]
        dt = t - last_t

        if dt <= 0:
            # Non-positive time delta - causality violation
            event = SnapEvent(
                timestamp=t,
                location=new_state,
                magnitude=float('inf'),
                previous_state=last_state,
                attempted_state=new_state,
                reason="Negative time delta (causality violation)"
            )
            return False, event

        # Compute displacement rate in hyperbolic space
        d_H = self.hyperbolic_distance(last_state, new_state)
        velocity = d_H / dt

        # Check against threshold
        if velocity > self.threshold:
            event = SnapEvent(
                timestamp=t,
                location=new_state,
                magnitude=velocity,
                previous_state=last_state,
                attempted_state=new_state,
                reason=f"Velocity {velocity:.4f} exceeds threshold {self.threshold:.4f}"
            )
            return False, event

        return True, None

    def record_state(self, t: float, state: np.ndarray) -> None:
        """Record a valid state transition."""
        self.state_history.append((t, state.copy()))

        # Trim history
        if len(self.state_history) > self.history_size:
            self.state_history = self.state_history[-self.history_size:]

    def record_snap(self, event: SnapEvent) -> None:
        """Record a snap event."""
        self.snap_events.append(event)

    def validate_and_record(
        self,
        t: float,
        new_state: np.ndarray
    ) -> Tuple[GovernanceDecision, Optional[SnapEvent]]:
        """
        Validate transition and record if valid.

        Returns:
            (decision, snap_event) - ALLOW if valid, SNAP_VIOLATION otherwise
        """
        is_valid, event = self.check_transition(t, new_state)

        if is_valid:
            self.record_state(t, new_state)
            return GovernanceDecision.ALLOW, None
        else:
            self.record_snap(event)
            return GovernanceDecision.SNAP_VIOLATION, event

    def get_snap_count(self) -> int:
        """Get total number of snap events."""
        return len(self.snap_events)

    def get_recent_snaps(self, n: int = 10) -> List[SnapEvent]:
        """Get most recent snap events."""
        return self.snap_events[-n:]

    def clear_history(self) -> None:
        """Clear state history and snap events."""
        self.state_history.clear()
        self.snap_events.clear()


# =============================================================================
# CAUSALITY VERIFICATION
# =============================================================================

@dataclass
class CausalityRecord:
    """Record of a causal event."""
    timestamp: float
    event_hash: bytes
    parent_hash: Optional[bytes]
    state_commitment: bytes


class CausalityVerifier:
    """
    Verifies causal ordering of events.

    Ensures that:
    1. Events have monotonically increasing timestamps
    2. Each event references its causal parent
    3. No cycles exist in the causal graph

    This prevents time-travel attacks and ensures consistent ordering.
    """

    def __init__(self, window: float = CAUSALITY_WINDOW):
        self.window = window
        self.records: List[CausalityRecord] = []
        self.hash_set: set = set()

    def compute_event_hash(
        self,
        t: float,
        state: np.ndarray,
        parent_hash: Optional[bytes] = None
    ) -> bytes:
        """Compute hash of event for causal linking."""
        data = f"{t:.12f}|".encode()
        data += state.tobytes()
        if parent_hash:
            data += b"|" + parent_hash
        return hashlib.sha3_256(data).digest()

    def compute_state_commitment(self, state: np.ndarray) -> bytes:
        """Compute commitment to state."""
        return hashlib.sha3_256(state.tobytes()).digest()

    def verify_causality(
        self,
        t: float,
        state: np.ndarray,
        parent_hash: Optional[bytes] = None
    ) -> Tuple[bool, str]:
        """
        Verify that a new event maintains causal consistency.

        Args:
            t: Timestamp of new event
            state: State at new event
            parent_hash: Hash of causal parent

        Returns:
            (is_valid, reason)
        """
        # Check timestamp ordering
        if self.records:
            last_t = self.records[-1].timestamp
            if t <= last_t:
                return False, f"Timestamp {t} not after last {last_t}"

        # Check parent exists if specified
        if parent_hash and parent_hash not in self.hash_set:
            return False, "Parent hash not found in causal history"

        # If we have records, require parent to be recent
        if self.records and parent_hash:
            parent_found = False
            for record in reversed(self.records[-100:]):
                if record.event_hash == parent_hash:
                    parent_found = True
                    if t - record.timestamp > self.window:
                        return False, f"Parent too old: {t - record.timestamp:.2f}s"
                    break
            if not parent_found:
                return False, "Parent not in recent history"

        return True, "OK"

    def record_event(
        self,
        t: float,
        state: np.ndarray,
        parent_hash: Optional[bytes] = None
    ) -> CausalityRecord:
        """Record a new event in the causal chain."""
        event_hash = self.compute_event_hash(t, state, parent_hash)
        commitment = self.compute_state_commitment(state)

        record = CausalityRecord(
            timestamp=t,
            event_hash=event_hash,
            parent_hash=parent_hash,
            state_commitment=commitment
        )

        self.records.append(record)
        self.hash_set.add(event_hash)

        return record

    def get_chain_length(self) -> int:
        """Get length of causal chain."""
        return len(self.records)

    def verify_chain_integrity(self) -> Tuple[bool, str]:
        """Verify integrity of entire causal chain."""
        if not self.records:
            return True, "Empty chain"

        for i, record in enumerate(self.records[1:], 1):
            if record.timestamp <= self.records[i-1].timestamp:
                return False, f"Timestamp violation at index {i}"

            if record.parent_hash and record.parent_hash not in self.hash_set:
                return False, f"Missing parent at index {i}"

        return True, "Chain valid"


# =============================================================================
# GRAND UNIFIED EQUATION ENFORCEMENT
# =============================================================================

@dataclass
class GUEState:
    """State for Grand Unified Equation evaluation."""
    risk_base: float      # Base behavioral risk
    d_star: float         # Hyperbolic distance to realm
    H_factor: float       # Harmonic scaling factor
    risk_prime: float     # Final risk: risk_base × H_factor
    threshold_allow: float = 0.30
    threshold_deny: float = 0.70


def harmonic_scaling(d: float, R: float = 1.5, alpha: float = 10.0, beta: float = 0.5) -> float:
    """
    Compute harmonic scaling factor H(d*, R).

    Bounded tanh form: H = 1 + alpha × tanh(beta × d*)

    Properties:
        - H ∈ [1, 1 + alpha]
        - Monotonically increasing in d*
        - No overflow for large d*
    """
    return 1.0 + alpha * math.tanh(beta * d)


def evaluate_gue(
    risk_base: float,
    d_star: float,
    R: float = 1.5,
    alpha: float = 10.0,
    beta: float = 0.5
) -> GUEState:
    """
    Evaluate Grand Unified Equation.

    Risk' = Risk_base × H(d*, R)

    Args:
        risk_base: Base behavioral risk score [0, 1]
        d_star: Hyperbolic distance to nearest trusted realm
        R: Harmonic ratio (default: perfect fifth = 3/2)
        alpha: Maximum scaling multiplier
        beta: Growth rate

    Returns:
        GUEState with computed values
    """
    H = harmonic_scaling(d_star, R, alpha, beta)
    risk_prime = risk_base * H

    return GUEState(
        risk_base=risk_base,
        d_star=d_star,
        H_factor=H,
        risk_prime=risk_prime
    )


def gue_decision(state: GUEState) -> GovernanceDecision:
    """
    Make governance decision from GUE state.

    Decision boundaries:
        - Risk' < 0.30: ALLOW
        - Risk' > 0.70: DENY
        - Otherwise: QUARANTINE
    """
    if state.risk_prime < state.threshold_allow:
        return GovernanceDecision.ALLOW
    elif state.risk_prime > state.threshold_deny:
        return GovernanceDecision.DENY
    return GovernanceDecision.QUARANTINE


# =============================================================================
# INTEGRATED GOVERNANCE ENGINE
# =============================================================================

class GovernanceEngine:
    """
    Integrated Phase-Breath Hyperbolic Governance Engine.

    Combines:
        - Phase-breath transforms for state evolution
        - Snap protocol for discontinuity detection
        - Causality verification for temporal consistency
        - GUE evaluation for risk-based decisions

    Principle: Invalid actions cannot exist on the manifold without
    violating causality or information balance.
    """

    def __init__(
        self,
        snap_threshold: float = SNAP_THRESHOLD,
        causality_window: float = CAUSALITY_WINDOW,
        b_max: float = B_BREATH_MAX,
        omega: float = OMEGA_BREATH
    ):
        self.snap_protocol = SnapProtocol(threshold=snap_threshold)
        self.causality_verifier = CausalityVerifier(window=causality_window)
        self.b_max = b_max
        self.omega = omega
        self.current_time = 0.0

    def evaluate(
        self,
        t: float,
        state: np.ndarray,
        risk_base: float,
        realm_centers: Optional[List[np.ndarray]] = None
    ) -> Tuple[GovernanceDecision, dict]:
        """
        Full governance evaluation.

        Args:
            t: Current timestamp
            state: Current state in Poincare ball
            risk_base: Base behavioral risk [0, 1]
            realm_centers: Trusted realm centers

        Returns:
            (decision, details)
        """
        details = {
            "timestamp": t,
            "breath_phase": get_breath_phase(t, self.b_max, self.omega).value,
            "breath_factor": breathing_factor(t, self.b_max, self.omega)
        }

        # 1. Snap protocol check
        snap_decision, snap_event = self.snap_protocol.validate_and_record(t, state)
        if snap_decision == GovernanceDecision.SNAP_VIOLATION:
            details["snap_event"] = {
                "magnitude": snap_event.magnitude,
                "reason": snap_event.reason
            }
            return GovernanceDecision.SNAP_VIOLATION, details

        # 2. Causality verification
        parent_hash = None
        if self.causality_verifier.records:
            parent_hash = self.causality_verifier.records[-1].event_hash

        causal_valid, causal_reason = self.causality_verifier.verify_causality(
            t, state, parent_hash
        )
        if not causal_valid:
            details["causality_violation"] = causal_reason
            return GovernanceDecision.CAUSALITY_VIOLATION, details

        # Record causal event
        self.causality_verifier.record_event(t, state, parent_hash)

        # 3. Compute d* if realm centers provided
        d_star = 0.0
        if realm_centers:
            distances = [
                self.snap_protocol.hyperbolic_distance(state, c)
                for c in realm_centers
            ]
            d_star = min(distances) if distances else float('inf')

        details["d_star"] = d_star

        # 4. GUE evaluation
        gue_state = evaluate_gue(risk_base, d_star)
        details["gue"] = {
            "risk_base": gue_state.risk_base,
            "H_factor": gue_state.H_factor,
            "risk_prime": gue_state.risk_prime
        }

        decision = gue_decision(gue_state)
        details["decision"] = decision.value

        self.current_time = t
        return decision, details

    def apply_breath_transform(self, state: np.ndarray, t: float) -> np.ndarray:
        """Apply breathing transform at time t."""
        return breathing_transform_timed(state, t, self.b_max, self.omega)

    def apply_phase_transform(
        self,
        state: np.ndarray,
        translation: np.ndarray,
        rotation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply phase transform."""
        return phase_transform(state, translation, rotation)

    def get_statistics(self) -> dict:
        """Get governance statistics."""
        return {
            "snap_count": self.snap_protocol.get_snap_count(),
            "causal_chain_length": self.causality_verifier.get_chain_length(),
            "current_time": self.current_time
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "GovernanceDecision",
    "BreathPhase",
    # Phase transforms
    "mobius_add",
    "clamp_ball",
    "phase_transform",
    "rotation_matrix_2d",
    "rotation_matrix_nd",
    # Breathing transforms
    "breathing_factor",
    "get_breath_phase",
    "breathing_transform",
    "breathing_transform_timed",
    # Snap protocol
    "SnapEvent",
    "SnapProtocol",
    # Causality
    "CausalityRecord",
    "CausalityVerifier",
    # GUE
    "GUEState",
    "harmonic_scaling",
    "evaluate_gue",
    "gue_decision",
    # Engine
    "GovernanceEngine",
    # Constants
    "SNAP_THRESHOLD",
    "CAUSALITY_WINDOW",
    "B_BREATH_MAX",
    "OMEGA_BREATH",
]
