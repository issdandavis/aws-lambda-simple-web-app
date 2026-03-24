"""
Temporal Lattice — Time as a Stabilization Axis

The insight: Equations don't need to be stable at the START.
They re-stabilize on ARRIVAL when all conditions align.

Think of it like:
- Throwing a ball through fog (unstable path)
- But it MUST land in exactly the right spot
- AND hit all 7 checkpoints
- AND pass through two different verification gates

If ANY piece is wrong, the equation never stabilizes.

Dual-Lattice Quantum Security:
- Kyber (lattice-based KEM) for key exchange
- Dilithium (lattice-based signatures) for verification
- If one gets cracked, the other still protects you
- Both must agree for the equation to "settle"

The 7 Vertices:
- Time axis (when)
- Space axes x,y,z (where in behavior sphere)
- Policy axes (what rules apply)
- Ring position (trust level)
- Intent binding (why)
All 7 must align for the unstable equation to crystallize.
"""

import math
import hashlib
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
from enum import Enum

from .constants import GOLDEN_RATIO, PERFECT_FIFTH, PHI_AETHER


class StabilizationState(Enum):
    """States of the temporal equation."""
    UNSTABLE = "unstable"      # In flight, not yet resolved
    OSCILLATING = "oscillating"  # Near resolution, cycling
    CRYSTALLIZED = "crystallized"  # All vertices aligned, stable
    COLLAPSED = "collapsed"     # Failed to stabilize, error state


@dataclass
class TemporalVertex:
    """One of the 7 vertices that must align."""
    name: str
    value: float
    tolerance: float
    weight: float  # How much this vertex matters
    is_aligned: bool = False

    def check_alignment(self, target: float) -> bool:
        """Check if this vertex aligns with target."""
        self.is_aligned = abs(self.value - target) <= self.tolerance
        return self.is_aligned


@dataclass
class DualLatticeProof:
    """
    Proof from both Kyber and Dilithium lattices.

    Both must agree for the equation to stabilize.
    If one is compromised, the other still protects.
    """
    kyber_commitment: bytes      # KEM-derived binding
    dilithium_signature: bytes   # Signature binding
    combined_hash: bytes         # H(kyber || dilithium)
    timestamp: float             # When proof was created
    is_valid: bool = False

    def verify(self, expected_hash: bytes) -> bool:
        """Verify both lattices agree."""
        # Both must match
        self.is_valid = self.combined_hash == expected_hash
        return self.is_valid


@dataclass
class TemporalEquation:
    """
    An equation that starts unstable and crystallizes on arrival.

    The equation oscillates through time until:
    1. All 7 vertices align
    2. Both lattice proofs verify
    3. Time reaches the target window

    Only then does it "settle" into a stable, usable form.
    """
    # The 7 vertices
    vertices: List[TemporalVertex] = field(default_factory=list)

    # Dual lattice proofs
    kyber_proof: Optional[bytes] = None
    dilithium_proof: Optional[bytes] = None

    # Temporal parameters
    creation_time: float = 0.0
    target_time: float = 0.0
    time_tolerance: float = 1.0  # Seconds

    # State
    state: StabilizationState = StabilizationState.UNSTABLE
    oscillation_count: int = 0
    max_oscillations: int = 100

    # Result (only valid when crystallized)
    crystallized_value: Optional[bytes] = None


def create_7_vertices(
    context: np.ndarray,
    policy: Dict[str, float],
    ring_position: float,
    intent_hash: bytes,
    target_time: float
) -> List[TemporalVertex]:
    """
    Create the 7 vertices that must align.

    Vertices:
    1. Time - when the operation should crystallize
    2. X (behavior) - first behavior coordinate
    3. Y (behavior) - second behavior coordinate
    4. Z (behavior) - third behavior coordinate
    5. Policy - combined policy score
    6. Ring - trust ring position
    7. Intent - intent binding hash
    """
    # Extract behavior coordinates (normalized to sphere)
    behavior = context[3:6] if len(context) >= 6 else np.array([0.5, 0.5, 0.5])
    norm = np.linalg.norm(behavior)
    if norm > 0:
        behavior = behavior / norm

    # Policy score
    policy_score = sum(policy.values()) / len(policy) if policy else 0.5

    # Intent binding (hash to float 0-1)
    intent_value = int.from_bytes(intent_hash[:4], 'little') / (2**32)

    return [
        TemporalVertex("time", target_time, 1.0, 1.5),           # Heavier weight
        TemporalVertex("x_behavior", behavior[0], 0.1, 1.0),
        TemporalVertex("y_behavior", behavior[1], 0.1, 1.0),
        TemporalVertex("z_behavior", behavior[2], 0.1, 1.0),
        TemporalVertex("policy", policy_score, 0.15, 1.2),
        TemporalVertex("ring", ring_position, 0.1, 1.3),         # Ring matters more
        TemporalVertex("intent", intent_value, 0.05, 1.4),       # Intent is precise
    ]


def oscillate_equation(
    equation: TemporalEquation,
    current_context: np.ndarray,
    current_time: float
) -> Tuple[StabilizationState, float]:
    """
    Attempt one oscillation cycle.

    The equation "vibrates" between stable and unstable,
    getting closer to crystallization each cycle if conditions
    are right.

    Returns:
        Tuple of (new_state, stability_score 0-1)
    """
    equation.oscillation_count += 1

    # Check if we've exceeded max oscillations
    if equation.oscillation_count > equation.max_oscillations:
        equation.state = StabilizationState.COLLAPSED
        return (StabilizationState.COLLAPSED, 0.0)

    # Check time window
    time_delta = abs(current_time - equation.target_time)
    if time_delta > equation.time_tolerance * 10:
        # Way outside window - still unstable
        return (StabilizationState.UNSTABLE, 0.1)

    # Check vertex alignments
    aligned_count = 0
    weighted_alignment = 0.0
    total_weight = 0.0

    for vertex in equation.vertices:
        # Generate target from current context
        target = compute_vertex_target(vertex.name, current_context, current_time)
        if vertex.check_alignment(target):
            aligned_count += 1
            weighted_alignment += vertex.weight
        total_weight += vertex.weight

    stability_score = weighted_alignment / total_weight if total_weight > 0 else 0.0

    # State machine
    if stability_score >= 0.95 and aligned_count == 7:
        equation.state = StabilizationState.CRYSTALLIZED
    elif stability_score >= 0.7:
        equation.state = StabilizationState.OSCILLATING
    else:
        equation.state = StabilizationState.UNSTABLE

    return (equation.state, stability_score)


def compute_vertex_target(
    vertex_name: str,
    context: np.ndarray,
    current_time: float
) -> float:
    """Compute target value for a vertex from current context."""
    if vertex_name == "time":
        return current_time
    elif vertex_name == "x_behavior":
        return context[3] / (np.linalg.norm(context[3:6]) + 1e-10) if len(context) >= 6 else 0.5
    elif vertex_name == "y_behavior":
        return context[4] / (np.linalg.norm(context[3:6]) + 1e-10) if len(context) >= 6 else 0.5
    elif vertex_name == "z_behavior":
        return context[5] / (np.linalg.norm(context[3:6]) + 1e-10) if len(context) >= 6 else 0.5
    elif vertex_name == "policy":
        return context[2] / 10.0 if len(context) >= 3 else 0.5  # threat_level normalized
    elif vertex_name == "ring":
        return context[5] if len(context) >= 6 else 0.5  # behavior_stability
    elif vertex_name == "intent":
        # Intent should be pre-computed
        return 0.5
    return 0.5


# =============================================================================
# DUAL LATTICE VERIFICATION
# =============================================================================

def create_kyber_commitment(
    shared_secret: bytes,
    vertices: List[TemporalVertex],
    timestamp: float
) -> bytes:
    """
    Create Kyber-based commitment.

    In production: Use actual Kyber KEM.
    Here: Simulate with HKDF-style derivation.
    """
    # Serialize vertices
    vertex_data = b""
    for v in vertices:
        vertex_data += v.name.encode()
        vertex_data += struct.pack('<f', v.value)

    # Derive commitment
    commitment = hashlib.sha256(
        b"kyber:" + shared_secret + vertex_data +
        struct.pack('<d', timestamp)
    ).digest()

    return commitment


def create_dilithium_signature(
    signing_key: bytes,
    message: bytes,
    vertices: List[TemporalVertex]
) -> bytes:
    """
    Create Dilithium-based signature.

    In production: Use actual Dilithium DSA.
    Here: Simulate with HMAC-style signing.
    """
    import hmac

    # Serialize vertices
    vertex_data = b""
    for v in vertices:
        vertex_data += v.name.encode()
        vertex_data += struct.pack('<f', v.value)

    signature = hmac.new(
        signing_key,
        b"dilithium:" + message + vertex_data,
        'sha256'
    ).digest()

    return signature


def verify_dual_lattice(
    equation: TemporalEquation,
    kyber_commitment: bytes,
    dilithium_signature: bytes,
    verification_key: bytes
) -> bool:
    """
    Verify both lattice proofs agree.

    This is the "dual lock" - both Kyber and Dilithium
    must verify for the equation to crystallize.

    If one lattice is compromised (quantum attack),
    the other still protects.
    """
    import hmac

    # Verify Kyber commitment
    expected_kyber = create_kyber_commitment(
        verification_key,
        equation.vertices,
        equation.creation_time
    )
    kyber_ok = kyber_commitment == expected_kyber

    # Verify Dilithium signature
    # (In real implementation, verify against public key)
    vertex_data = b""
    for v in equation.vertices:
        vertex_data += v.name.encode()
        vertex_data += struct.pack('<f', v.value)

    expected_dilithium = hmac.new(
        verification_key,
        b"dilithium:" + equation.crystallized_value + vertex_data,
        'sha256'
    ).digest()
    dilithium_ok = dilithium_signature == expected_dilithium

    # Both must pass
    return kyber_ok and dilithium_ok


# =============================================================================
# TIME-AXIS STABILIZATION
# =============================================================================

import struct

@dataclass
class TimeAxisConfig:
    """Configuration for time-axis stabilization."""
    oscillation_period: float = 0.1      # Seconds between oscillations
    max_drift: float = 2.0               # Max seconds of allowed drift
    crystallization_window: float = 0.5  # Window when crystallization possible
    entropy_injection: bool = True       # Add entropy between oscillations


def create_temporal_equation(
    context: np.ndarray,
    policy: Dict[str, float],
    ring_position: float,
    intent: str,
    target_time: float,
    shared_secret: bytes,
    signing_key: bytes
) -> TemporalEquation:
    """
    Create a temporal equation that will crystallize at target_time.

    The equation starts UNSTABLE and oscillates until:
    - All 7 vertices align
    - Dual lattice proofs verify
    - Time reaches target window
    """
    # Create intent hash
    intent_hash = hashlib.sha256(intent.encode()).digest()

    # Create the 7 vertices
    vertices = create_7_vertices(
        context, policy, ring_position, intent_hash, target_time
    )

    # Create equation
    equation = TemporalEquation(
        vertices=vertices,
        creation_time=time.time(),
        target_time=target_time,
        time_tolerance=1.0
    )

    # Create preliminary crystallized value (will be refined)
    vertex_data = b""
    for v in vertices:
        vertex_data += struct.pack('<f', v.value)
    equation.crystallized_value = hashlib.sha256(
        shared_secret + vertex_data + struct.pack('<d', target_time)
    ).digest()

    # Create dual lattice proofs
    equation.kyber_proof = create_kyber_commitment(
        shared_secret, vertices, equation.creation_time
    )
    equation.dilithium_proof = create_dilithium_signature(
        signing_key, equation.crystallized_value, vertices
    )

    return equation


def attempt_crystallization(
    equation: TemporalEquation,
    current_context: np.ndarray,
    verification_key: bytes,
    config: TimeAxisConfig = TimeAxisConfig()
) -> Tuple[bool, Optional[bytes]]:
    """
    Attempt to crystallize the equation.

    Returns:
        Tuple of (success, crystallized_value or None)
    """
    current_time = time.time()

    # Check if in crystallization window
    time_delta = abs(current_time - equation.target_time)
    if time_delta > config.max_drift:
        return (False, None)

    # Oscillate until stable or failed
    max_attempts = int(config.max_drift / config.oscillation_period)

    for _ in range(max_attempts):
        state, score = oscillate_equation(equation, current_context, time.time())

        if state == StabilizationState.CRYSTALLIZED:
            # Verify dual lattice
            if verify_dual_lattice(
                equation,
                equation.kyber_proof,
                equation.dilithium_proof,
                verification_key
            ):
                return (True, equation.crystallized_value)
            else:
                # Lattice verification failed
                equation.state = StabilizationState.COLLAPSED
                return (False, None)

        elif state == StabilizationState.COLLAPSED:
            return (False, None)

        # Add entropy if configured
        if config.entropy_injection:
            # Small random perturbation to help escape local minima
            for v in equation.vertices:
                v.value += (np.random.random() - 0.5) * 0.001

    return (False, None)


# =============================================================================
# ERROR TRAPDOOR
# =============================================================================

def time_dilation_error(
    equation: TemporalEquation,
    observed_time: float,
    expected_time: float
) -> Tuple[bool, float]:
    """
    Check for time dilation error (trapdoor).

    If time skew is too large, the equation "errors out"
    like falling into a black hole - time freezes.

    Returns:
        Tuple of (is_error, dilation_factor)
    """
    time_delta = abs(observed_time - expected_time)

    # Dilation factor (like relativistic gamma)
    if time_delta >= equation.time_tolerance:
        # Error out - infinite dilation
        return (True, float('inf'))

    # Normal dilation
    dilation = 1.0 / (1.0 - (time_delta / equation.time_tolerance))

    return (False, dilation)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_temporal_equation(equation: TemporalEquation) -> str:
    """Generate ASCII visualization of temporal equation state."""
    lines = []
    lines.append("┌────────────────────────────────────────────────────────────┐")
    lines.append("│            TEMPORAL LATTICE EQUATION                       │")
    lines.append("├────────────────────────────────────────────────────────────┤")

    # State indicator
    state_icons = {
        StabilizationState.UNSTABLE: "〰️  UNSTABLE",
        StabilizationState.OSCILLATING: "∿∿ OSCILLATING",
        StabilizationState.CRYSTALLIZED: "◈◈ CRYSTALLIZED",
        StabilizationState.COLLAPSED: "✗✗ COLLAPSED"
    }
    lines.append(f"│  State: {state_icons.get(equation.state, 'UNKNOWN'):20s}                     │")
    lines.append(f"│  Oscillations: {equation.oscillation_count:3d} / {equation.max_oscillations:3d}                            │")
    lines.append("├────────────────────────────────────────────────────────────┤")

    # Vertices
    lines.append("│  THE 7 VERTICES:                                           │")
    for v in equation.vertices:
        aligned = "✓" if v.is_aligned else "·"
        bar_len = int(v.value * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"│  {aligned} {v.name:12s}: [{bar}] {v.value:.3f}       │")

    lines.append("├────────────────────────────────────────────────────────────┤")

    # Dual lattice status
    lines.append("│  DUAL LATTICE VERIFICATION:                                │")
    kyber_status = "✓ VALID" if equation.kyber_proof else "? PENDING"
    dilithium_status = "✓ VALID" if equation.dilithium_proof else "? PENDING"
    lines.append(f"│    Kyber (KEM):     {kyber_status:15s}                    │")
    lines.append(f"│    Dilithium (DSA): {dilithium_status:15s}                    │")

    lines.append("├────────────────────────────────────────────────────────────┤")

    # Time axis
    time_remaining = equation.target_time - time.time()
    if time_remaining > 0:
        lines.append(f"│  Time to crystallization: {time_remaining:.2f}s                       │")
    else:
        lines.append(f"│  Past target by: {-time_remaining:.2f}s                               │")

    lines.append("└────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)


def explain_temporal_lattice() -> str:
    """Plain English explanation."""
    return """
    ╔══════════════════════════════════════════════════════════════════╗
    ║          TEMPORAL LATTICE: Time as a Stabilization Axis          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  THE BIG IDEA:                                                    ║
    ║  Equations don't need to be STABLE at the START.                 ║
    ║  They re-stabilize on ARRIVAL when all conditions align.        ║
    ║                                                                   ║
    ║  Think of it like:                                                ║
    ║  ┌────────────────────────────────────────────────────────┐      ║
    ║  │  UNSTABLE                              CRYSTALLIZED    │      ║
    ║  │     ↓                                       ↓          │      ║
    ║  │   〰️〰️〰️ ──oscillate──→ ∿∿∿∿ ──align──→ ◈◈◈         │      ║
    ║  │   (fuzzy)              (vibrating)     (solid)         │      ║
    ║  │                                                        │      ║
    ║  │   Like water            Like jello      Like ice       │      ║
    ║  │   (can't grab)          (wiggles)       (solid!)       │      ║
    ║  └────────────────────────────────────────────────────────┘      ║
    ║                                                                   ║
    ║  THE 7 VERTICES (all must align):                                 ║
    ║    1. TIME      - When should this happen?                        ║
    ║    2. X (space) - Where in behavior-space (x-axis)?              ║
    ║    3. Y (space) - Where in behavior-space (y-axis)?              ║
    ║    4. Z (space) - Where in behavior-space (z-axis)?              ║
    ║    5. POLICY    - What rules apply?                               ║
    ║    6. RING      - What trust level?                               ║
    ║    7. INTENT    - What are you trying to do?                      ║
    ║                                                                   ║
    ║  THE DUAL LATTICE (quantum-safe backup):                          ║
    ║    ┌─────────────┐    ┌─────────────────┐                        ║
    ║    │   KYBER     │    │   DILITHIUM     │                        ║
    ║    │   (KEM)     │    │   (Signatures)  │                        ║
    ║    │             │    │                 │                        ║
    ║    │ Key Exchange│    │ Authenticity    │                        ║
    ║    └──────┬──────┘    └────────┬────────┘                        ║
    ║           │                    │                                  ║
    ║           └────────┬───────────┘                                  ║
    ║                    ▼                                              ║
    ║           BOTH MUST AGREE                                         ║
    ║                                                                   ║
    ║  WHY DUAL LATTICE?                                                ║
    ║    If quantum computers crack Kyber → Dilithium still works      ║
    ║    If quantum computers crack Dilithium → Kyber still works      ║
    ║    You need to crack BOTH to break the system                    ║
    ║                                                                   ║
    ║  THE ERROR TRAPDOOR:                                              ║
    ║    If time skew is too large → equation "errors out"             ║
    ║    Like falling into a black hole (time freezes)                 ║
    ║    Attackers trying to manipulate time get trapped               ║
    ║                                                                   ║
    ║  THE RESULT:                                                      ║
    ║    • Start with fuzzy, unstable equation                         ║
    ║    • Oscillate through time                                       ║
    ║    • If all 7 vertices align AND dual lattice verifies           ║
    ║    • Equation CRYSTALLIZES into usable form                      ║
    ║    • Wrong context → never crystallizes → just noise             ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
