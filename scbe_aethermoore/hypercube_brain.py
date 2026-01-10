"""
Hypercube-Brain Geometric Security Model

Core Concept:
- HYPERCUBE [0,1]^n: Multi-dimensional policy/rules space (expandable/retractable)
- SPHERE S^(n-1): "Brain" - behavioral/cognitive manifold
- INTERSECTION: Determines signature mode (Kyber internal vs Dilithium external)
- TIME DILATION TRAPDOOR: High velocity = infinite dilation = frozen/trapped

Key Innovation:
- Risk-based geometric expansion/retraction
- Multi-modular projection (mod 1) for dimension reduction
- Trapdoor puzzle using relativistic time dilation
"""

import numpy as np
import hashlib
import struct
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from enum import Enum
import hmac


class SignatureMode(Enum):
    """Signature mode based on geometric position"""
    KYBER_INTERNAL = "kyber"      # Inside sphere - internal operations
    DILITHIUM_EXTERNAL = "dilithium"  # Outside sphere - external operations
    TRAPDOOR_FROZEN = "frozen"    # Time dilation trap activated


@dataclass
class HypercubeConfig:
    """Configuration for hypercube policy space"""
    dimensions: int = 4           # Number of policy dimensions
    base_scale: float = 1.0       # Base scaling factor
    max_expansion: float = 3.0    # Maximum expansion under high risk
    min_retraction: float = 0.3   # Minimum retraction under low risk
    modular_projection_dims: int = 3  # Dimensions after mod projection


@dataclass
class BrainSphereConfig:
    """Configuration for brain sphere"""
    dimensions: int = 3           # Sphere embedding dimension
    epsilon: float = 0.01         # Boundary tolerance for "inside"
    delta: float = 0.05           # Direction alignment tolerance
    center: Optional[np.ndarray] = None  # Sphere center (origin if None)


@dataclass
class TrapdoorConfig:
    """Configuration for time dilation trapdoor"""
    c: float = 1.0                # Speed of light (normalized)
    gamma_threshold: float = 2.0  # Dilation threshold for trap
    gamma_freeze: float = 10.0    # Dilation value that freezes completely
    velocity_index: int = 5       # Context index for velocity fraction


@dataclass
class GeometricState:
    """Complete state of the hypercube-brain system"""
    hypercube_point: np.ndarray   # v ∈ [0,1]^n
    sphere_point: np.ndarray      # u ∈ S^(n-1)
    scaled_hypercube: np.ndarray  # v after risk scaling
    projected_point: np.ndarray   # v projected to sphere dimensions (mod 1)
    radial_distance: float        # ||v_proj|| distance from origin
    direction_alignment: float    # ||u - v_hat|| direction difference
    is_inside: bool               # Whether inside sphere
    gamma: float                  # Time dilation factor
    signature_mode: SignatureMode
    risk_factor: float            # Current risk level
    expansion_factor: float       # Current expansion/retraction


def compute_time_dilation(v_fraction: float, c: float = 1.0) -> float:
    """
    Compute Lorentz time dilation factor.

    γ = 1 / √(1 - v²/c²)

    As v → c, γ → ∞ (time freezes)
    """
    if v_fraction >= c:
        return float('inf')
    if v_fraction <= 0:
        return 1.0
    return 1.0 / np.sqrt(1.0 - (v_fraction / c) ** 2)


def context_to_seed(context: np.ndarray) -> int:
    """Convert context array to deterministic seed"""
    context_bytes = context.tobytes()
    hash_bytes = hashlib.sha256(context_bytes).digest()
    return int.from_bytes(hash_bytes[:8], 'big')


def generate_hypercube_point(seed: int, dimensions: int) -> np.ndarray:
    """
    Generate deterministic point in [0,1]^n hypercube from seed.

    Each dimension extracted from different bits of the seed hash.
    """
    v = np.zeros(dimensions)

    # Use HKDF-like expansion for more dimensions
    for i in range(dimensions):
        dim_hash = hashlib.sha256(
            struct.pack('<QI', seed, i)
        ).digest()
        # Convert first 4 bytes to float in [0, 1]
        int_val = int.from_bytes(dim_hash[:4], 'big')
        v[i] = int_val / (2**32 - 1)

    return v


def generate_sphere_point(seed: int, dimensions: int) -> np.ndarray:
    """
    Generate deterministic point on S^(n-1) unit sphere.

    Uses seed to generate angles, then converts to Cartesian coordinates.
    """
    # Generate n-1 angles for n-dimensional sphere
    angles = np.zeros(dimensions - 1)

    for i in range(dimensions - 1):
        angle_hash = hashlib.sha256(
            struct.pack('<QI', seed, i + 1000)  # Offset to differ from hypercube
        ).digest()
        int_val = int.from_bytes(angle_hash[:4], 'big')

        if i < dimensions - 2:
            # θ_i ∈ [0, π] for all but last angle
            angles[i] = (int_val / (2**32 - 1)) * np.pi
        else:
            # φ ∈ [0, 2π] for last angle
            angles[i] = (int_val / (2**32 - 1)) * 2 * np.pi

    # Convert spherical to Cartesian
    u = np.zeros(dimensions)

    if dimensions == 3:
        # Standard 3D spherical coordinates
        theta, phi = angles[0], angles[1]
        u[0] = np.sin(theta) * np.cos(phi)
        u[1] = np.sin(theta) * np.sin(phi)
        u[2] = np.cos(theta)
    else:
        # General n-dimensional hyperspherical coordinates
        u[0] = np.cos(angles[0])
        for i in range(1, dimensions - 1):
            u[i] = np.prod([np.sin(angles[j]) for j in range(i)]) * np.cos(angles[i])
        u[dimensions - 1] = np.prod([np.sin(a) for a in angles])

    # Ensure unit norm
    norm = np.linalg.norm(u)
    if norm > 1e-12:
        u = u / norm

    return u


def compute_risk_factor(context: np.ndarray, risk_index: int = 2) -> float:
    """
    Extract risk factor from context.

    Default: context[2] is threat/risk level
    Returns value in [0, 1] range
    """
    if len(context) <= risk_index:
        return 0.5  # Default moderate risk

    raw_risk = context[risk_index]

    # Normalize to [0, 1] assuming raw_risk is typically 0-10 scale
    if raw_risk < 0:
        return 0.0
    elif raw_risk > 10:
        return 1.0
    else:
        return raw_risk / 10.0


def compute_expansion_factor(
    risk: float,
    config: HypercubeConfig
) -> float:
    """
    Compute hypercube expansion/retraction based on risk.

    High risk → Expansion (more restrictive policy space)
    Low risk → Retraction (relaxed policy space)
    """
    # Linear interpolation between retraction and expansion
    # risk=0 → min_retraction, risk=1 → max_expansion
    return config.min_retraction + risk * (config.max_expansion - config.min_retraction)


def project_to_sphere_dimensions(
    v: np.ndarray,
    target_dims: int
) -> np.ndarray:
    """
    Project hypercube point to sphere dimensions using modular arithmetic.

    This is the "multi-modular" projection:
    - Take first target_dims coordinates
    - Apply mod 1 to wrap into [0, 1]^target_dims
    """
    v_proj = v[:target_dims].copy()
    v_proj = v_proj % 1.0  # Multi-modular wrap
    return v_proj


def check_sphere_intersection(
    v_proj: np.ndarray,
    u: np.ndarray,
    config: BrainSphereConfig
) -> Tuple[bool, float, float]:
    """
    Check if projected hypercube point intersects with brain sphere.

    Returns:
        (is_inside, radial_distance, direction_alignment)

    Intersection criteria:
    1. Radial distance ρ = ||v_proj|| ≤ 1 + ε (inside sphere)
    2. Direction alignment ||u - v̂|| ≤ δ (pointing same way)
    """
    rho = np.linalg.norm(v_proj)

    # Compute unit direction of projected point
    if rho > 1e-12:
        v_hat = v_proj / rho
    else:
        v_hat = np.zeros_like(v_proj)

    # Direction alignment
    direction_diff = np.linalg.norm(u - v_hat)

    # Inside check
    inside_radius = rho <= (1.0 + config.epsilon)
    aligned_direction = direction_diff <= config.delta

    is_inside = inside_radius and aligned_direction

    return is_inside, rho, direction_diff


def determine_signature_mode(
    is_inside: bool,
    gamma: float,
    trapdoor_config: TrapdoorConfig
) -> SignatureMode:
    """
    Determine signature mode based on geometric position and time dilation.

    Priority:
    1. If gamma > threshold → TRAPDOOR_FROZEN
    2. If inside sphere → KYBER_INTERNAL
    3. Otherwise → DILITHIUM_EXTERNAL
    """
    if gamma > trapdoor_config.gamma_threshold:
        return SignatureMode.TRAPDOOR_FROZEN
    elif is_inside:
        return SignatureMode.KYBER_INTERNAL
    else:
        return SignatureMode.DILITHIUM_EXTERNAL


def hypercube_brain_classify(
    context: np.ndarray,
    hypercube_config: HypercubeConfig = HypercubeConfig(),
    sphere_config: BrainSphereConfig = BrainSphereConfig(),
    trapdoor_config: TrapdoorConfig = TrapdoorConfig()
) -> GeometricState:
    """
    Main classification function for hypercube-brain security model.

    Process:
    1. Generate hypercube point v from context
    2. Generate sphere point u from context
    3. Compute risk → expansion factor
    4. Scale hypercube: v_scaled = v × expansion
    5. Project to sphere dimensions: v_proj = v_scaled[:3] mod 1
    6. Check intersection with sphere
    7. Compute time dilation from velocity
    8. Determine signature mode

    Args:
        context: Security context array [timestamp, location, risk, entropy, ...]
        hypercube_config: Hypercube parameters
        sphere_config: Brain sphere parameters
        trapdoor_config: Time dilation trapdoor parameters

    Returns:
        GeometricState with full classification
    """
    # Step 1: Generate seed from context
    seed = context_to_seed(context)

    # Step 2: Generate hypercube point
    v = generate_hypercube_point(seed, hypercube_config.dimensions)

    # Step 3: Generate sphere point
    u = generate_sphere_point(seed, sphere_config.dimensions)

    # Step 4: Compute risk and expansion
    risk = compute_risk_factor(context)
    expansion = compute_expansion_factor(risk, hypercube_config)

    # Step 5: Scale hypercube
    v_scaled = v * expansion

    # Step 6: Project to sphere dimensions (multi-modular)
    v_proj = project_to_sphere_dimensions(
        v_scaled,
        hypercube_config.modular_projection_dims
    )

    # Step 7: Check sphere intersection
    is_inside, rho, direction_diff = check_sphere_intersection(
        v_proj, u, sphere_config
    )

    # Step 8: Compute time dilation
    v_fraction = 0.5  # Default velocity
    if len(context) > trapdoor_config.velocity_index:
        v_fraction = min(context[trapdoor_config.velocity_index], 0.9999)
    gamma = compute_time_dilation(v_fraction, trapdoor_config.c)

    # Step 9: Determine signature mode
    mode = determine_signature_mode(is_inside, gamma, trapdoor_config)

    return GeometricState(
        hypercube_point=v,
        sphere_point=u,
        scaled_hypercube=v_scaled,
        projected_point=v_proj,
        radial_distance=rho,
        direction_alignment=direction_diff,
        is_inside=is_inside,
        gamma=gamma,
        signature_mode=mode,
        risk_factor=risk,
        expansion_factor=expansion
    )


# ============================================================================
# SIGNATURE OPERATIONS (Kyber/Dilithium simulation)
# ============================================================================

def create_kyber_commitment(
    shared_secret: bytes,
    state: GeometricState,
    timestamp: float
) -> bytes:
    """
    Create Kyber-style commitment for INTERNAL operations.

    Used when intersection is INSIDE the brain sphere.
    Binds the geometric state to the shared secret.
    """
    # Serialize geometric state
    state_data = (
        state.hypercube_point.tobytes() +
        state.sphere_point.tobytes() +
        struct.pack('<f', state.radial_distance) +
        struct.pack('<f', state.gamma) +
        struct.pack('<d', timestamp)
    )

    # Create commitment
    commitment = hashlib.sha256(
        b"kyber:internal:" + shared_secret + state_data
    ).digest()

    return commitment


def create_dilithium_signature(
    signing_key: bytes,
    message: bytes,
    state: GeometricState
) -> bytes:
    """
    Create Dilithium-style signature for EXTERNAL operations.

    Used when intersection is OUTSIDE the brain sphere.
    Provides stronger authentication for external boundaries.
    """
    # Serialize geometric state
    state_data = (
        state.projected_point.tobytes() +
        struct.pack('<f', state.direction_alignment) +
        struct.pack('<f', state.expansion_factor)
    )

    # Create signature using HMAC
    signature = hmac.new(
        signing_key,
        b"dilithium:external:" + message + state_data,
        'sha256'
    ).digest()

    return signature


def execute_trapdoor_puzzle(
    context: np.ndarray,
    secret: bytes,
    target_gamma: float = 1.5
) -> Tuple[bool, Optional[bytes], str]:
    """
    Execute the trapdoor puzzle.

    The puzzle requires finding a context that produces:
    1. gamma close to target_gamma (not too high, not too low)
    2. Valid geometric intersection

    This is computationally hard because:
    - gamma depends on velocity in context
    - Intersection depends on hash-derived positions
    - Both must align simultaneously

    Returns:
        (success, derived_key, message)
    """
    state = hypercube_brain_classify(context)

    # Check trapdoor
    if state.signature_mode == SignatureMode.TRAPDOOR_FROZEN:
        return (False, None, f"TRAPPED: gamma={state.gamma:.2f} exceeds threshold")

    # Check gamma is in acceptable range
    gamma_error = abs(state.gamma - target_gamma)
    if gamma_error > 0.3:
        return (False, None, f"FAILED: gamma={state.gamma:.2f} too far from target={target_gamma}")

    # Derive key based on mode
    if state.signature_mode == SignatureMode.KYBER_INTERNAL:
        key = create_kyber_commitment(secret, state, context[0])
        return (True, key, "SUCCESS: Internal operation via Kyber")
    else:
        key = create_dilithium_signature(secret, b"puzzle_solution", state)
        return (True, key, "SUCCESS: External operation via Dilithium")


# ============================================================================
# EXPANSION/RETRACTION OPERATIONS
# ============================================================================

def expand_hypercube(
    current_state: GeometricState,
    new_risk: float,
    config: HypercubeConfig
) -> GeometricState:
    """
    Expand hypercube in response to increased risk.

    Higher risk → larger hypercube → more restrictive policy enforcement
    """
    new_expansion = compute_expansion_factor(new_risk, config)

    # Re-scale from original point
    new_scaled = current_state.hypercube_point * new_expansion
    new_proj = project_to_sphere_dimensions(
        new_scaled,
        config.modular_projection_dims
    )

    # Re-check intersection
    sphere_config = BrainSphereConfig()
    is_inside, rho, direction_diff = check_sphere_intersection(
        new_proj,
        current_state.sphere_point,
        sphere_config
    )

    # Re-determine mode
    trapdoor_config = TrapdoorConfig()
    mode = determine_signature_mode(is_inside, current_state.gamma, trapdoor_config)

    return GeometricState(
        hypercube_point=current_state.hypercube_point,
        sphere_point=current_state.sphere_point,
        scaled_hypercube=new_scaled,
        projected_point=new_proj,
        radial_distance=rho,
        direction_alignment=direction_diff,
        is_inside=is_inside,
        gamma=current_state.gamma,
        signature_mode=mode,
        risk_factor=new_risk,
        expansion_factor=new_expansion
    )


def retract_hypercube(
    current_state: GeometricState,
    new_risk: float,
    config: HypercubeConfig
) -> GeometricState:
    """
    Retract hypercube in response to decreased risk.

    Lower risk → smaller hypercube → relaxed policy enforcement
    """
    # Same as expand, just with lower risk value
    return expand_hypercube(current_state, new_risk, config)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_state(state: GeometricState) -> str:
    """Generate ASCII visualization of geometric state"""
    lines = []
    lines.append("=" * 60)
    lines.append("HYPERCUBE-BRAIN GEOMETRIC STATE")
    lines.append("=" * 60)

    # Hypercube
    lines.append(f"\n📦 HYPERCUBE [0,1]^{len(state.hypercube_point)}")
    lines.append(f"   Original:  {np.array2string(state.hypercube_point, precision=3)}")
    lines.append(f"   Scaled:    {np.array2string(state.scaled_hypercube, precision=3)}")
    lines.append(f"   Expansion: {state.expansion_factor:.2f}x (risk={state.risk_factor:.2f})")

    # Sphere
    lines.append(f"\n🧠 BRAIN SPHERE S^{len(state.sphere_point)-1}")
    lines.append(f"   Point u:   {np.array2string(state.sphere_point, precision=3)}")
    lines.append(f"   ||u|| = {np.linalg.norm(state.sphere_point):.6f}")

    # Projection
    lines.append(f"\n📐 MULTI-MODULAR PROJECTION")
    lines.append(f"   v_proj:    {np.array2string(state.projected_point, precision=3)}")
    lines.append(f"   ρ (radial): {state.radial_distance:.4f}")
    lines.append(f"   Δ (align):  {state.direction_alignment:.4f}")

    # Intersection
    inside_str = "✓ INSIDE" if state.is_inside else "✗ OUTSIDE"
    lines.append(f"\n🎯 INTERSECTION: {inside_str}")

    # Time dilation
    lines.append(f"\n⏱️ TIME DILATION")
    lines.append(f"   γ = {state.gamma:.4f}")
    if state.gamma > 2.0:
        lines.append(f"   ⚠️  HIGH DILATION - Approaching trapdoor!")

    # Signature mode
    mode_emoji = {
        SignatureMode.KYBER_INTERNAL: "🔐",
        SignatureMode.DILITHIUM_EXTERNAL: "🔏",
        SignatureMode.TRAPDOOR_FROZEN: "🧊"
    }
    lines.append(f"\n{mode_emoji[state.signature_mode]} SIGNATURE MODE: {state.signature_mode.value.upper()}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("HYPERCUBE-BRAIN SECURITY MODEL DEMO")
    print("=" * 60)

    # Test contexts with different risk levels
    test_contexts = [
        ("Low risk, slow", np.array([1704700000.0, 101.0, 1.0, 0.45, 12.0, 0.3])),
        ("Medium risk, medium speed", np.array([1704700000.0, 101.0, 5.0, 0.45, 12.0, 0.5])),
        ("High risk, fast", np.array([1704700000.0, 101.0, 9.0, 0.45, 12.0, 0.8])),
        ("Trapdoor trigger", np.array([1704700000.0, 101.0, 5.0, 0.45, 12.0, 0.95])),
    ]

    for name, context in test_contexts:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"Context: {context}")

        state = hypercube_brain_classify(context)
        print(visualize_state(state))

        # Try puzzle
        success, key, msg = execute_trapdoor_puzzle(context, b"secret_key")
        print(f"\nPuzzle Result: {msg}")
        if key:
            print(f"Derived Key: {key.hex()[:32]}...")
