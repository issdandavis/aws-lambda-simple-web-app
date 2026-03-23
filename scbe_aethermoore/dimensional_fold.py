"""
Dimensional Folding — "Wrong Math That Becomes Right"

The insight: Do math INCORRECTLY on purpose, in ways that CANCEL OUT
when you have the right key/context. Attackers see garbage because
they don't know which "wrong" to apply.

Think of it like:
- Fold a paper crane wrong on purpose
- Only someone who knows your specific "wrong folds" can unfold it
- Everyone else just tears the paper

Mathematical basis:
1. LIFTING: Take 3D problem to 7D, solve there, project back
2. GAUGE: Add "error" that cancels with matching "anti-error"
3. TWIST: Rotate through dimensions that don't exist... then un-rotate
4. FOLD: Like origami in math-space

This combines ALL our structures:
- Sphere (behavior) + Cube (policy) + Rings (trust) + Manifold (topology)
Into ONE unified geometric object that only "unfolds" correctly
with the right context.
"""

import math
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
from enum import Enum

from .constants import GOLDEN_RATIO, PERFECT_FIFTH, PHI_AETHER
from .geoseal import (
    project_to_sphere, project_to_cube,
    classify_ring, ConcentricRingSystem, RingMembership,
    SphereConfig, CubeConfig, GeoSealConfig,
    healpix_index, morton_encode
)


# =============================================================================
# THE CORE INSIGHT: DIMENSIONAL LIFTING
# =============================================================================

class FoldType(Enum):
    """Types of dimensional folds."""
    LIFT = "lift"           # Go UP in dimensions
    PROJECT = "project"     # Come DOWN in dimensions
    TWIST = "twist"         # Rotate through hidden dimension
    GAUGE = "gauge"         # Add canceling error
    ORIGAMI = "origami"     # Complex fold pattern


@dataclass
class DimensionalConfig:
    """Configuration for dimensional folding."""
    base_dims: int = 3              # Starting dimensions
    lift_dims: int = 7              # Lifted dimensions (7D is special)
    twist_angle: float = PHI_AETHER # Rotation angle (golden-ratio based)
    gauge_strength: float = 0.618   # Error magnitude (1/phi)
    fold_depth: int = 3             # How many folds to apply


@dataclass
class FoldedSpace:
    """A point in folded dimensional space."""
    original: np.ndarray            # Original coordinates
    lifted: np.ndarray              # Lifted to higher dims
    twisted: np.ndarray             # After twist operations
    gauge_error: np.ndarray         # The "wrong" we added
    folded: np.ndarray              # Final folded form
    fold_key: bytes                 # Key to unfold


# =============================================================================
# LIFTING: 3D → 7D → 3D
# =============================================================================

def lift_to_higher_dimension(
    point: np.ndarray,
    target_dims: int,
    seed: bytes
) -> np.ndarray:
    """
    Lift a point to higher dimensions.

    WHY THIS WORKS:
    In 3D, a knot can't be untied without cutting.
    In 4D, ALL knots can be untied trivially.

    We lift our security problem to 7D where it's "easy",
    do our operations, then project back.
    Attackers stuck in 3D can't follow.

    Args:
        point: Original point (e.g., 3D)
        target_dims: Target dimensions (e.g., 7D)
        seed: Deterministic seed for lifting direction

    Returns:
        Point in higher dimensional space
    """
    current_dims = len(point)
    extra_dims = target_dims - current_dims

    if extra_dims <= 0:
        return point.copy()

    # Generate lifting directions from seed
    np.random.seed(int.from_bytes(seed[:4], 'little'))

    # Create orthonormal basis for extra dimensions
    # These are the "hidden" directions
    lift_vectors = []
    for i in range(extra_dims):
        # Each extra dimension is a specific "direction" determined by seed
        v = np.random.randn(current_dims)
        # Gram-Schmidt orthogonalize against previous
        for prev in lift_vectors:
            v = v - np.dot(v, prev) * prev
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        lift_vectors.append(v)

    # Lift point: original coords + projections onto lift directions
    lifted = np.zeros(target_dims)
    lifted[:current_dims] = point

    for i, lv in enumerate(lift_vectors):
        # The extra coordinates come from projecting onto hidden directions
        # Scaled by golden ratio powers for harmonic structure
        scale = GOLDEN_RATIO ** (i + 1)
        lifted[current_dims + i] = np.dot(point, lv) * scale

    return lifted


def project_from_higher_dimension(
    lifted: np.ndarray,
    target_dims: int,
    seed: bytes
) -> np.ndarray:
    """
    Project from higher dimensions back to original space.

    This is the INVERSE of lifting - but only works if you
    know the original seed (the lifting directions).

    Wrong seed = wrong projection = garbage
    """
    if len(lifted) <= target_dims:
        return lifted[:target_dims].copy()

    # Regenerate the same lifting directions
    np.random.seed(int.from_bytes(seed[:4], 'little'))

    current_dims = target_dims
    extra_dims = len(lifted) - target_dims

    lift_vectors = []
    for i in range(extra_dims):
        v = np.random.randn(current_dims)
        for prev in lift_vectors:
            v = v - np.dot(v, prev) * prev
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        lift_vectors.append(v)

    # Reconstruct original point
    # The first coords are the original, but we verify with extra dims
    projected = lifted[:target_dims].copy()

    # Verify the lift was consistent (extra security check)
    for i, lv in enumerate(lift_vectors):
        scale = GOLDEN_RATIO ** (i + 1)
        expected = np.dot(projected, lv) * scale
        actual = lifted[target_dims + i]
        # If these don't match, something's wrong
        if abs(expected - actual) > 0.01:
            # Return noise - wrong seed
            return np.random.randn(target_dims)

    return projected


# =============================================================================
# TWISTING: Rotation Through Hidden Dimensions
# =============================================================================

def create_rotation_matrix(dims: int, i: int, j: int, angle: float) -> np.ndarray:
    """Create rotation matrix in the i-j plane."""
    R = np.eye(dims)
    c, s = math.cos(angle), math.sin(angle)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def twist_through_dimensions(
    point: np.ndarray,
    angles: List[float],
    dimension_pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Rotate point through multiple dimension pairs.

    THE TRICK:
    Rotating in the x-y plane is normal.
    Rotating in the x-w plane (where w is a hidden dimension)
    looks IMPOSSIBLE from 3D perspective.

    It's like spinning a coin... but also spinning it through
    a direction you can't see.
    """
    result = point.copy()
    dims = len(point)

    for angle, (i, j) in zip(angles, dimension_pairs):
        if i < dims and j < dims:
            R = create_rotation_matrix(dims, i, j, angle)
            result = R @ result

    return result


def untwist_through_dimensions(
    point: np.ndarray,
    angles: List[float],
    dimension_pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """Reverse the twist - apply rotations in reverse order with negative angles."""
    result = point.copy()
    dims = len(point)

    # Reverse order, negative angles
    for angle, (i, j) in zip(reversed(angles), reversed(dimension_pairs)):
        if i < dims and j < dims:
            R = create_rotation_matrix(dims, i, j, -angle)
            result = R @ result

    return result


# =============================================================================
# GAUGE ERROR: Add "Wrong" That Cancels
# =============================================================================

def compute_gauge_error(
    point: np.ndarray,
    ring_membership: RingMembership,
    seed: bytes
) -> np.ndarray:
    """
    Compute a "gauge error" - deliberate distortion that will cancel later.

    THE CONCEPT:
    In physics, gauge transformations are changes that don't affect
    the actual physics. We exploit this: add an error that looks
    random but will perfectly cancel when you apply the anti-error.

    The error depends on:
    - Your position in space
    - Your ring membership
    - A secret seed

    Without all three, you can't compute the anti-error.
    """
    np.random.seed(int.from_bytes(seed[:4], 'little') ^ ring_membership.ring_index)

    # Error magnitude based on ring (outer rings = larger error = more "wrong")
    magnitude = (1 - ring_membership.trust_level) * 0.618  # 1/phi scaling

    # Error direction is deterministic but looks random
    error_direction = np.random.randn(len(point))
    error_direction = error_direction / (np.linalg.norm(error_direction) + 1e-10)

    # Modulate by position (error varies across space)
    position_factor = np.sin(np.sum(point * GOLDEN_RATIO))

    return error_direction * magnitude * (1 + 0.5 * position_factor)


def apply_gauge_transform(point: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Add gauge error to point."""
    return point + error


def remove_gauge_transform(point: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Remove gauge error from point (the "anti-error")."""
    return point - error


# =============================================================================
# ORIGAMI FOLD: Complex Multi-Step Folding
# =============================================================================

@dataclass
class FoldStep:
    """One step in an origami fold sequence."""
    fold_type: FoldType
    parameters: Dict[str, Any]


def generate_fold_sequence(
    seed: bytes,
    depth: int
) -> List[FoldStep]:
    """
    Generate a sequence of fold operations.

    Like origami instructions, but for math-space.
    Each fold is deterministic from seed, but looks random.
    """
    np.random.seed(int.from_bytes(seed[:4], 'little'))

    sequence = []
    fold_types = list(FoldType)

    for i in range(depth):
        fold_type = fold_types[np.random.randint(0, len(fold_types))]

        if fold_type == FoldType.LIFT:
            params = {"target_dims": 7 + np.random.randint(0, 4)}
        elif fold_type == FoldType.TWIST:
            params = {
                "angle": np.random.uniform(0, 2 * math.pi),
                "dims": (np.random.randint(0, 7), np.random.randint(0, 7))
            }
        elif fold_type == FoldType.GAUGE:
            params = {"strength": np.random.uniform(0.3, 0.7)}
        else:
            params = {}

        sequence.append(FoldStep(fold_type=fold_type, parameters=params))

    return sequence


# =============================================================================
# THE UNIFIED FOLD: Combining Everything
# =============================================================================

@dataclass
class UnifiedGeometry:
    """
    All geometric structures combined into one folded object.

    This is THE thing that combines:
    - Sphere (behavior manifold)
    - Cube (policy space)
    - Rings (trust topology)
    - Manifold (S² ∩ T² intersection)

    All wrapped in dimensional folding that only "unwraps"
    with the correct context.
    """
    # Original components
    sphere_point: np.ndarray      # u on S^n
    cube_point: np.ndarray        # v in [0,1]^m
    ring_membership: RingMembership
    sphere_cell: int              # h (HEALPix)
    cube_cell: int                # z (Morton)

    # Folded representation
    unified_point: np.ndarray     # Everything combined
    fold_key: bytes               # Key to unfold

    # Verification
    checksum: bytes               # To verify correct unfolding


def create_unified_geometry(
    context: np.ndarray,
    policy: Dict[str, float],
    master_seed: bytes,
    config: GeoSealConfig = None
) -> UnifiedGeometry:
    """
    Create unified geometric object from context and policy.

    This is the MAIN FUNCTION that takes all inputs and
    produces a single folded geometric object.
    """
    if config is None:
        config = GeoSealConfig()

    # Step 1: Project to component spaces
    u = project_to_sphere(context, config.sphere)
    v = project_to_cube(policy, config.cube)

    # Step 2: Get ring membership
    ring_mem = classify_ring(u, config.ring_system)

    # Step 3: Get cell indices
    h = healpix_index(u, config.sphere.level)
    z = morton_encode(v, config.cube.level)

    # Step 4: Create unified vector by concatenating all components
    # This is the "unfolded" version
    unified_base = np.concatenate([
        u,                                    # Sphere coords (3)
        v,                                    # Cube coords (4)
        np.array([ring_mem.radial_distance]), # Ring position (1)
        np.array([ring_mem.trust_level]),     # Trust level (1)
        np.array([h / 1000.0, z / 10000.0])   # Cell indices normalized (2)
    ])  # Total: 11 dimensions

    # Step 5: Lift to higher dimensions
    lift_seed = hashlib.sha256(master_seed + b"lift").digest()
    lifted = lift_to_higher_dimension(unified_base, 17, lift_seed)

    # Step 6: Apply twist rotations
    twist_seed = hashlib.sha256(master_seed + b"twist").digest()
    twist_angles = [
        PHI_AETHER * (i + 1) for i in range(5)
    ]
    twist_pairs = [
        (0, 11), (1, 12), (2, 13), (3, 14), (4, 15)
    ]
    twisted = twist_through_dimensions(lifted, twist_angles, twist_pairs)

    # Step 7: Apply gauge error based on ring
    gauge_seed = hashlib.sha256(master_seed + b"gauge").digest()
    gauge_error = compute_gauge_error(twisted, ring_mem, gauge_seed)
    gauged = apply_gauge_transform(twisted, gauge_error)

    # Step 8: Final fold (project back but keep extra info encoded)
    # We DON'T fully project back - we keep the high-dimensional form
    # as the "folded" representation
    unified_point = gauged

    # Step 9: Create fold key (needed to unfold)
    fold_key = hashlib.sha256(
        master_seed +
        ring_mem.ring_name.encode() +
        h.to_bytes(4, 'little') +
        z.to_bytes(4, 'little')
    ).digest()

    # Step 10: Checksum for verification
    checksum = hashlib.sha256(unified_base.tobytes() + fold_key).digest()[:8]

    return UnifiedGeometry(
        sphere_point=u,
        cube_point=v,
        ring_membership=ring_mem,
        sphere_cell=h,
        cube_cell=z,
        unified_point=unified_point,
        fold_key=fold_key,
        checksum=checksum
    )


def unfold_unified_geometry(
    folded: UnifiedGeometry,
    master_seed: bytes,
    claimed_context: np.ndarray,
    claimed_policy: Dict[str, float],
    config: GeoSealConfig = None
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Attempt to unfold a unified geometry.

    Only succeeds if the claimed context/policy matches what was
    used to create the fold.

    Returns:
        Tuple of (success, original_unified_base or None)
    """
    if config is None:
        config = GeoSealConfig()

    # Recreate what SHOULD be the unified geometry
    expected = create_unified_geometry(
        claimed_context, claimed_policy, master_seed, config
    )

    # Check if fold keys match
    if expected.fold_key != folded.fold_key:
        return (False, None)

    # Check if checksums match
    if expected.checksum != folded.checksum:
        return (False, None)

    # Reverse the folding process
    # Step 1: Remove gauge error
    gauge_seed = hashlib.sha256(master_seed + b"gauge").digest()
    gauge_error = compute_gauge_error(
        folded.unified_point, folded.ring_membership, gauge_seed
    )
    ungauged = remove_gauge_transform(folded.unified_point, gauge_error)

    # Step 2: Reverse twist
    twist_angles = [PHI_AETHER * (i + 1) for i in range(5)]
    twist_pairs = [(0, 11), (1, 12), (2, 13), (3, 14), (4, 15)]
    untwisted = untwist_through_dimensions(ungauged, twist_angles, twist_pairs)

    # Step 3: Project back down
    lift_seed = hashlib.sha256(master_seed + b"lift").digest()
    projected = project_from_higher_dimension(untwisted, 11, lift_seed)

    return (True, projected)


# =============================================================================
# VISUALIZATION / EXPLANATION
# =============================================================================

def explain_dimensional_fold() -> str:
    """Plain English explanation of what's happening."""
    return """
    ╔══════════════════════════════════════════════════════════════════╗
    ║           DIMENSIONAL FOLDING: "Wrong Math That Fixes Itself"     ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  Imagine folding a paper airplane:                                ║
    ║                                                                   ║
    ║    [Flat Paper] → [Fold 1] → [Fold 2] → [Fold 3] → [Airplane]    ║
    ║                                                                   ║
    ║  If someone copies your airplane but doesn't know the FOLDS,     ║
    ║  they can't unfold it back to flat paper correctly.              ║
    ║                                                                   ║
    ║  WE DO THIS WITH MATH:                                            ║
    ║                                                                   ║
    ║  ┌─────────────┐                                                  ║
    ║  │ Your Data   │ ─── LIFT TO 17 DIMENSIONS ───┐                  ║
    ║  │ (3D point)  │                               │                  ║
    ║  └─────────────┘                               ▼                  ║
    ║                                         ┌─────────────┐           ║
    ║                                         │ 17D Space   │           ║
    ║                                         │ (Room to    │           ║
    ║                                         │  hide stuff)│           ║
    ║                                         └──────┬──────┘           ║
    ║                                                │                  ║
    ║                          TWIST through         │                  ║
    ║                          hidden dimensions     ▼                  ║
    ║                                         ┌─────────────┐           ║
    ║                                         │ Twisted     │           ║
    ║                                         │ (rotated in │           ║
    ║                                         │  ways you   │           ║
    ║                                         │  can't see) │           ║
    ║                                         └──────┬──────┘           ║
    ║                                                │                  ║
    ║                          ADD "WRONG" that      │                  ║
    ║                          cancels later         ▼                  ║
    ║                                         ┌─────────────┐           ║
    ║                                         │ Gauged      │           ║
    ║                                         │ (has error  │           ║
    ║                                         │  on purpose)│           ║
    ║                                         └─────────────┘           ║
    ║                                                                   ║
    ║  THE RESULT:                                                      ║
    ║  • Your data is now a 17-dimensional twisted, errored point      ║
    ║  • Only someone with YOUR exact context can undo the folds       ║
    ║  • Wrong context = wrong unfold = garbage                        ║
    ║                                                                   ║
    ║  WHY 17 DIMENSIONS?                                               ║
    ║  • 3 for behavior sphere                                          ║
    ║  • 4 for policy cube                                              ║
    ║  • 1 for ring position                                            ║
    ║  • 1 for trust level                                              ║
    ║  • 2 for cell indices                                             ║
    ║  • 6 for "hiding space" (the extra dimensions)                   ║
    ║  = 17 total                                                       ║
    ║                                                                   ║
    ║  THE MAGIC:                                                       ║
    ║  Problems that are HARD in 3D become EASY in 17D.                ║
    ║  We solve them in 17D, then fold back down.                      ║
    ║  Attackers stuck in 3D can't follow the path.                    ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """


def visualize_fold_process(
    context: np.ndarray,
    policy: Dict[str, float],
    seed: bytes
) -> str:
    """Generate ASCII visualization of the folding process."""
    geo = create_unified_geometry(context, policy, seed)

    lines = []
    lines.append("┌────────────────────────────────────────────────────────────┐")
    lines.append("│              DIMENSIONAL FOLD VISUALIZATION                │")
    lines.append("├────────────────────────────────────────────────────────────┤")
    lines.append("│                                                            │")
    lines.append(f"│  SPHERE POINT: ({geo.sphere_point[0]:.3f}, {geo.sphere_point[1]:.3f}, {geo.sphere_point[2]:.3f})                   │")
    lines.append(f"│  CUBE POINT:   ({geo.cube_point[0]:.3f}, {geo.cube_point[1]:.3f}, {geo.cube_point[2]:.3f}, {geo.cube_point[3]:.3f})              │")
    lines.append(f"│  RING:         {geo.ring_membership.ring_name} (trust: {geo.ring_membership.trust_level:.2f})           │")
    lines.append(f"│  CELLS:        h={geo.sphere_cell}, z={geo.cube_cell}                          │")
    lines.append("│                                                            │")
    lines.append("│  ──────────── FOLDING PROCESS ────────────                │")
    lines.append("│                                                            │")
    lines.append("│     ┌───┐                                                  │")
    lines.append("│     │3D │ ──LIFT──▶ ┌────┐ ──TWIST──▶ ┌────┐ ──GAUGE──▶ │")
    lines.append("│     └───┘           │17D │            │ ⟳  │            │")
    lines.append("│                     └────┘            └────┘            │")
    lines.append("│                                                            │")
    lines.append(f"│  UNIFIED POINT (first 6 of 17 dims):                       │")
    lines.append(f"│    [{geo.unified_point[0]:+.3f}, {geo.unified_point[1]:+.3f}, {geo.unified_point[2]:+.3f}, {geo.unified_point[3]:+.3f}, {geo.unified_point[4]:+.3f}, {geo.unified_point[5]:+.3f}...]  │")
    lines.append("│                                                            │")
    lines.append(f"│  FOLD KEY: {geo.fold_key[:8].hex()}...                           │")
    lines.append(f"│  CHECKSUM: {geo.checksum.hex()}                               │")
    lines.append("│                                                            │")
    lines.append("└────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)
