"""
GeoSeal v0.1 - Geometric Trust Kernel

Dual-manifold authorization system mapping interactions to:
- State Sphere (S²): AI cognition/behavior space
- Policy Hypercube ([0,1]^m): Governance rules

Authorization is granted if sphere coordinate intersects hypercube cell.

Features:
- Dual-Lane Cryptography (K_in, K_out, K_∩)
- Time-Dilation Scaling
- Multi-Scale Tiling (HEALPix-style for sphere)
"""

import math
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Time dilation parameters
TAU_0 = 1.0          # Base latency budget (seconds)
GAMMA_DILATION = 0.5  # Dilation decay rate
POW_0 = 8            # Base PoW bits
KAPPA = 4            # PoW scaling factor
R_0 = 0.5            # PoW threshold radius

# Sphere resolution
DEFAULT_NSIDE = 8    # HEALPix-style resolution


# ═══════════════════════════════════════════════════════════════
# Sphere Geometry (S²)
# ═══════════════════════════════════════════════════════════════

@dataclass
class SphericalCoord:
    """Point on unit sphere in spherical coordinates."""
    theta: float  # Polar angle [0, π]
    phi: float    # Azimuthal angle [0, 2π]

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert to Cartesian (x, y, z)."""
        x = math.sin(self.theta) * math.cos(self.phi)
        y = math.sin(self.theta) * math.sin(self.phi)
        z = math.cos(self.theta)
        return (x, y, z)

    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float) -> "SphericalCoord":
        """Create from Cartesian coordinates."""
        r = math.sqrt(x*x + y*y + z*z)
        if r < 1e-10:
            return cls(0.0, 0.0)

        theta = math.acos(max(-1, min(1, z / r)))
        phi = math.atan2(y, x)
        if phi < 0:
            phi += 2 * math.pi

        return cls(theta, phi)

    def radial_distance(self) -> float:
        """
        Radial distance from sphere center (always 1 for unit sphere).
        Extended to support Poincaré ball embedding.
        """
        return 1.0


@dataclass
class SphereCell:
    """A cell on the sphere (HEALPix-style tile)."""
    index: int
    nside: int
    center: SphericalCoord

    @property
    def area(self) -> float:
        """Approximate area of cell."""
        total_cells = 12 * self.nside * self.nside
        return 4 * math.pi / total_cells


def sphere_to_cell(coord: SphericalCoord, nside: int = DEFAULT_NSIDE) -> int:
    """
    Map spherical coordinate to cell index (simplified HEALPix).

    Returns cell index in [0, 12*nside²).
    """
    # Simplified: divide sphere into latitude bands and longitude sectors
    n_lat = 3 * nside
    n_lon = 4 * nside

    lat_idx = int(coord.theta / math.pi * n_lat)
    lat_idx = max(0, min(n_lat - 1, lat_idx))

    lon_idx = int(coord.phi / (2 * math.pi) * n_lon)
    lon_idx = max(0, min(n_lon - 1, lon_idx))

    return lat_idx * n_lon + lon_idx


# ═══════════════════════════════════════════════════════════════
# Hypercube Geometry ([0,1]^m)
# ═══════════════════════════════════════════════════════════════

@dataclass
class HypercubeCoord:
    """Point in m-dimensional hypercube [0,1]^m."""
    values: Tuple[float, ...]

    def __post_init__(self):
        for v in self.values:
            if not 0 <= v <= 1:
                raise ValueError(f"Hypercube coords must be in [0,1], got {v}")

    @property
    def dim(self) -> int:
        return len(self.values)

    def cell_index(self, resolution: int = 4) -> int:
        """
        Map to cell index using Morton code (Z-order curve).

        resolution: divisions per dimension
        """
        indices = [int(v * resolution) for v in self.values]
        indices = [min(i, resolution - 1) for i in indices]

        # Simple interleaving for Morton code
        code = 0
        for bit in range(10):  # Up to 10 bits per dimension
            for d, idx in enumerate(indices):
                if idx & (1 << bit):
                    code |= 1 << (bit * len(indices) + d)

        return code


@dataclass
class PolicyCell:
    """A cell in the policy hypercube."""
    index: int
    bounds_min: Tuple[float, ...]
    bounds_max: Tuple[float, ...]

    def contains(self, coord: HypercubeCoord) -> bool:
        """Check if coordinate is within this cell."""
        for v, lo, hi in zip(coord.values, self.bounds_min, self.bounds_max):
            if not lo <= v <= hi:
                return False
        return True


# ═══════════════════════════════════════════════════════════════
# Dual-Manifold Intersection
# ═══════════════════════════════════════════════════════════════

@dataclass
class GeoSealState:
    """Combined state on both manifolds."""
    sphere: SphericalCoord
    hypercube: HypercubeCoord
    timestamp: float = 0.0

    def radial_distance(self) -> float:
        """
        Effective radial distance for time dilation.

        Combines sphere position with hypercube distance from center.
        """
        # Hypercube distance from center (0.5, 0.5, ...)
        center = [0.5] * self.hypercube.dim
        hc_dist = math.sqrt(sum(
            (v - c) ** 2 for v, c in zip(self.hypercube.values, center)
        ))

        # Normalize to [0, 1]
        max_hc_dist = math.sqrt(self.hypercube.dim * 0.25)
        return hc_dist / max_hc_dist


class IntersectionType(Enum):
    """Types of manifold intersections."""
    NONE = "none"           # No intersection - denied
    INNER = "inner"         # Inside both - standard auth
    OUTER = "outer"         # Edge of sphere - enhanced auth
    BOUNDARY = "boundary"   # Intersection boundary - critical auth


@dataclass
class IntersectionResult:
    """Result of checking manifold intersection."""
    type: IntersectionType
    sphere_cell: int
    hypercube_cell: int
    radial_distance: float
    authorized: bool


def check_intersection(
    state: GeoSealState,
    policy_cells: List[PolicyCell],
    nside: int = DEFAULT_NSIDE,
) -> IntersectionResult:
    """
    Check if state intersects with any authorized policy cell.
    """
    sphere_cell = sphere_to_cell(state.sphere, nside)
    hc_cell = state.hypercube.cell_index()
    r = state.radial_distance()

    # Check each policy cell
    for policy in policy_cells:
        if policy.contains(state.hypercube):
            # Determine intersection type based on radial distance
            if r < 0.3:
                int_type = IntersectionType.INNER
            elif r < 0.7:
                int_type = IntersectionType.OUTER
            else:
                int_type = IntersectionType.BOUNDARY

            return IntersectionResult(
                type=int_type,
                sphere_cell=sphere_cell,
                hypercube_cell=hc_cell,
                radial_distance=r,
                authorized=True,
            )

    return IntersectionResult(
        type=IntersectionType.NONE,
        sphere_cell=sphere_cell,
        hypercube_cell=hc_cell,
        radial_distance=r,
        authorized=False,
    )


# ═══════════════════════════════════════════════════════════════
# Dual-Lane Cryptography
# ═══════════════════════════════════════════════════════════════

@dataclass
class DualLaneKeys:
    """Keys derived from geometric state."""
    k_inner: bytes     # Brain ops key (from sphere)
    k_outer: bytes     # Oversight key (from hypercube)
    k_composite: bytes  # Critical ops key (intersection)


def derive_dual_lane_keys(
    master_secret: bytes,
    state: GeoSealState,
    shared_secret: bytes = b"",
) -> DualLaneKeys:
    """
    Derive dual-lane keys from geometric state.

    K_in: From sphere geometry (brain operations)
    K_out: From hypercube geometry (oversight)
    K_∩: Composite for critical operations (boundary intersection)
    """
    sphere_cell = sphere_to_cell(state.sphere)
    hc_cell = state.hypercube.cell_index()

    # Inner key from sphere
    k_inner = hmac.new(
        master_secret,
        f"geoseal:inner:{sphere_cell}".encode() + shared_secret,
        hashlib.sha256,
    ).digest()

    # Outer key from hypercube
    k_outer = hmac.new(
        master_secret,
        f"geoseal:outer:{hc_cell}".encode() + shared_secret,
        hashlib.sha256,
    ).digest()

    # Composite key (XOR of both, then re-hash)
    xor_key = bytes(a ^ b for a, b in zip(k_inner, k_outer))
    k_composite = hmac.new(
        master_secret,
        b"geoseal:composite:" + xor_key + shared_secret,
        hashlib.sha256,
    ).digest()

    return DualLaneKeys(
        k_inner=k_inner,
        k_outer=k_outer,
        k_composite=k_composite,
    )


def select_key_for_operation(
    keys: DualLaneKeys,
    intersection: IntersectionResult,
) -> bytes:
    """Select appropriate key based on intersection type."""
    if intersection.type == IntersectionType.INNER:
        return keys.k_inner
    elif intersection.type == IntersectionType.OUTER:
        return keys.k_outer
    elif intersection.type == IntersectionType.BOUNDARY:
        return keys.k_composite
    else:
        raise ValueError("No key available for unauthorized state")


# ═══════════════════════════════════════════════════════════════
# Time-Dilation Scaling
# ═══════════════════════════════════════════════════════════════

@dataclass
class TimeDilationResult:
    """Result of time dilation calculation."""
    tau_allowed: float    # Allowed latency budget
    pow_bits: int         # Required PoW bits
    radial_distance: float


def compute_time_dilation(
    state: GeoSealState,
    tau_0: float = TAU_0,
    gamma: float = GAMMA_DILATION,
    pow_0: int = POW_0,
    kappa: int = KAPPA,
    r_0: float = R_0,
) -> TimeDilationResult:
    """
    Compute time dilation based on radial distance.

    τ_allow = τ₀ · exp(-γ · r)
    PoW_bits = pow₀ + κ · max(0, r - r₀)

    Entities far from center get less time and more PoW requirements.
    """
    r = state.radial_distance()

    # Latency shrinks exponentially with distance
    tau_allowed = tau_0 * math.exp(-gamma * r)

    # PoW increases linearly beyond threshold
    pow_bits = pow_0 + int(kappa * max(0, r - r_0))

    return TimeDilationResult(
        tau_allowed=tau_allowed,
        pow_bits=pow_bits,
        radial_distance=r,
    )


def verify_timing(
    elapsed: float,
    dilation: TimeDilationResult,
) -> bool:
    """Verify operation completed within allowed time."""
    return elapsed <= dilation.tau_allowed


# ═══════════════════════════════════════════════════════════════
# GeoSeal Engine
# ═══════════════════════════════════════════════════════════════

class GeoSealEngine:
    """
    Main GeoSeal engine combining all components.
    """

    def __init__(
        self,
        master_secret: bytes,
        policy_cells: Optional[List[PolicyCell]] = None,
        nside: int = DEFAULT_NSIDE,
    ):
        self.master_secret = master_secret
        self.nside = nside

        # Default policy: allow center region
        if policy_cells is None:
            self.policy_cells = [
                PolicyCell(
                    index=0,
                    bounds_min=(0.2, 0.2, 0.2),
                    bounds_max=(0.8, 0.8, 0.8),
                ),
            ]
        else:
            self.policy_cells = policy_cells

    def authorize(
        self,
        state: GeoSealState,
        shared_secret: bytes = b"",
    ) -> Tuple[IntersectionResult, Optional[DualLaneKeys], TimeDilationResult]:
        """
        Full authorization check.

        Returns:
            - Intersection result
            - Keys (if authorized)
            - Time dilation parameters
        """
        # Check intersection
        intersection = check_intersection(state, self.policy_cells, self.nside)

        # Compute time dilation
        dilation = compute_time_dilation(state)

        # Derive keys if authorized
        keys = None
        if intersection.authorized:
            keys = derive_dual_lane_keys(
                self.master_secret,
                state,
                shared_secret,
            )

        return intersection, keys, dilation

    def create_state(
        self,
        theta: float,
        phi: float,
        policy_coords: Tuple[float, ...],
        timestamp: float = 0.0,
    ) -> GeoSealState:
        """Helper to create GeoSealState."""
        return GeoSealState(
            sphere=SphericalCoord(theta, phi),
            hypercube=HypercubeCoord(policy_coords),
            timestamp=timestamp,
        )


# ═══════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════

def quick_authorize(
    master_secret: bytes,
    theta: float,
    phi: float,
    policy_coords: Tuple[float, ...],
) -> bool:
    """Quick authorization check."""
    engine = GeoSealEngine(master_secret)
    state = engine.create_state(theta, phi, policy_coords)
    intersection, _, _ = engine.authorize(state)
    return intersection.authorized
