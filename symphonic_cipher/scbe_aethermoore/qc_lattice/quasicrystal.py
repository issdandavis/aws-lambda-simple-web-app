"""
Icosahedral Quasicrystal Module - AETHERMOORE Integration

Implements 6D→3D aperiodic projection using icosahedral symmetry.
Quasicrystals discovered by Dan Shechtman (1982, Nobel Prize 2011)
provide mathematically precise structures without periodic repetition.

Key Properties:
1. 5-fold rotational symmetry (impossible in periodic crystals)
2. Aperiodic tiling - no repeating unit cell
3. Sharp diffraction peaks - long-range order without periodicity
4. Self-similarity at multiple scales
5. Projection from 6D hypercubic lattice

For AETHERMOORE:
- 6D state vector projects to 3D "crystallographic" signature
- Aperiodic structure prevents pattern-based attacks
- Icosahedral symmetry provides verification checkpoints
- Diffraction pattern serves as state fingerprint

Mathematical Foundation:
The projection uses the "cut-and-project" method:
- Start with Z⁶ (6D integer lattice)
- Project to 3D "physical space" E∥
- Points within a 3D "window" W in perpendicular space E⊥ are kept

Document ID: AETHER-QC-2026-001
Version: 1.0.0
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..constants import PHI


# =============================================================================
# CONSTANTS
# =============================================================================

# Golden ratio τ (same as φ)
TAU = PHI  # ≈ 1.6180339887

# Icosahedral symmetry constants
# The icosahedron has 12 vertices, 30 edges, 20 faces
# Symmetry group: I_h (order 120)
ICOSAHEDRAL_VERTICES_3D = np.array([
    [0, 1, TAU], [0, 1, -TAU], [0, -1, TAU], [0, -1, -TAU],
    [1, TAU, 0], [1, -TAU, 0], [-1, TAU, 0], [-1, -TAU, 0],
    [TAU, 0, 1], [TAU, 0, -1], [-TAU, 0, 1], [-TAU, 0, -1]
]) / math.sqrt(1 + TAU**2)

# 6D to 3D projection matrix (cut-and-project)
# This projects the 6D hypercubic lattice to 3D with icosahedral symmetry
# Columns are unit vectors pointing to icosahedron vertices
_c1 = 1 / math.sqrt(1 + TAU**2)
_c2 = TAU / math.sqrt(1 + TAU**2)

ICOSAHEDRAL_MATRIX = np.array([
    [_c1, _c2, 0, _c1, -_c2, 0],      # x
    [_c2, 0, _c1, -_c2, 0, _c1],      # y
    [0, _c1, _c2, 0, _c1, -_c2]       # z
])

# Perpendicular projection (to E⊥ for windowing)
PERPENDICULAR_MATRIX = np.array([
    [_c1, -_c2, 0, _c1, _c2, 0],
    [-_c2, 0, _c1, _c2, 0, _c1],
    [0, _c1, -_c2, 0, _c1, _c2]
])

# Window radius for cut-and-project (triacontahedron)
WINDOW_RADIUS = 1.0


# =============================================================================
# QUASICRYSTAL VERTEX
# =============================================================================

@dataclass
class QuasicrystalVertex:
    """
    A vertex in the quasicrystal structure.

    Contains both the 6D lattice coordinates and the
    3D projected position.
    """
    lattice_6d: np.ndarray  # Integer coordinates in Z⁶
    position_3d: np.ndarray  # Projected position in E∥
    perp_3d: np.ndarray  # Position in E⊥ (perpendicular space)
    distance_from_origin: float
    index: int

    def __hash__(self):
        return hash(tuple(self.lattice_6d))

    def __eq__(self, other):
        if not isinstance(other, QuasicrystalVertex):
            return False
        return np.array_equal(self.lattice_6d, other.lattice_6d)


# =============================================================================
# ICOSAHEDRAL PROJECTOR
# =============================================================================

class IcosahedralProjector:
    """
    Projects 6D vectors to 3D using icosahedral symmetry.

    The projection preserves the aperiodic, self-similar structure
    of the quasicrystal while mapping to physically meaningful 3D.
    """

    def __init__(
        self,
        window_radius: float = WINDOW_RADIUS,
        parallel_matrix: Optional[np.ndarray] = None,
        perp_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize projector.

        Args:
            window_radius: Radius of acceptance window in E⊥
            parallel_matrix: Custom 3×6 projection matrix to E∥
            perp_matrix: Custom 3×6 projection matrix to E⊥
        """
        self.window_radius = window_radius
        self.parallel_matrix = parallel_matrix if parallel_matrix is not None else ICOSAHEDRAL_MATRIX
        self.perp_matrix = perp_matrix if perp_matrix is not None else PERPENDICULAR_MATRIX

    def project_parallel(self, v6d: np.ndarray) -> np.ndarray:
        """
        Project 6D vector to 3D parallel space (physical space).

        Args:
            v6d: 6D vector (can be float or int)

        Returns:
            3D projected position
        """
        v6d = np.asarray(v6d, dtype=np.float64)
        return self.parallel_matrix @ v6d

    def project_perpendicular(self, v6d: np.ndarray) -> np.ndarray:
        """
        Project 6D vector to 3D perpendicular space.

        Args:
            v6d: 6D vector

        Returns:
            3D position in perpendicular space
        """
        v6d = np.asarray(v6d, dtype=np.float64)
        return self.perp_matrix @ v6d

    def is_in_window(self, v6d: np.ndarray) -> bool:
        """
        Check if 6D point projects into acceptance window.

        Points whose perpendicular projection falls within the
        window are included in the quasicrystal.

        Args:
            v6d: 6D vector

        Returns:
            True if point is in the acceptance window
        """
        perp = self.project_perpendicular(v6d)
        return np.linalg.norm(perp) <= self.window_radius

    def project(self, v6d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Full projection of 6D vector.

        Args:
            v6d: 6D vector

        Returns:
            Tuple of (parallel_3d, perp_3d, is_in_window)
        """
        par = self.project_parallel(v6d)
        perp = self.project_perpendicular(v6d)
        in_window = np.linalg.norm(perp) <= self.window_radius
        return par, perp, in_window

    def generate_vertices(
        self,
        max_lattice_coord: int = 3,
        max_vertices: int = 1000
    ) -> List[QuasicrystalVertex]:
        """
        Generate quasicrystal vertices within a lattice box.

        Uses cut-and-project method: iterate over Z⁶ lattice points
        and keep those that project into the acceptance window.

        Args:
            max_lattice_coord: Maximum coordinate in each dimension
            max_vertices: Maximum number of vertices to generate

        Returns:
            List of QuasicrystalVertex objects
        """
        vertices = []
        vertex_index = 0

        # Iterate over 6D lattice
        for n1 in range(-max_lattice_coord, max_lattice_coord + 1):
            for n2 in range(-max_lattice_coord, max_lattice_coord + 1):
                for n3 in range(-max_lattice_coord, max_lattice_coord + 1):
                    for n4 in range(-max_lattice_coord, max_lattice_coord + 1):
                        for n5 in range(-max_lattice_coord, max_lattice_coord + 1):
                            for n6 in range(-max_lattice_coord, max_lattice_coord + 1):
                                if len(vertices) >= max_vertices:
                                    return vertices

                                lattice = np.array([n1, n2, n3, n4, n5, n6])
                                par, perp, in_window = self.project(lattice)

                                if in_window:
                                    v = QuasicrystalVertex(
                                        lattice_6d=lattice,
                                        position_3d=par,
                                        perp_3d=perp,
                                        distance_from_origin=np.linalg.norm(par),
                                        index=vertex_index
                                    )
                                    vertices.append(v)
                                    vertex_index += 1

        return vertices


# =============================================================================
# PROJECTION FUNCTIONS
# =============================================================================

def project_6d_to_3d(
    v6d: np.ndarray,
    projector: Optional[IcosahedralProjector] = None
) -> np.ndarray:
    """
    Project a 6D vector to 3D using icosahedral symmetry.

    Args:
        v6d: 6D vector (state, context, etc.)
        projector: Optional custom projector

    Returns:
        3D projected position
    """
    if projector is None:
        projector = IcosahedralProjector()
    return projector.project_parallel(v6d)


def generate_quasicrystal_vertices(
    max_coord: int = 3,
    max_vertices: int = 500,
    window_radius: float = WINDOW_RADIUS
) -> List[QuasicrystalVertex]:
    """
    Generate quasicrystal vertices.

    Args:
        max_coord: Maximum lattice coordinate
        max_vertices: Maximum vertices to generate
        window_radius: Acceptance window radius

    Returns:
        List of vertices
    """
    projector = IcosahedralProjector(window_radius=window_radius)
    return projector.generate_vertices(max_coord, max_vertices)


# =============================================================================
# SYMMETRY VERIFICATION
# =============================================================================

def verify_icosahedral_symmetry(
    vertices: List[QuasicrystalVertex],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Verify that vertex set has icosahedral symmetry.

    Icosahedral symmetry includes:
    - 6 five-fold rotation axes
    - 10 three-fold rotation axes
    - 15 two-fold rotation axes

    Args:
        vertices: List of quasicrystal vertices
        tolerance: Numerical tolerance for symmetry checks

    Returns:
        Dictionary with symmetry analysis
    """
    if len(vertices) == 0:
        return {"valid": False, "reason": "No vertices"}

    positions = np.array([v.position_3d for v in vertices])
    center = np.mean(positions, axis=0)

    # Recenter
    centered = positions - center

    # Check 5-fold symmetry around z-axis
    # Rotation by 72° should map vertices to vertices
    angle = 2 * math.pi / 5
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R5 = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    five_fold_preserved = 0
    for pos in centered:
        rotated = R5 @ pos
        # Check if rotated position is close to any original position
        distances = np.linalg.norm(centered - rotated, axis=1)
        if np.min(distances) < tolerance:
            five_fold_preserved += 1

    five_fold_ratio = five_fold_preserved / len(vertices)

    # Check 3-fold symmetry
    angle = 2 * math.pi / 3
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Rotation around [1,1,1] axis
    axis = np.array([1, 1, 1]) / math.sqrt(3)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R3 = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    three_fold_preserved = 0
    for pos in centered:
        rotated = R3 @ pos
        distances = np.linalg.norm(centered - rotated, axis=1)
        if np.min(distances) < tolerance:
            three_fold_preserved += 1

    three_fold_ratio = three_fold_preserved / len(vertices)

    # Determine overall validity
    # For a true quasicrystal, we expect high preservation under symmetry operations
    is_valid = five_fold_ratio > 0.8 and three_fold_ratio > 0.8

    return {
        "valid": is_valid,
        "vertex_count": len(vertices),
        "five_fold_preserved_ratio": five_fold_ratio,
        "three_fold_preserved_ratio": three_fold_ratio,
        "center": center.tolist(),
        "tolerance": tolerance
    }


# =============================================================================
# DIFFRACTION PATTERN
# =============================================================================

def compute_diffraction_pattern(
    vertices: List[QuasicrystalVertex],
    q_max: float = 10.0,
    resolution: int = 50
) -> np.ndarray:
    """
    Compute diffraction pattern (structure factor) of quasicrystal.

    The diffraction pattern shows the long-range order of the
    quasicrystal as sharp peaks despite aperiodicity.

    S(q) = |Σ_j exp(i q · r_j)|² / N

    Args:
        vertices: List of quasicrystal vertices
        q_max: Maximum wavevector magnitude
        resolution: Grid resolution for q-space

    Returns:
        2D array of diffraction intensities (z=0 slice)
    """
    if len(vertices) == 0:
        return np.zeros((resolution, resolution))

    positions = np.array([v.position_3d for v in vertices])
    N = len(positions)

    # Create q-space grid (z=0 slice)
    qx = np.linspace(-q_max, q_max, resolution)
    qy = np.linspace(-q_max, q_max, resolution)
    QX, QY = np.meshgrid(qx, qy)

    pattern = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            q = np.array([QX[i, j], QY[i, j], 0])

            # Structure factor
            phases = np.sum(positions * q, axis=1)  # q · r_j for all j
            F = np.sum(np.exp(1j * phases))

            # Intensity
            pattern[i, j] = np.abs(F)**2 / N

    return pattern


def diffraction_fingerprint(
    vertices: List[QuasicrystalVertex],
    q_max: float = 5.0,
    resolution: int = 20
) -> bytes:
    """
    Compute a hash fingerprint of the diffraction pattern.

    This serves as a unique identifier for the quasicrystal state.

    Args:
        vertices: Quasicrystal vertices
        q_max: Maximum wavevector
        resolution: Pattern resolution

    Returns:
        SHA3-256 hash of the diffraction pattern
    """
    pattern = compute_diffraction_pattern(vertices, q_max, resolution)

    # Quantize to 8-bit for consistent hashing
    normalized = pattern / (np.max(pattern) + 1e-10)
    quantized = (normalized * 255).astype(np.uint8)

    return hashlib.sha3_256(quantized.tobytes()).digest()


# =============================================================================
# STATE VERIFICATION
# =============================================================================

def verify_state_in_quasicrystal(
    state_6d: np.ndarray,
    vertices: List[QuasicrystalVertex],
    tolerance: float = 0.1
) -> Tuple[bool, Optional[QuasicrystalVertex], float]:
    """
    Verify that a 6D state maps to a valid quasicrystal position.

    Args:
        state_6d: 6D state vector
        vertices: Known quasicrystal vertices
        tolerance: Maximum distance to count as "on vertex"

    Returns:
        Tuple of (is_valid, nearest_vertex, distance)
    """
    projector = IcosahedralProjector()
    pos_3d = projector.project_parallel(state_6d)

    min_dist = float('inf')
    nearest = None

    for v in vertices:
        dist = np.linalg.norm(pos_3d - v.position_3d)
        if dist < min_dist:
            min_dist = dist
            nearest = v

    is_valid = min_dist <= tolerance

    return is_valid, nearest, min_dist


def quasicrystal_coherence(
    trajectory_6d: List[np.ndarray],
    vertices: List[QuasicrystalVertex]
) -> float:
    """
    Compute coherence of a trajectory with quasicrystal structure.

    Higher coherence means the trajectory stays closer to the
    quasicrystal vertices (more "crystalline" behavior).

    Args:
        trajectory_6d: List of 6D state vectors
        vertices: Quasicrystal vertices

    Returns:
        Coherence score in [0, 1]
    """
    if len(trajectory_6d) == 0 or len(vertices) == 0:
        return 0.0

    total_distance = 0.0
    projector = IcosahedralProjector()

    for state in trajectory_6d:
        pos_3d = projector.project_parallel(state)

        # Find minimum distance to any vertex
        min_dist = float('inf')
        for v in vertices:
            dist = np.linalg.norm(pos_3d - v.position_3d)
            if dist < min_dist:
                min_dist = dist

        total_distance += min_dist

    avg_distance = total_distance / len(trajectory_6d)

    # Convert to coherence (inverse relationship)
    # Using exponential decay: coherence = exp(-avg_distance)
    coherence = math.exp(-avg_distance)

    return coherence
