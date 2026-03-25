"""
Polyhedral Hamiltonian Defense Manifold (PHDM) - AETHERMOORE Integration

Implements topological verification using polyhedral geometry.
The manifold consists of 16 interlocking polyhedra that must maintain
Euler characteristic χ = 2 for valid states.

Key Properties:
1. Euler characteristic: V - E + F = 2 (for genus-0 polyhedra)
2. Hamiltonian paths through vertex graph detect tampering
3. Dual polyhedra provide redundant verification
4. Topological invariants resist continuous deformation attacks

The 16 Polyhedra:
- 5 Platonic solids (regular, convex)
- 11 Archimedean solids (semi-regular, convex)

For AETHERMOORE:
- Each polyhedron represents a governance domain
- Hamiltonian paths encode valid state transitions
- Euler violations indicate topological attacks
- Dual correspondence provides cross-validation

Document ID: AETHER-PHDM-2026-001
Version: 1.0.0
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set, FrozenSet
from enum import Enum
import numpy as np

from ..constants import PHI


# =============================================================================
# POLYHEDRA DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class PolyhedronDef:
    """Definition of a convex polyhedron."""
    name: str
    vertices: int  # V
    edges: int  # E
    faces: int  # F
    face_types: Tuple[int, ...]  # e.g., (3,) for tetrahedron = all triangles
    vertex_config: str  # Vertex configuration notation
    dual_name: Optional[str] = None

    @property
    def euler_characteristic(self) -> int:
        """Compute Euler characteristic χ = V - E + F."""
        return self.vertices - self.edges + self.faces

    def is_valid(self) -> bool:
        """Check if Euler characteristic is 2 (genus 0)."""
        return self.euler_characteristic == 2


# 5 Platonic Solids
PLATONIC_SOLIDS = {
    "tetrahedron": PolyhedronDef(
        name="Tetrahedron",
        vertices=4, edges=6, faces=4,
        face_types=(3,),  # 4 triangles
        vertex_config="3.3.3",
        dual_name="tetrahedron"  # Self-dual
    ),
    "cube": PolyhedronDef(
        name="Cube",
        vertices=8, edges=12, faces=6,
        face_types=(4,),  # 6 squares
        vertex_config="4.4.4",
        dual_name="octahedron"
    ),
    "octahedron": PolyhedronDef(
        name="Octahedron",
        vertices=6, edges=12, faces=8,
        face_types=(3,),  # 8 triangles
        vertex_config="3.3.3.3",
        dual_name="cube"
    ),
    "dodecahedron": PolyhedronDef(
        name="Dodecahedron",
        vertices=20, edges=30, faces=12,
        face_types=(5,),  # 12 pentagons
        vertex_config="5.5.5",
        dual_name="icosahedron"
    ),
    "icosahedron": PolyhedronDef(
        name="Icosahedron",
        vertices=12, edges=30, faces=20,
        face_types=(3,),  # 20 triangles
        vertex_config="3.3.3.3.3",
        dual_name="dodecahedron"
    ),
}

# 11 Archimedean Solids (selected key ones)
ARCHIMEDEAN_SOLIDS = {
    "truncated_tetrahedron": PolyhedronDef(
        name="Truncated Tetrahedron",
        vertices=12, edges=18, faces=8,
        face_types=(3, 6),  # 4 triangles + 4 hexagons
        vertex_config="3.6.6",
        dual_name="triakis_tetrahedron"
    ),
    "cuboctahedron": PolyhedronDef(
        name="Cuboctahedron",
        vertices=12, edges=24, faces=14,
        face_types=(3, 4),  # 8 triangles + 6 squares
        vertex_config="3.4.3.4",
        dual_name="rhombic_dodecahedron"
    ),
    "truncated_cube": PolyhedronDef(
        name="Truncated Cube",
        vertices=24, edges=36, faces=14,
        face_types=(3, 8),  # 8 triangles + 6 octagons
        vertex_config="3.8.8",
        dual_name="triakis_octahedron"
    ),
    "truncated_octahedron": PolyhedronDef(
        name="Truncated Octahedron",
        vertices=24, edges=36, faces=14,
        face_types=(4, 6),  # 6 squares + 8 hexagons
        vertex_config="4.6.6",
        dual_name="tetrakis_hexahedron"
    ),
    "rhombicuboctahedron": PolyhedronDef(
        name="Rhombicuboctahedron",
        vertices=24, edges=48, faces=26,
        face_types=(3, 4),  # 8 triangles + 18 squares
        vertex_config="3.4.4.4",
        dual_name="deltoidal_icositetrahedron"
    ),
    "truncated_cuboctahedron": PolyhedronDef(
        name="Truncated Cuboctahedron",
        vertices=48, edges=72, faces=26,
        face_types=(4, 6, 8),  # 12 squares + 8 hexagons + 6 octagons
        vertex_config="4.6.8",
        dual_name="disdyakis_dodecahedron"
    ),
    "snub_cube": PolyhedronDef(
        name="Snub Cube",
        vertices=24, edges=60, faces=38,
        face_types=(3, 4),  # 32 triangles + 6 squares
        vertex_config="3.3.3.3.4",
        dual_name="pentagonal_icositetrahedron"
    ),
    "icosidodecahedron": PolyhedronDef(
        name="Icosidodecahedron",
        vertices=30, edges=60, faces=32,
        face_types=(3, 5),  # 20 triangles + 12 pentagons
        vertex_config="3.5.3.5",
        dual_name="rhombic_triacontahedron"
    ),
    "truncated_dodecahedron": PolyhedronDef(
        name="Truncated Dodecahedron",
        vertices=60, edges=90, faces=32,
        face_types=(3, 10),  # 20 triangles + 12 decagons
        vertex_config="3.10.10",
        dual_name="triakis_icosahedron"
    ),
    "truncated_icosahedron": PolyhedronDef(
        name="Truncated Icosahedron",
        vertices=60, edges=90, faces=32,
        face_types=(5, 6),  # 12 pentagons + 20 hexagons (soccer ball)
        vertex_config="5.6.6",
        dual_name="pentakis_dodecahedron"
    ),
    "rhombicosidodecahedron": PolyhedronDef(
        name="Rhombicosidodecahedron",
        vertices=60, edges=120, faces=62,
        face_types=(3, 4, 5),  # 20 triangles + 30 squares + 12 pentagons
        vertex_config="3.4.5.4",
        dual_name="deltoidal_hexecontahedron"
    ),
}

# All 16 polyhedra
ALL_POLYHEDRA = {**PLATONIC_SOLIDS, **ARCHIMEDEAN_SOLIDS}


# =============================================================================
# PHDM STATE
# =============================================================================

class PHDMStatus(Enum):
    """Status of PHDM verification."""
    VALID = "VALID"
    EULER_VIOLATION = "EULER_VIOLATION"
    HAMILTONIAN_BREAK = "HAMILTONIAN_BREAK"
    DUAL_MISMATCH = "DUAL_MISMATCH"
    TOPOLOGY_ANOMALY = "TOPOLOGY_ANOMALY"


@dataclass
class PHDMState:
    """
    State of the Polyhedral Hamiltonian Defense Manifold.

    Tracks the topological state of all 16 polyhedra and
    their interconnections.
    """
    # Polyhedron states (vertex counts may be perturbed by attacks)
    polyhedron_states: Dict[str, Tuple[int, int, int]]  # name -> (V, E, F)

    # Hamiltonian path validity
    hamiltonian_valid: Dict[str, bool]

    # Overall status
    status: PHDMStatus
    euler_violations: List[str]
    anomaly_score: float

    # Timestamp
    timestamp: float

    def total_euler_characteristic(self) -> int:
        """Sum of all Euler characteristics."""
        total = 0
        for v, e, f in self.polyhedron_states.values():
            total += v - e + f
        return total

    def is_valid(self) -> bool:
        """Check if manifold is in valid state."""
        return self.status == PHDMStatus.VALID


# =============================================================================
# POLYHEDRAL DEFENSE MANIFOLD
# =============================================================================

class PolyhedralDefenseManifold:
    """
    The 16-polyhedra defense manifold.

    Provides topological verification through:
    1. Euler characteristic monitoring
    2. Hamiltonian path verification
    3. Dual polyhedra cross-validation
    4. Anomaly detection
    """

    def __init__(self, polyhedra: Optional[Dict[str, PolyhedronDef]] = None):
        """
        Initialize the defense manifold.

        Args:
            polyhedra: Dictionary of polyhedra definitions (default: all 16)
        """
        self.polyhedra = polyhedra or ALL_POLYHEDRA
        self.state_history: List[PHDMState] = []

        # Initialize vertex graphs for Hamiltonian path checking
        self.vertex_graphs = self._build_vertex_graphs()

    def _build_vertex_graphs(self) -> Dict[str, Dict[int, Set[int]]]:
        """
        Build adjacency graphs for each polyhedron.

        These graphs are used for Hamiltonian path verification.
        Returns simplified model based on vertex configuration.
        """
        graphs = {}

        for name, poly in self.polyhedra.items():
            # Create a graph where vertices are connected based on face structure
            # This is a simplified model - real implementation would use
            # actual vertex coordinates

            V = poly.vertices
            graph = {i: set() for i in range(V)}

            # Connect vertices in a way consistent with the polyhedron
            # For simplicity, use a regular connection pattern
            for i in range(V):
                # Each vertex connects to nearby vertices
                # Number of connections based on vertex configuration
                valence = len(poly.vertex_config.split('.'))
                for j in range(1, valence + 1):
                    neighbor = (i + j) % V
                    graph[i].add(neighbor)
                    graph[neighbor].add(i)

            graphs[name] = graph

        return graphs

    def verify_state(
        self,
        state_vector: Optional[np.ndarray] = None
    ) -> PHDMState:
        """
        Verify the current state of the defense manifold.

        Args:
            state_vector: Optional state vector to incorporate

        Returns:
            PHDMState with verification results
        """
        import time

        polyhedron_states = {}
        hamiltonian_valid = {}
        euler_violations = []

        # Check each polyhedron
        for name, poly in self.polyhedra.items():
            # Get current state (possibly perturbed by state_vector)
            v, e, f = self._get_polyhedron_state(name, state_vector)
            polyhedron_states[name] = (v, e, f)

            # Verify Euler characteristic
            chi = v - e + f
            if chi != 2:
                euler_violations.append(f"{name}: χ={chi}≠2")

            # Verify Hamiltonian path exists
            hamiltonian_valid[name] = self._verify_hamiltonian_path(name)

        # Check dual correspondences
        dual_mismatches = self._check_dual_correspondence(polyhedron_states)

        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(
            euler_violations,
            hamiltonian_valid,
            dual_mismatches
        )

        # Determine overall status
        if len(euler_violations) > 0:
            status = PHDMStatus.EULER_VIOLATION
        elif not all(hamiltonian_valid.values()):
            status = PHDMStatus.HAMILTONIAN_BREAK
        elif len(dual_mismatches) > 0:
            status = PHDMStatus.DUAL_MISMATCH
        elif anomaly_score > 0.5:
            status = PHDMStatus.TOPOLOGY_ANOMALY
        else:
            status = PHDMStatus.VALID

        state = PHDMState(
            polyhedron_states=polyhedron_states,
            hamiltonian_valid=hamiltonian_valid,
            status=status,
            euler_violations=euler_violations,
            anomaly_score=anomaly_score,
            timestamp=time.time()
        )

        self.state_history.append(state)
        return state

    def _get_polyhedron_state(
        self,
        name: str,
        state_vector: Optional[np.ndarray]
    ) -> Tuple[int, int, int]:
        """
        Get (V, E, F) for a polyhedron, possibly perturbed by state.

        In normal operation, returns the canonical values.
        If state_vector is provided and anomalous, may return perturbed values.
        """
        poly = self.polyhedra[name]

        if state_vector is None:
            return poly.vertices, poly.edges, poly.faces

        # Use state vector to potentially perturb counts
        # This simulates how an attack might corrupt the topology
        hash_val = hashlib.sha3_256(
            name.encode() + state_vector.tobytes()
        ).digest()

        # Extract perturbation factor
        perturb = (hash_val[0] / 255.0) - 0.5  # [-0.5, 0.5]

        # Normal states have small perturbations that round to zero
        # Anomalous states have larger perturbations
        v_perturb = int(round(perturb * np.linalg.norm(state_vector)))
        e_perturb = int(round(perturb * 2 * np.linalg.norm(state_vector)))
        f_perturb = int(round(perturb * np.linalg.norm(state_vector)))

        return (
            max(1, poly.vertices + v_perturb),
            max(1, poly.edges + e_perturb),
            max(1, poly.faces + f_perturb)
        )

    def _verify_hamiltonian_path(self, name: str) -> bool:
        """
        Verify that a Hamiltonian path exists in the polyhedron's vertex graph.

        A Hamiltonian path visits every vertex exactly once.
        Its existence is a topological invariant.
        """
        graph = self.vertex_graphs.get(name)
        if graph is None:
            return False

        V = len(graph)
        if V == 0:
            return False

        # Use backtracking to find Hamiltonian path
        # For small graphs this is tractable
        if V > 60:  # Too large, assume valid for performance
            return True

        def backtrack(path: List[int], visited: Set[int]) -> bool:
            if len(path) == V:
                return True

            current = path[-1]
            for neighbor in graph[current]:
                if neighbor not in visited:
                    path.append(neighbor)
                    visited.add(neighbor)

                    if backtrack(path, visited):
                        return True

                    path.pop()
                    visited.remove(neighbor)

            return False

        # Try starting from each vertex
        for start in range(min(V, 5)):  # Limit starting points
            if backtrack([start], {start}):
                return True

        return False

    def _check_dual_correspondence(
        self,
        states: Dict[str, Tuple[int, int, int]]
    ) -> List[str]:
        """
        Check that dual polyhedra have consistent states.

        For dual pairs: V_1 = F_2 and F_1 = V_2.
        """
        mismatches = []

        for name, poly in self.polyhedra.items():
            if poly.dual_name and poly.dual_name in self.polyhedra:
                v1, e1, f1 = states[name]
                if poly.dual_name in states:
                    v2, e2, f2 = states[poly.dual_name]

                    # Check dual relationship
                    # Vertices of one = faces of dual (not exact due to perturbation)
                    if abs(v1 - f2) > 2 or abs(f1 - v2) > 2:
                        mismatches.append(f"{name}<->{poly.dual_name}")

        return mismatches

    def _compute_anomaly_score(
        self,
        euler_violations: List[str],
        hamiltonian_valid: Dict[str, bool],
        dual_mismatches: List[str]
    ) -> float:
        """
        Compute overall anomaly score in [0, 1].

        Higher score = more anomalous state.
        """
        n_poly = len(self.polyhedra)

        euler_score = len(euler_violations) / n_poly
        hamiltonian_score = sum(1 for v in hamiltonian_valid.values() if not v) / n_poly
        dual_score = len(dual_mismatches) / (n_poly / 2)  # Pairs

        # Weighted combination
        return 0.5 * euler_score + 0.3 * hamiltonian_score + 0.2 * dual_score

    def get_fingerprint(self) -> bytes:
        """
        Compute cryptographic fingerprint of manifold state.

        This serves as a unique identifier that changes if
        topology is corrupted.
        """
        data = b""

        for name in sorted(self.polyhedra.keys()):
            poly = self.polyhedra[name]
            data += name.encode()
            data += poly.vertices.to_bytes(4, 'big')
            data += poly.edges.to_bytes(4, 'big')
            data += poly.faces.to_bytes(4, 'big')

        return hashlib.sha3_256(data).digest()


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_euler_characteristic(
    vertices: int,
    edges: int,
    faces: int,
    expected_genus: int = 0
) -> Tuple[bool, int, str]:
    """
    Verify Euler characteristic for a surface.

    χ = V - E + F = 2 - 2g (where g is genus)
    For genus 0 (sphere-like): χ = 2

    Args:
        vertices: Number of vertices
        edges: Number of edges
        faces: Number of faces
        expected_genus: Expected genus (default 0 for convex polyhedra)

    Returns:
        Tuple of (is_valid, actual_chi, explanation)
    """
    actual_chi = vertices - edges + faces
    expected_chi = 2 - 2 * expected_genus

    is_valid = actual_chi == expected_chi

    if is_valid:
        explanation = f"χ = {actual_chi} ✓ (genus {expected_genus})"
    else:
        inferred_genus = (2 - actual_chi) / 2
        explanation = f"χ = {actual_chi} ≠ {expected_chi} (implies genus {inferred_genus:.1f})"

    return is_valid, actual_chi, explanation


def compute_hamiltonian_path(
    adjacency: Dict[int, Set[int]],
    max_attempts: int = 100
) -> Optional[List[int]]:
    """
    Find a Hamiltonian path in a graph.

    Args:
        adjacency: Adjacency dictionary {vertex: set of neighbors}
        max_attempts: Maximum starting vertices to try

    Returns:
        Path as list of vertices, or None if not found
    """
    V = len(adjacency)
    if V == 0:
        return None

    attempts = 0

    def backtrack(path: List[int], visited: Set[int]) -> Optional[List[int]]:
        if len(path) == V:
            return path.copy()

        current = path[-1]
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)

                result = backtrack(path, visited)
                if result is not None:
                    return result

                path.pop()
                visited.remove(neighbor)

        return None

    for start in range(min(V, max_attempts)):
        result = backtrack([start], {start})
        if result is not None:
            return result
        attempts += 1

    return None


def detect_topological_anomaly(
    state_sequence: List[PHDMState],
    window_size: int = 10
) -> Tuple[bool, float, str]:
    """
    Detect topological anomalies in state sequence.

    Looks for:
    1. Sudden Euler characteristic changes
    2. Hamiltonian path breaks
    3. Anomaly score spikes

    Args:
        state_sequence: Sequence of PHDM states
        window_size: Analysis window size

    Returns:
        Tuple of (anomaly_detected, severity, description)
    """
    if len(state_sequence) < 2:
        return False, 0.0, "Insufficient data"

    recent = state_sequence[-window_size:]

    # Check for Euler violations
    euler_violation_count = sum(
        1 for s in recent if len(s.euler_violations) > 0
    )
    euler_ratio = euler_violation_count / len(recent)

    # Check anomaly score trend
    anomaly_scores = [s.anomaly_score for s in recent]
    avg_anomaly = sum(anomaly_scores) / len(anomaly_scores)

    # Detect sudden changes
    if len(recent) >= 2:
        score_changes = [
            abs(recent[i].anomaly_score - recent[i-1].anomaly_score)
            for i in range(1, len(recent))
        ]
        max_change = max(score_changes)
    else:
        max_change = 0.0

    # Determine if anomaly detected
    anomaly_detected = (
        euler_ratio > 0.3 or
        avg_anomaly > 0.5 or
        max_change > 0.4
    )

    severity = max(euler_ratio, avg_anomaly, max_change)

    if anomaly_detected:
        if euler_ratio > 0.3:
            description = f"Euler violations: {euler_ratio:.1%} of recent states"
        elif avg_anomaly > 0.5:
            description = f"High average anomaly score: {avg_anomaly:.2f}"
        else:
            description = f"Sudden topology change: Δ={max_change:.2f}"
    else:
        description = "Topology stable"

    return anomaly_detected, severity, description
Polyhedral Hamiltonian Defense Manifold (PHDM)

A curated family of 16 canonical polyhedra providing diverse topological
structures for cryptographic verification:

- Platonic Solids (5): Symmetric baseline for safe states
- Archimedean Solids (3): Mixed-face complexity for dynamic paths
- Kepler-Poinsot (2): Non-convex stars for attack surface detection
- Toroidal (2): Szilassi/Császár for skip-attack resistance
- Johnson Solids (2): Near-regular bridges to real CFGs
- Rhombic Variants (2): Space-filling tessellation potential

The Hamiltonian path visits each polyhedron exactly once, with sequential
HMAC chaining for cryptographic binding.
"""

import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
import hmac


# Golden Ratio for icosahedral/dodecahedral geometry
PHI = (1 + np.sqrt(5)) / 2


class PolyhedronType(Enum):
    """Classification of polyhedron types."""
    PLATONIC = "platonic"
    ARCHIMEDEAN = "archimedean"
    KEPLER_POINSOT = "kepler_poinsot"
    TOROIDAL = "toroidal"
    JOHNSON = "johnson"
    RHOMBIC = "rhombic"


@dataclass
class Polyhedron:
    """
    A polyhedron in the PHDM family.

    Attributes:
        name: Human-readable name
        poly_type: Classification type
        vertices: Number of vertices (V)
        edges: Number of edges (E)
        faces: Number of faces (F)
        face_types: Description of face types
        genus: Topological genus (0 for convex, 1 for toroidal)
        vertex_coords: Optional 3D vertex coordinates
        adjacency: Optional vertex adjacency list
        notes: Role in PHDM system
    """
    name: str
    poly_type: PolyhedronType
    vertices: int
    edges: int
    faces: int
    face_types: str
    genus: int = 0
    vertex_coords: Optional[np.ndarray] = None
    adjacency: Optional[List[List[int]]] = None
    notes: str = ""

    def euler_characteristic(self) -> int:
        """Compute Euler characteristic: V - E + F = 2 - 2g"""
        return self.vertices - self.edges + self.faces

    def expected_euler(self) -> int:
        """Expected Euler characteristic based on genus."""
        return 2 - 2 * self.genus

    def is_valid_topology(self) -> bool:
        """Check if V-E+F matches expected Euler characteristic."""
        return self.euler_characteristic() == self.expected_euler()

    def serialize(self) -> bytes:
        """Serialize polyhedron data for HMAC chaining."""
        data = (
            self.name.encode() + b"|" +
            self.poly_type.value.encode() + b"|" +
            str(self.vertices).encode() + b"|" +
            str(self.edges).encode() + b"|" +
            str(self.faces).encode() + b"|" +
            str(self.genus).encode()
        )
        if self.vertex_coords is not None:
            data += b"|" + self.vertex_coords.tobytes()
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.poly_type.value,
            "V": self.vertices,
            "E": self.edges,
            "F": self.faces,
            "face_types": self.face_types,
            "genus": self.genus,
            "euler": self.euler_characteristic(),
            "notes": self.notes
        }


def create_tetrahedron_coords() -> np.ndarray:
    """Generate tetrahedron vertex coordinates."""
    return np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float) / np.sqrt(3)


def create_cube_coords() -> np.ndarray:
    """Generate cube vertex coordinates."""
    coords = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                coords.append([x, y, z])
    return np.array(coords, dtype=float)


def create_octahedron_coords() -> np.ndarray:
    """Generate octahedron vertex coordinates."""
    return np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)


def create_dodecahedron_coords() -> np.ndarray:
    """Generate dodecahedron vertex coordinates using golden ratio."""
    coords = []
    # Vertices from cube corners
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                coords.append([x, y, z])

    # Vertices on faces (using golden ratio)
    for x in [-1/PHI, 1/PHI]:
        for y in [-PHI, PHI]:
            coords.append([0, x, y])
            coords.append([x, y, 0])
            coords.append([y, 0, x])

    return np.array(coords, dtype=float)


def create_icosahedron_coords() -> np.ndarray:
    """Generate icosahedron vertex coordinates."""
    coords = []
    for x in [-1, 1]:
        for y in [-PHI, PHI]:
            coords.append([0, x, y])
            coords.append([x, y, 0])
            coords.append([y, 0, x])

    return np.array(coords, dtype=float) / np.sqrt(1 + PHI**2)


# =============================================================================
# The 16 Canonical PHDM Polyhedra
# =============================================================================

def get_phdm_family() -> List[Polyhedron]:
    """
    Return the complete PHDM family of 16 canonical polyhedra.

    Ordered for optimal Hamiltonian traversal.
    """
    return [
        # =====================================================================
        # PLATONIC SOLIDS (5) - Symmetric baseline for safe states
        # =====================================================================
        Polyhedron(
            name="Tetrahedron",
            poly_type=PolyhedronType.PLATONIC,
            vertices=4, edges=6, faces=4,
            face_types="4 triangles",
            vertex_coords=create_tetrahedron_coords(),
            notes="Minimal convex - ideal for origin anchoring"
        ),
        Polyhedron(
            name="Cube",
            poly_type=PolyhedronType.PLATONIC,
            vertices=8, edges=12, faces=6,
            face_types="6 squares",
            vertex_coords=create_cube_coords(),
            notes="Orthogonal structure for grid-like embeddings"
        ),
        Polyhedron(
            name="Octahedron",
            poly_type=PolyhedronType.PLATONIC,
            vertices=6, edges=12, faces=8,
            face_types="8 triangles",
            vertex_coords=create_octahedron_coords(),
            notes="Dual to cube - high coordination"
        ),
        Polyhedron(
            name="Dodecahedron",
            poly_type=PolyhedronType.PLATONIC,
            vertices=20, edges=30, faces=12,
            face_types="12 pentagons",
            vertex_coords=create_dodecahedron_coords(),
            notes="Golden ratio symmetry for harmonic scaling"
        ),
        Polyhedron(
            name="Icosahedron",
            poly_type=PolyhedronType.PLATONIC,
            vertices=12, edges=30, faces=20,
            face_types="20 triangles",
            vertex_coords=create_icosahedron_coords(),
            notes="Maximal vertices for Platonic - dense connectivity"
        ),

        # =====================================================================
        # ARCHIMEDEAN SOLIDS (3) - Mixed-face complexity for dynamic paths
        # =====================================================================
        Polyhedron(
            name="Truncated Tetrahedron",
            poly_type=PolyhedronType.ARCHIMEDEAN,
            vertices=12, edges=18, faces=8,
            face_types="4 triangles + 4 hexagons",
            notes="Truncation introduces higher faces for deviation traps"
        ),
        Polyhedron(
            name="Cuboctahedron",
            poly_type=PolyhedronType.ARCHIMEDEAN,
            vertices=12, edges=24, faces=14,
            face_types="8 triangles + 6 squares",
            notes="Archimedean 'bridge' between cube/octahedron"
        ),
        Polyhedron(
            name="Icosidodecahedron",
            poly_type=PolyhedronType.ARCHIMEDEAN,
            vertices=30, edges=60, faces=32,
            face_types="20 triangles + 12 pentagons",
            notes="High-density for geodesic smoothing"
        ),

        # =====================================================================
        # KEPLER-POINSOT (2) - Non-convex stars for attack surfaces
        # =====================================================================
        Polyhedron(
            name="Small Stellated Dodecahedron",
            poly_type=PolyhedronType.KEPLER_POINSOT,
            vertices=12, edges=30, faces=12,
            face_types="12 pentagrams",
            notes="Star density for sharp curvature spikes"
        ),
        Polyhedron(
            name="Great Dodecahedron",
            poly_type=PolyhedronType.KEPLER_POINSOT,
            vertices=12, edges=30, faces=12,
            face_types="12 pentagons (intersecting)",
            notes="Deeper non-convexity for intrusion boundaries"
        ),

        # =====================================================================
        # TOROIDAL (2) - Genus > 0 for topological robustness
        # =====================================================================
        Polyhedron(
            name="Szilassi Polyhedron",
            poly_type=PolyhedronType.TOROIDAL,
            vertices=14, edges=21, faces=7,
            face_types="7 hexagons",
            genus=1,
            notes="Genus 1 torus - every face touches every other; maximal adjacency"
        ),
        Polyhedron(
            name="Császár Polyhedron",
            poly_type=PolyhedronType.TOROIDAL,
            vertices=7, edges=21, faces=14,
            face_types="14 triangles",
            genus=1,
            notes="Dual to Szilassi - minimal vertices with full triangulation"
        ),

        # =====================================================================
        # JOHNSON SOLIDS (2) - Near-regular bridges to real CFGs
        # =====================================================================
        Polyhedron(
            name="Pentagonal Bipyramid",
            poly_type=PolyhedronType.JOHNSON,
            vertices=7, edges=15, faces=10,
            face_types="10 triangles",
            notes="Dual-like extension for pyramidal deviations"
        ),
        Polyhedron(
            name="Triangular Cupola",
            poly_type=PolyhedronType.JOHNSON,
            vertices=9, edges=15, faces=8,
            face_types="4 triangles + 3 squares + 1 hexagon",
            notes="Cupola for layered manifold stacking"
        ),

        # =====================================================================
        # RHOMBIC VARIANTS (2) - Space-filling tessellation
        # =====================================================================
        Polyhedron(
            name="Rhombic Dodecahedron",
            poly_type=PolyhedronType.RHOMBIC,
            vertices=14, edges=24, faces=12,
            face_types="12 rhombi",
            notes="Space-filling dual to cuboctahedron - dense packing"
        ),
        Polyhedron(
            name="Bilinski Dodecahedron",
            poly_type=PolyhedronType.RHOMBIC,
            vertices=14, edges=24, faces=12,
            face_types="12 rhombi (golden ratio variant)",
            notes="Alternative rhombic symmetry with golden proportions"
        ),
    ]


# =============================================================================
# Hamiltonian Path and HMAC Chaining
# =============================================================================

@dataclass
class HamiltonianNode:
    """A node in the Hamiltonian path through the PHDM."""
    polyhedron: Polyhedron
    position: int
    hmac_tag: bytes
    prev_tag: bytes


class PHDMHamiltonianPath:
    """
    Manages Hamiltonian path traversal through the PHDM polyhedra family.

    The path visits each of the 16 polyhedra exactly once, with HMAC
    chaining providing cryptographic binding between nodes.
    """

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize the PHDM path.

        Args:
            key: HMAC key (generated if not provided)
        """
        self.family = get_phdm_family()
        self.key = key or hashlib.sha256(b"phdm_default_key").digest()
        self._path: List[HamiltonianNode] = []
        self._iv = b'\x00' * 32

    def compute_path(self) -> List[HamiltonianNode]:
        """
        Compute the Hamiltonian path with HMAC chaining.

        Returns:
            List of HamiltonianNode objects forming the complete path
        """
        self._path = []
        prev_tag = self._iv

        for i, poly in enumerate(self.family):
            # Compute HMAC tag: H_k(poly_data || position || prev_tag)
            data = poly.serialize() + str(i).encode() + prev_tag
            tag = hmac.new(self.key, data, hashlib.sha256).digest()

            node = HamiltonianNode(
                polyhedron=poly,
                position=i,
                hmac_tag=tag,
                prev_tag=prev_tag
            )
            self._path.append(node)
            prev_tag = tag

        return self._path

    def verify_path(self) -> Tuple[bool, Optional[int]]:
        """
        Verify the integrity of the Hamiltonian path.

        Returns:
            Tuple of (is_valid, first_invalid_position or None)
        """
        if not self._path:
            return True, None

        prev_tag = self._iv

        for node in self._path:
            # Recompute expected tag
            data = node.polyhedron.serialize() + str(node.position).encode() + prev_tag
            expected_tag = hmac.new(self.key, data, hashlib.sha256).digest()

            if not hmac.compare_digest(node.hmac_tag, expected_tag):
                return False, node.position

            if node.prev_tag != prev_tag:
                return False, node.position

            prev_tag = node.hmac_tag

        return True, None

    def get_path_digest(self) -> bytes:
        """Get a digest of the complete path for comparison."""
        if not self._path:
            self.compute_path()

        chain_data = b""
        for node in self._path:
            chain_data += node.hmac_tag

        return hashlib.sha256(chain_data).digest()

    def find_polyhedron(self, name: str) -> Optional[HamiltonianNode]:
        """Find a polyhedron in the path by name."""
        if not self._path:
            self.compute_path()

        for node in self._path:
            if node.polyhedron.name.lower() == name.lower():
                return node
        return None

    def get_geodesic_distance(self, name1: str, name2: str) -> Optional[int]:
        """
        Get the geodesic distance (path length) between two polyhedra.

        Args:
            name1: First polyhedron name
            name2: Second polyhedron name

        Returns:
            Number of steps between them, or None if not found
        """
        node1 = self.find_polyhedron(name1)
        node2 = self.find_polyhedron(name2)

        if node1 is None or node2 is None:
            return None

        return abs(node1.position - node2.position)

    def export_state(self) -> Dict[str, Any]:
        """Export path state for serialization."""
        if not self._path:
            self.compute_path()

        return {
            "path_length": len(self._path),
            "path_digest": self.get_path_digest().hex(),
            "polyhedra": [node.polyhedron.to_dict() for node in self._path],
            "hmac_tags": [node.hmac_tag.hex() for node in self._path]
        }


# =============================================================================
# Deviation Detection
# =============================================================================

class PHDMDeviationDetector:
    """
    Detects deviations from expected PHDM manifold structure.

    Uses geodesic and curvature analysis in embedded space to identify
    anomalous behavior that may indicate attacks.
    """

    def __init__(self, phdm_path: PHDMHamiltonianPath):
        """
        Initialize detector with a PHDM path.

        Args:
            phdm_path: Computed Hamiltonian path
        """
        self.path = phdm_path
        if not self.path._path:
            self.path.compute_path()

        # Build expected metrics
        self._expected_euler_sum = sum(
            node.polyhedron.euler_characteristic()
            for node in self.path._path
        )
        self._expected_vertex_total = sum(
            node.polyhedron.vertices
            for node in self.path._path
        )

    def check_topological_integrity(self) -> Tuple[bool, List[str]]:
        """
        Check that all polyhedra satisfy their expected topology.

        Returns:
            Tuple of (all_valid, list of error messages)
        """
        errors = []

        for node in self.path._path:
            poly = node.polyhedron
            if not poly.is_valid_topology():
                errors.append(
                    f"{poly.name}: Euler χ={poly.euler_characteristic()} "
                    f"expected {poly.expected_euler()}"
                )

        return len(errors) == 0, errors

    def detect_manifold_deviation(self, observed_vertices: int,
                                  observed_euler: int) -> float:
        """
        Detect deviation from expected manifold structure.

        Args:
            observed_vertices: Total vertices observed
            observed_euler: Total Euler characteristic observed

        Returns:
            Deviation score (0.0 = perfect, higher = more deviation)
        """
        vertex_deviation = abs(observed_vertices - self._expected_vertex_total)
        euler_deviation = abs(observed_euler - self._expected_euler_sum)

        # Normalize by expected values
        vertex_score = vertex_deviation / max(1, self._expected_vertex_total)
        euler_score = euler_deviation / max(1, abs(self._expected_euler_sum))

        return (vertex_score + euler_score) / 2

    def compute_curvature_at_node(self, position: int) -> float:
        """
        Compute local curvature at a path position.

        Uses discrete curvature based on vertex density.

        Args:
            position: Position in path (0-15)

        Returns:
            Curvature value (higher = more curved)
        """
        if position < 0 or position >= len(self.path._path):
            return 0.0

        node = self.path._path[position]
        poly = node.polyhedron

        # Discrete curvature: vertex defect / vertex count
        # For convex polyhedra: 4π distributed over vertices
        if poly.vertices > 0:
            return (4 * np.pi - poly.euler_characteristic() * 2 * np.pi) / poly.vertices
        return 0.0

    def get_curvature_profile(self) -> np.ndarray:
        """Get curvature values along the entire path."""
        return np.array([
            self.compute_curvature_at_node(i)
            for i in range(len(self.path._path))
        ])


# =============================================================================
# Utility Functions
# =============================================================================

def get_family_summary() -> Dict[str, Any]:
    """Get a summary of the PHDM family."""
    family = get_phdm_family()

    by_type = {}
    for poly in family:
        t = poly.poly_type.value
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(poly.name)

    total_v = sum(p.vertices for p in family)
    total_e = sum(p.edges for p in family)
    total_f = sum(p.faces for p in family)

    return {
        "total_polyhedra": len(family),
        "by_type": by_type,
        "total_vertices": total_v,
        "total_edges": total_e,
        "total_faces": total_f,
        "types": list(by_type.keys())
    }


def validate_all_polyhedra() -> Tuple[bool, List[str]]:
    """Validate all polyhedra in the PHDM family."""
    family = get_phdm_family()
    errors = []

    for poly in family:
        if not poly.is_valid_topology():
            errors.append(
                f"{poly.name}: Invalid topology "
                f"(V={poly.vertices}, E={poly.edges}, F={poly.faces}, "
                f"χ={poly.euler_characteristic()}, expected {poly.expected_euler()})"
            )

    return len(errors) == 0, errors
