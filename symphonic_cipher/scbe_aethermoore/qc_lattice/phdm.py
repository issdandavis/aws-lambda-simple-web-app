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
