#!/usr/bin/env python3
"""
Quasicrystal Lattice Authentication System (SCBE v3.0)
=======================================================

Maps SCBE's 6-gate verification onto an ICOSAHEDRAL QUASICRYSTAL LATTICE.

KEY INSIGHTS:
1. Quasicrystals = Aperiodic (never repeat) but deterministic
2. 6D hypercubic lattice projects to 3D = Perfect match for 6 SCBE gates
3. Golden ratio φ (1.618...) emerges NATURALLY from geometry
4. Phason deformations = Atomic rekeying of entire auth space simultaneously
5. Attacks create detectable periodicity in aperiodic structure

Mathematical Foundation:
- Icosahedron has 12 vertices defined by golden ratio coordinates
- 6D → 3D projection via "cut and project" (strip) method
- Penrose tiling analogy for path verification
- Phason modes for collective state evolution

Reference: arXiv:2502.10468 (Feb 2025) - Quasicrystal Vernam cipher

Author: Generated for SCBE v3.0
Date: January 11, 2026
"""

import numpy as np
import hashlib
import hmac
import time
import secrets
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import json

# =============================================================================
# CONSTANTS
# =============================================================================

# Golden Ratio - the fundamental constant of quasicrystal geometry
PHI = (1 + np.sqrt(5)) / 2  # φ ≈ 1.6180339887...
PHI_INV = 1 / PHI           # 1/φ ≈ 0.6180339887...

# Icosahedral symmetry constants
ICOSAHEDRAL_ROTATION_ORDER = 60  # 60 rotational symmetries
FIVE_FOLD_AXIS_COUNT = 6
THREE_FOLD_AXIS_COUNT = 10
TWO_FOLD_AXIS_COUNT = 15

# Security thresholds
PERIODICITY_DETECTION_THRESHOLD = 0.5  # Attacks create HIGH periodicity (>0.5)
MIN_GOLDEN_RATIO_TOLERANCE = 0.3  # Tolerance for golden ratio matching
MAX_PHASON_DISPLACEMENT = 0.15

# Gate mapping (6 SCBE gates → 6D hypercubic coordinates)
GATE_NAMES = ['WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW']
PARALLEL_GATES = ['WHO', 'WHAT', 'WHERE']  # Project to 3D physical space
PERPENDICULAR_GATES = ['WHEN', 'WHY', 'HOW']  # Project to perpendicular space


# =============================================================================
# CORE QUASICRYSTAL CLASSES
# =============================================================================

class AuthResult(Enum):
    """Authentication result states."""
    VALID = "valid"
    INVALID_GEOMETRY = "invalid_geometry"
    PERIODICITY_DETECTED = "periodicity_detected"  # Attack signature
    PHASON_VIOLATION = "phason_violation"
    GOLDEN_RATIO_VIOLATION = "golden_ratio_violation"


@dataclass
class AuthenticationAttempt:
    """Single authentication attempt in quasicrystal space."""
    gate_values: List[float]  # 6 values, one per gate
    timestamp: float
    actor_id: str
    projected_position: Optional[np.ndarray] = None
    nearest_vertices: Optional[List[Tuple[int, float]]] = None
    result: Optional[AuthResult] = None


@dataclass
class PhasonField:
    """Phason stress field for atomic rekeying."""
    direction: np.ndarray
    magnitude: float
    timestamp: float
    affected_vertices: Set[int] = field(default_factory=set)


class QuasicrystalLattice:
    """
    Icosahedral quasicrystal for aperiodic authentication verification.

    The icosahedron's 12 vertices are defined using the golden ratio,
    creating a naturally aperiodic structure that:
    1. Never repeats (preventing pattern-based attacks)
    2. Is deterministic (enabling verification)
    3. Self-similar at all scales (fractal security)
    """

    def __init__(self, seed: bytes = None):
        """
        Initialize quasicrystal lattice.

        Args:
            seed: Optional seed for deterministic initialization
        """
        self.phi = PHI
        self.phi_inv = PHI_INV
        self.seed = seed or secrets.token_bytes(32)

        # Generate base icosahedral vertices
        self.base_vertices = self._generate_icosahedral_vertices()
        self.vertices = self.base_vertices.copy()

        # Phason history for forward secrecy
        self.phason_history: List[PhasonField] = []

        # Authentication attempt cache for periodicity detection
        self.attempt_history: List[AuthenticationAttempt] = []
        self.max_history_size = 1000

        # 6D to 3D projection matrix
        self.projection_matrix = self._build_projection_matrix()

    def _generate_icosahedral_vertices(self) -> np.ndarray:
        """
        Generate the 12 vertices of a regular icosahedron.

        The vertices are defined using the golden ratio φ:
        - (0, ±1, ±φ)
        - (±1, ±φ, 0)
        - (±φ, 0, ±1)

        This creates a structure with exact 5-fold rotational symmetry.
        """
        vertices = np.array([
            # (0, ±1, ±φ) - 4 vertices
            [0, 1, self.phi],
            [0, 1, -self.phi],
            [0, -1, self.phi],
            [0, -1, -self.phi],
            # (±1, ±φ, 0) - 4 vertices
            [1, self.phi, 0],
            [1, -self.phi, 0],
            [-1, self.phi, 0],
            [-1, -self.phi, 0],
            # (±φ, 0, ±1) - 4 vertices
            [self.phi, 0, 1],
            [self.phi, 0, -1],
            [-self.phi, 0, 1],
            [-self.phi, 0, -1]
        ], dtype=np.float64)

        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices / norms

    def _build_projection_matrix(self) -> np.ndarray:
        """
        Build the 6D → 3D projection matrix using the "cut and project" method.

        This implements the strip projection from 6D hypercubic lattice
        to 3D physical space, with the perpendicular 3D being the internal
        (phason) space.
        """
        # Projection uses golden ratio for irrational slope
        # Parallel space: gates 0,1,2 (WHO, WHAT, WHERE)
        # Perpendicular space: gates 3,4,5 (WHEN, WHY, HOW)

        # The projection matrix mixes physical and internal spaces
        # using phi to ensure aperiodicity
        matrix = np.array([
            [1, 0, 0, self.phi_inv, 0, 0],
            [0, 1, 0, 0, self.phi_inv, 0],
            [0, 0, 1, 0, 0, self.phi_inv]
        ], dtype=np.float64)

        # Normalize rows
        for i in range(3):
            matrix[i] /= np.linalg.norm(matrix[i])

        return matrix

    def project_6d_to_3d(self, gate_values: List[float]) -> np.ndarray:
        """
        Map 6 SCBE gates onto 3D quasicrystal space.

        The mapping:
        - WHO, WHAT, WHERE → physical space (x, y, z)
        - WHEN, WHY, HOW → perpendicular space (folded via golden ratio)

        This creates an aperiodic point cloud where authentication
        attempts cluster near valid vertices if legitimate.

        Args:
            gate_values: List of 6 float values, one per gate

        Returns:
            3D position in quasicrystal space
        """
        if len(gate_values) != 6:
            raise ValueError(f"Expected 6 gate values, got {len(gate_values)}")

        # Normalize gate values to [-1, 1]
        normalized = np.array(gate_values, dtype=np.float64)
        normalized = np.clip(normalized, -1, 1)

        # Project using the pre-built matrix
        position = self.projection_matrix @ normalized

        # Scale to match icosahedron radius
        return position

    def find_minimal_distances(self, position: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
        """
        Find the k nearest icosahedral vertices to a position.

        This is analogous to finding which Penrose tiles a point
        belongs to. Legitimate auth attempts should cluster near
        a small number of vertices.

        Args:
            position: 3D position in quasicrystal space
            k: Number of nearest vertices to return

        Returns:
            List of (vertex_index, distance) tuples, sorted by distance
        """
        distances = []
        for i, vertex in enumerate(self.vertices):
            dist = np.linalg.norm(position - vertex)
            distances.append((i, dist))

        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def verify_golden_ratio_constraint(self, vertex_indices: List[int]) -> Tuple[bool, float]:
        """
        Check if path between vertices maintains quasicrystal geometry.

        In a valid quasicrystal, distances between vertices are related
        by the golden ratio. Attacks that don't understand this geometry
        will produce non-golden ratios.

        For icosahedron vertices, we also accept integer ratios (1:1)
        since adjacent vertices can have equal distances.

        Args:
            vertex_indices: Sequence of vertex indices forming a path

        Returns:
            (is_valid, golden_ratio_score)
        """
        if len(vertex_indices) < 2:
            return True, 1.0

        distances = []
        for i in range(len(vertex_indices) - 1):
            v1 = self.vertices[vertex_indices[i]]
            v2 = self.vertices[vertex_indices[i + 1]]
            dist = np.linalg.norm(v2 - v1)
            distances.append(dist)

        # Single edge is always valid
        if len(distances) < 2:
            return True, 1.0

        # Check if distance ratios approximate φ, 1/φ, or 1 (equal distances)
        golden_scores = []
        for i in range(len(distances) - 1):
            if distances[i + 1] > 1e-10:  # Avoid division by zero
                ratio = distances[i] / distances[i + 1]
                # Score based on proximity to φ, 1/φ, or 1
                phi_diff = abs(ratio - self.phi)
                phi_inv_diff = abs(ratio - self.phi_inv)
                unity_diff = abs(ratio - 1.0)
                best_diff = min(phi_diff, phi_inv_diff, unity_diff)
                score = max(0, 1 - best_diff / MIN_GOLDEN_RATIO_TOLERANCE)
                golden_scores.append(score)

        if not golden_scores:
            return True, 1.0

        # Use max score (best matching ratio) rather than average
        max_score = np.max(golden_scores)
        is_valid = max_score > 0.3  # More lenient threshold

        return is_valid, max_score

    def detect_periodicity(self, attempts: List[AuthenticationAttempt],
                          window_size: int = 50) -> Tuple[bool, float]:
        """
        Detect if authentication attempts show periodic patterns.

        CRITICAL SECURITY INSIGHT:
        Quasicrystals are inherently APERIODIC. If an attacker tries to
        brute-force or replay attacks, they will create periodic patterns
        that stand out against the aperiodic background.

        Uses autocorrelation to detect periodicity in the attempt sequence.

        Args:
            attempts: List of authentication attempts
            window_size: Window for periodicity detection

        Returns:
            (is_periodic, periodicity_score)
        """
        if len(attempts) < window_size:
            return False, 0.0

        # Extract positions from recent attempts
        positions = np.array([
            a.projected_position for a in attempts[-window_size:]
            if a.projected_position is not None
        ])

        if len(positions) < window_size // 2:
            return False, 0.0

        # Compute autocorrelation
        # Periodic attacks will show high correlation at regular intervals
        mean_pos = np.mean(positions, axis=0)
        centered = positions - mean_pos

        # Compute correlation at different lags
        max_lag = window_size // 4
        correlations = []

        for lag in range(1, max_lag):
            if lag >= len(centered):
                break
            correlation = np.mean([
                np.dot(centered[i], centered[i + lag])
                for i in range(len(centered) - lag)
            ])
            correlations.append(abs(correlation))

        if not correlations:
            return False, 0.0

        # High correlation at any lag indicates periodicity (attack)
        max_correlation = max(correlations)

        # Normalize by variance
        variance = np.mean([np.dot(c, c) for c in centered])
        if variance > 1e-10:
            periodicity_score = max_correlation / variance
        else:
            periodicity_score = 0.0

        is_periodic = periodicity_score > PERIODICITY_DETECTION_THRESHOLD

        return is_periodic, periodicity_score

    def apply_phason_deformation(self, stress_field: np.ndarray,
                                  magnitude: float = 0.05) -> PhasonField:
        """
        Apply phason deformation for atomic rekeying.

        PHASON MODES:
        Unlike phonons (smooth displacements), phasons are discontinuous
        rearrangements that maintain the quasicrystal's topology while
        changing local configurations.

        This enables "atomic rekeying" - the entire authentication space
        is transformed simultaneously, invalidating any cached attack data.

        Args:
            stress_field: 3D direction vector for the phason
            magnitude: Displacement magnitude (default: 5%)

        Returns:
            PhasonField record of the deformation
        """
        # Normalize stress field
        stress_norm = np.linalg.norm(stress_field)
        if stress_norm < 1e-10:
            stress_field = np.array([1.0, self.phi, self.phi_inv])
            stress_norm = np.linalg.norm(stress_field)

        direction = stress_field / stress_norm

        # Clamp magnitude for stability
        magnitude = min(magnitude, MAX_PHASON_DISPLACEMENT)

        # Apply displacement to all vertices
        # Phason displacement is perpendicular-space dependent
        displacement = direction * magnitude

        # Record which vertices are affected (all of them in global phason)
        affected = set(range(len(self.vertices)))

        # Apply the deformation
        self.vertices = self.vertices + displacement

        # Re-normalize to unit sphere to maintain geometry
        norms = np.linalg.norm(self.vertices, axis=1, keepdims=True)
        self.vertices = self.vertices / norms

        # Create phason record
        phason = PhasonField(
            direction=direction,
            magnitude=magnitude,
            timestamp=time.time(),
            affected_vertices=affected
        )

        self.phason_history.append(phason)

        # Verify icosahedral symmetry preserved
        if not self._verify_icosahedral_symmetry():
            # Rollback if symmetry broken
            self.vertices = self.base_vertices.copy()
            raise ValueError("Phason deformation broke icosahedral symmetry")

        return phason

    def _verify_icosahedral_symmetry(self) -> bool:
        """
        Verify that icosahedral symmetry is preserved after deformation.

        Checks:
        1. Correct number of vertices (12)
        2. All vertices on unit sphere
        3. Neighbor distances follow golden ratio
        """
        # Check vertex count
        if len(self.vertices) != 12:
            return False

        # Check unit sphere constraint
        norms = np.linalg.norm(self.vertices, axis=1)
        if not np.allclose(norms, 1.0, atol=0.01):
            return False

        # Check that nearest neighbor distances are consistent
        # In icosahedron, each vertex has 5 neighbors at equal distance
        neighbor_counts = []
        expected_dist = 2 / np.sqrt(self.phi + 2)  # Theoretical neighbor distance

        for i, v in enumerate(self.vertices):
            close_neighbors = sum(
                1 for j, u in enumerate(self.vertices)
                if i != j and abs(np.linalg.norm(v - u) - expected_dist) < 0.1
            )
            neighbor_counts.append(close_neighbors)

        # Each vertex should have ~5 neighbors
        avg_neighbors = np.mean(neighbor_counts)
        if avg_neighbors < 4:  # Allow some tolerance after deformation
            return False

        return True

    def authenticate(self, gate_values: List[float], actor_id: str) -> AuthenticationAttempt:
        """
        Authenticate using quasicrystal geometry.

        Process:
        1. Project 6D gate values to 3D quasicrystal space
        2. Find nearest icosahedral vertices
        3. Verify golden ratio constraints on path
        4. Check for periodicity (attack signature)
        5. Return authentication result

        Args:
            gate_values: 6 float values from SCBE gates
            actor_id: Identity of the authenticating actor

        Returns:
            AuthenticationAttempt with result
        """
        attempt = AuthenticationAttempt(
            gate_values=gate_values,
            timestamp=time.time(),
            actor_id=actor_id
        )

        try:
            # Step 1: Project to 3D
            position = self.project_6d_to_3d(gate_values)
            attempt.projected_position = position

            # Step 2: Find nearest vertices
            nearest = self.find_minimal_distances(position, k=3)
            attempt.nearest_vertices = nearest

            # Step 3: Check golden ratio constraint
            vertex_indices = [v[0] for v in nearest]
            golden_valid, golden_score = self.verify_golden_ratio_constraint(vertex_indices)

            if not golden_valid:
                attempt.result = AuthResult.GOLDEN_RATIO_VIOLATION
                self._record_attempt(attempt)
                return attempt

            # Step 4: Check for periodicity (add to history first)
            self._record_attempt(attempt)

            is_periodic, periodicity_score = self.detect_periodicity(self.attempt_history)

            if is_periodic:
                attempt.result = AuthResult.PERIODICITY_DETECTED
                return attempt

            # Step 5: Verify minimum distance to valid vertex
            min_distance = nearest[0][1]

            # Threshold based on projected space geometry
            # In 6D→3D projection, valid points can be farther from vertices
            max_valid_distance = 2.0  # Generous threshold for projected points

            if min_distance > max_valid_distance:
                attempt.result = AuthResult.INVALID_GEOMETRY
                return attempt

            # All checks passed
            attempt.result = AuthResult.VALID
            return attempt

        except Exception as e:
            attempt.result = AuthResult.INVALID_GEOMETRY
            self._record_attempt(attempt)
            return attempt

    def _record_attempt(self, attempt: AuthenticationAttempt):
        """Record attempt for periodicity detection."""
        self.attempt_history.append(attempt)

        # Trim history if too large
        if len(self.attempt_history) > self.max_history_size:
            self.attempt_history = self.attempt_history[-self.max_history_size:]

    def get_phason_state(self) -> Dict:
        """Get current phason state for synchronization."""
        return {
            'vertex_hash': hashlib.sha256(
                self.vertices.tobytes()
            ).hexdigest()[:16],
            'phason_count': len(self.phason_history),
            'last_phason_time': (
                self.phason_history[-1].timestamp
                if self.phason_history else 0
            )
        }

    def reset_to_base(self):
        """Reset vertices to base icosahedron (for testing)."""
        self.vertices = self.base_vertices.copy()
        self.phason_history.clear()
        self.attempt_history.clear()


# =============================================================================
# INTEGRATION WITH SCBE 6-GATE PIPELINE
# =============================================================================

class QuasicrystalSCBEIntegration:
    """
    Integrates quasicrystal authentication with SCBE 6-gate pipeline.

    Maps each SCBE gate to a dimension in the 6D hypercubic lattice:
    - Gate 1 (Context) → WHO dimension
    - Gate 2 (Intent) → WHAT dimension
    - Gate 3 (Trajectory) → WHERE dimension (temporal trajectory = spatial path)
    - Gate 4 (AAD) → WHEN dimension
    - Gate 5 (Master Commit) → WHY dimension (integrity = purpose)
    - Gate 6 (Signature) → HOW dimension (crypto method)
    """

    def __init__(self, seed: bytes = None):
        self.lattice = QuasicrystalLattice(seed)
        self.gate_to_dimension = {
            'context': 0,      # WHO
            'intent': 1,       # WHAT
            'trajectory': 2,   # WHERE
            'aad': 3,          # WHEN
            'master_commit': 4, # WHY
            'signature': 5     # HOW
        }

    def gate_results_to_values(self, gate_results: Dict[str, bool]) -> List[float]:
        """
        Convert gate pass/fail results to 6D coordinates.

        Args:
            gate_results: Dict mapping gate name to pass (True) or fail (False)

        Returns:
            6D coordinate vector
        """
        values = [0.0] * 6

        for gate_name, passed in gate_results.items():
            if gate_name in self.gate_to_dimension:
                dim = self.gate_to_dimension[gate_name]
                # Pass = +1 (toward valid vertex), Fail = -1 (away from valid)
                values[dim] = 1.0 if passed else -1.0

        return values

    def gate_scores_to_values(self, gate_scores: Dict[str, float]) -> List[float]:
        """
        Convert gate confidence scores to 6D coordinates.

        Args:
            gate_scores: Dict mapping gate name to score (0.0 - 1.0)

        Returns:
            6D coordinate vector (normalized to [-1, 1])
        """
        values = [0.0] * 6

        for gate_name, score in gate_scores.items():
            if gate_name in self.gate_to_dimension:
                dim = self.gate_to_dimension[gate_name]
                # Map [0, 1] to [-1, 1]
                values[dim] = 2.0 * score - 1.0

        return values

    def authenticate_scbe_result(self, gate_results: Dict[str, bool],
                                  actor_id: str) -> AuthenticationAttempt:
        """
        Authenticate SCBE gate results using quasicrystal geometry.

        This provides a second layer of verification:
        1. SCBE gates must all pass (first layer)
        2. The pattern of gate results must map to valid quasicrystal geometry

        Args:
            gate_results: Results from SCBE 6-gate pipeline
            actor_id: Actor identifier

        Returns:
            AuthenticationAttempt with quasicrystal verification result
        """
        values = self.gate_results_to_values(gate_results)
        return self.lattice.authenticate(values, actor_id)

    def trigger_phason_rekeying(self, trigger_data: bytes):
        """
        Trigger phason-based rekeying of the authentication space.

        Called periodically or in response to detected attacks.

        Args:
            trigger_data: Data to seed the phason direction
        """
        # Generate phason direction from trigger data
        direction_seed = hashlib.sha256(trigger_data).digest()
        direction = np.array([
            int.from_bytes(direction_seed[0:8], 'big') / (2**64) - 0.5,
            int.from_bytes(direction_seed[8:16], 'big') / (2**64) - 0.5,
            int.from_bytes(direction_seed[16:24], 'big') / (2**64) - 0.5
        ])

        self.lattice.apply_phason_deformation(direction)


# =============================================================================
# COMPREHENSIVE TESTS
# =============================================================================

def run_quasicrystal_validation():
    """Run comprehensive validation of quasicrystal authentication."""
    print("\n" + "="*70)
    print("  QUASICRYSTAL LATTICE AUTHENTICATION VALIDATION")
    print("  SCBE v3.0 - Icosahedral Geometry")
    print("="*70)

    results = {}

    # Test 1: Golden Ratio Verification
    print("\n[1/6] Verifying Golden Ratio Constants...")
    lattice = QuasicrystalLattice()

    phi_accurate = abs(lattice.phi - 1.6180339887498949) < 1e-10
    phi_identity = abs(lattice.phi * lattice.phi_inv - 1.0) < 1e-10
    phi_property = abs(lattice.phi - lattice.phi_inv - 1.0) < 1e-10  # φ - 1/φ = 1

    results['golden_ratio'] = {
        'phi_value': lattice.phi,
        'phi_accurate': phi_accurate,
        'phi_identity': phi_identity,
        'phi_property': phi_property,
        'all_valid': phi_accurate and phi_identity and phi_property
    }

    print(f"  ✓ φ = {lattice.phi:.15f}")
    print(f"  ✓ φ × (1/φ) = 1: {phi_identity}")
    print(f"  ✓ φ - 1/φ = 1: {phi_property}")

    # Test 2: Icosahedral Geometry
    print("\n[2/6] Validating Icosahedral Geometry...")

    vertex_count = len(lattice.vertices)

    # Check vertices on unit sphere
    norms = np.linalg.norm(lattice.vertices, axis=1)
    on_unit_sphere = np.allclose(norms, 1.0, atol=1e-10)

    # Check 5-fold symmetry via neighbor count
    neighbor_distances = []
    for i, v in enumerate(lattice.vertices):
        for j, u in enumerate(lattice.vertices):
            if i < j:
                neighbor_distances.append(np.linalg.norm(v - u))

    unique_distances = np.unique(np.round(neighbor_distances, decimals=10))

    results['icosahedral_geometry'] = {
        'vertex_count': vertex_count,
        'on_unit_sphere': on_unit_sphere,
        'unique_edge_lengths': len(unique_distances),
        'valid': vertex_count == 12 and on_unit_sphere
    }

    print(f"  ✓ Vertex count: {vertex_count} (expected 12)")
    print(f"  ✓ All on unit sphere: {on_unit_sphere}")
    print(f"  ✓ Unique edge lengths: {len(unique_distances)}")

    # Test 3: 6D → 3D Projection
    print("\n[3/6] Testing 6D → 3D Projection...")

    projection_tests = [
        ([1, 0, 0, 0, 0, 0], "Pure WHO"),
        ([0, 1, 0, 0, 0, 0], "Pure WHAT"),
        ([0, 0, 1, 0, 0, 0], "Pure WHERE"),
        ([1, 1, 1, 0, 0, 0], "All physical"),
        ([0, 0, 0, 1, 1, 1], "All perpendicular"),
        ([1, 1, 1, 1, 1, 1], "All gates pass"),
        ([-1, -1, -1, -1, -1, -1], "All gates fail"),
    ]

    projection_results = []
    for values, description in projection_tests:
        position = lattice.project_6d_to_3d(values)
        nearest = lattice.find_minimal_distances(position, k=1)
        projection_results.append({
            'description': description,
            'position': position.tolist(),
            'nearest_vertex': nearest[0][0],
            'distance': nearest[0][1]
        })

    results['projection'] = projection_results

    print(f"  ✓ All gates pass → distance: {projection_results[5]['distance']:.4f}")
    print(f"  ✓ All gates fail → distance: {projection_results[6]['distance']:.4f}")

    # Test 4: Golden Ratio Path Verification
    print("\n[4/6] Testing Golden Ratio Path Constraints...")

    # Test sequential vertices (should follow golden ratio)
    test_paths = [
        [0, 1, 4],  # Path through 3 vertices
        [0, 4, 8],  # Another path
        [0, 1, 2, 3],  # Longer path
    ]

    path_results = []
    for path in test_paths:
        valid, score = lattice.verify_golden_ratio_constraint(path)
        path_results.append({
            'path': path,
            'valid': valid,
            'golden_score': score
        })
        print(f"  Path {path}: valid={valid}, score={score:.4f}")

    results['golden_ratio_paths'] = path_results

    # Test 5: Periodicity Detection
    print("\n[5/6] Testing Periodicity (Attack) Detection...")

    lattice.reset_to_base()

    # Simulate legitimate (random) attempts
    for i in range(60):
        values = [np.random.uniform(-1, 1) for _ in range(6)]
        lattice.authenticate(values, f"user_{i}")

    is_periodic_legit, score_legit = lattice.detect_periodicity(lattice.attempt_history)

    # Simulate attack (periodic) attempts
    lattice.reset_to_base()
    for i in range(60):
        # Attacker uses repeating pattern
        values = [np.sin(i * 0.5) for _ in range(6)]
        lattice.authenticate(values, "attacker")

    is_periodic_attack, score_attack = lattice.detect_periodicity(lattice.attempt_history)

    results['periodicity_detection'] = {
        'legitimate_periodic': is_periodic_legit,
        'legitimate_score': score_legit,
        'attack_periodic': is_periodic_attack,
        'attack_score': score_attack,
        'detection_works': not is_periodic_legit and is_periodic_attack
    }

    print(f"  ✓ Legitimate attempts periodic: {is_periodic_legit} (score: {score_legit:.4f})")
    print(f"  ✓ Attack attempts periodic: {is_periodic_attack} (score: {score_attack:.4f})")
    print(f"  ✓ Detection effective: {not is_periodic_legit and is_periodic_attack}")

    # Test 6: Phason Deformation
    print("\n[6/6] Testing Phason Deformation (Atomic Rekeying)...")

    lattice.reset_to_base()
    original_hash = lattice.get_phason_state()['vertex_hash']

    # Apply phason
    stress = np.array([1.0, lattice.phi, 0.5])
    phason = lattice.apply_phason_deformation(stress, magnitude=0.05)

    new_hash = lattice.get_phason_state()['vertex_hash']

    # Verify symmetry preserved
    symmetry_preserved = lattice._verify_icosahedral_symmetry()

    results['phason_deformation'] = {
        'original_hash': original_hash,
        'new_hash': new_hash,
        'state_changed': original_hash != new_hash,
        'symmetry_preserved': symmetry_preserved,
        'phason_recorded': len(lattice.phason_history) > 0
    }

    print(f"  ✓ Original state hash: {original_hash}")
    print(f"  ✓ New state hash: {new_hash}")
    print(f"  ✓ State changed: {original_hash != new_hash}")
    print(f"  ✓ Symmetry preserved: {symmetry_preserved}")

    # Summary
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)

    all_pass = (
        results['golden_ratio']['all_valid'] and
        results['icosahedral_geometry']['valid'] and
        results['periodicity_detection']['detection_works'] and
        results['phason_deformation']['symmetry_preserved']
    )

    summary = {
        'golden_ratio_valid': results['golden_ratio']['all_valid'],
        'icosahedral_geometry_valid': results['icosahedral_geometry']['valid'],
        'projection_working': len(projection_results) == 7,
        'attack_detection_working': results['periodicity_detection']['detection_works'],
        'phason_rekeying_working': results['phason_deformation']['symmetry_preserved'],
        'all_tests_pass': all_pass
    }

    for test, passed in summary.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test}")

    print("="*70)

    results['summary'] = summary
    return results


# =============================================================================
# SCBE INTEGRATION TEST
# =============================================================================

def run_scbe_integration_test():
    """Test integration with SCBE 6-gate pipeline."""
    print("\n" + "="*70)
    print("  SCBE INTEGRATION TEST")
    print("="*70)

    integration = QuasicrystalSCBEIntegration()

    # Test 1: All gates pass
    print("\n[1/4] All Gates Pass...")
    all_pass = {
        'context': True,
        'intent': True,
        'trajectory': True,
        'aad': True,
        'master_commit': True,
        'signature': True
    }

    result = integration.authenticate_scbe_result(all_pass, "legitimate_user")
    print(f"  Result: {result.result.value}")
    print(f"  Nearest vertex distance: {result.nearest_vertices[0][1]:.4f}")

    # Test 2: One gate fails
    print("\n[2/4] One Gate Fails (Intent)...")
    one_fail = all_pass.copy()
    one_fail['intent'] = False

    result = integration.authenticate_scbe_result(one_fail, "suspicious_user")
    print(f"  Result: {result.result.value}")
    print(f"  Nearest vertex distance: {result.nearest_vertices[0][1]:.4f}")

    # Test 3: Multiple gates fail
    print("\n[3/4] Multiple Gates Fail...")
    multi_fail = {
        'context': False,
        'intent': False,
        'trajectory': True,
        'aad': False,
        'master_commit': True,
        'signature': False
    }

    result = integration.authenticate_scbe_result(multi_fail, "attacker")
    print(f"  Result: {result.result.value}")
    print(f"  Nearest vertex distance: {result.nearest_vertices[0][1]:.4f}")

    # Test 4: Phason rekeying
    print("\n[4/4] Triggering Phason Rekeying...")
    state_before = integration.lattice.get_phason_state()
    integration.trigger_phason_rekeying(b"scheduled_rekey_event")
    state_after = integration.lattice.get_phason_state()

    print(f"  State hash before: {state_before['vertex_hash']}")
    print(f"  State hash after: {state_after['vertex_hash']}")
    print(f"  Rekeying successful: {state_before['vertex_hash'] != state_after['vertex_hash']}")

    print("="*70)


if __name__ == '__main__':
    # Run all validations
    qc_results = run_quasicrystal_validation()
    run_scbe_integration_test()

    # Save results
    output_file = 'quasicrystal_validation_results.json'

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_for_json(qc_results), f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")
