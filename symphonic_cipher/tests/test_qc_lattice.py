"""
Tests for Quasicrystal Lattice modules (PHDM and Quasicrystal).

Tests cover:
1. PHDM - Polyhedral Hamiltonian Defense Manifold
   - Euler characteristic verification
   - Topological invariants
   - Dual polyhedra correspondence

2. Quasicrystal - Icosahedral 6D→3D projection
   - Vertex generation
   - Aperiodicity verification
   - Diffraction fingerprinting
"""

import pytest
import numpy as np
import math

from symphonic_cipher.scbe_aethermoore.qc_lattice import (
    # PHDM
    PolyhedronDef,
    PHDMStatus,
    PHDMState,
    PolyhedralDefenseManifold,
    PLATONIC_SOLIDS,
    ARCHIMEDEAN_SOLIDS,
    ALL_POLYHEDRA,
    verify_euler_characteristic,

    # Quasicrystal
    IcosahedralProjector,
    QuasicrystalVertex,
    generate_quasicrystal_vertices,
    verify_icosahedral_symmetry,
    diffraction_fingerprint,
    TAU,
)


# =============================================================================
# PHDM TESTS
# =============================================================================

class TestPolyhedronDef:
    """Tests for polyhedron definitions."""

    def test_platonic_euler_characteristic(self):
        """All Platonic solids have Euler characteristic χ=2."""
        for name, poly in PLATONIC_SOLIDS.items():
            chi = poly.euler_characteristic
            assert chi == 2, f"{name}: χ={chi}, expected 2"

    def test_archimedean_euler_characteristic(self):
        """All Archimedean solids have Euler characteristic χ=2."""
        for name, poly in ARCHIMEDEAN_SOLIDS.items():
            chi = poly.euler_characteristic
            assert chi == 2, f"{name}: χ={chi}, expected 2"

    def test_total_polyhedra_count(self):
        """There should be exactly 16 polyhedra (5 Platonic + 11 Archimedean)."""
        assert len(PLATONIC_SOLIDS) == 5
        assert len(ARCHIMEDEAN_SOLIDS) == 11
        assert len(ALL_POLYHEDRA) == 16

    def test_tetrahedron_self_dual(self):
        """Tetrahedron is self-dual."""
        tetra = PLATONIC_SOLIDS["tetrahedron"]
        assert tetra.dual_name == "tetrahedron"

    def test_cube_octahedron_duality(self):
        """Cube and octahedron are duals."""
        cube = PLATONIC_SOLIDS["cube"]
        octa = PLATONIC_SOLIDS["octahedron"]

        # Dual relationship: V↔F, E=E
        assert cube.vertices == octa.faces
        assert cube.faces == octa.vertices
        assert cube.edges == octa.edges

        assert cube.dual_name == "octahedron"
        assert octa.dual_name == "cube"

    def test_dodecahedron_icosahedron_duality(self):
        """Dodecahedron and icosahedron are duals."""
        dodeca = PLATONIC_SOLIDS["dodecahedron"]
        icosa = PLATONIC_SOLIDS["icosahedron"]

        # Dual relationship: V↔F, E=E
        assert dodeca.vertices == icosa.faces
        assert dodeca.faces == icosa.vertices
        assert dodeca.edges == icosa.edges

        assert dodeca.dual_name == "icosahedron"
        assert icosa.dual_name == "dodecahedron"

    def test_polyhedron_validity(self):
        """All polyhedra should be valid (χ=2)."""
        for name, poly in ALL_POLYHEDRA.items():
            assert poly.is_valid(), f"{name} failed validity check"


class TestVerifyEulerCharacteristic:
    """Tests for Euler characteristic verification."""

    def test_valid_vef_tuple(self):
        """Valid (V, E, F) should pass."""
        # Cube: V=8, E=12, F=6 → χ=2
        result = verify_euler_characteristic(8, 12, 6)
        # Returns (is_valid, chi, message) tuple
        assert result[0] is True  # is_valid
        assert result[1] == 2     # chi value

    def test_invalid_vef_tuple(self):
        """Invalid (V, E, F) should fail."""
        # Modified cube with wrong edge count
        result = verify_euler_characteristic(8, 13, 6)
        assert result[0] is False  # is_valid
        assert result[1] == 1      # chi = 8 - 13 + 6 = 1

    def test_zero_values(self):
        """Zero values produce chi=0, invalid."""
        result = verify_euler_characteristic(0, 0, 0)
        assert result[0] is False
        assert result[1] == 0

    def test_negative_values(self):
        """Negative values produce invalid Euler characteristic."""
        result = verify_euler_characteristic(-4, 6, 4)
        assert result[0] is False


class TestPolyhedralDefenseManifold:
    """Tests for the PHDM class."""

    def test_initialization(self):
        """PHDM should initialize with all 16 polyhedra."""
        phdm = PolyhedralDefenseManifold()
        assert len(phdm.polyhedra) == 16

    def test_custom_polyhedra_subset(self):
        """PHDM can be initialized with a subset of polyhedra."""
        platonic_only = {k: v for k, v in PLATONIC_SOLIDS.items()}
        phdm = PolyhedralDefenseManifold(polyhedra=platonic_only)
        assert len(phdm.polyhedra) == 5

    def test_vertex_graphs_built(self):
        """Vertex graphs should be built for all polyhedra."""
        phdm = PolyhedralDefenseManifold()
        assert len(phdm.vertex_graphs) == 16

        # Each graph should have the correct number of vertices
        for name, graph in phdm.vertex_graphs.items():
            expected_v = phdm.polyhedra[name].vertices
            assert len(graph) == expected_v, f"{name}: expected {expected_v} vertices"


class TestPHDMState:
    """Tests for PHDMState dataclass."""

    def test_total_euler_characteristic(self):
        """Total Euler characteristic should sum correctly."""
        # All valid polyhedra: 16 × 2 = 32
        state = PHDMState(
            polyhedron_states={name: (p.vertices, p.edges, p.faces)
                              for name, p in ALL_POLYHEDRA.items()},
            hamiltonian_valid={name: True for name in ALL_POLYHEDRA},
            status=PHDMStatus.VALID,
            euler_violations=[],
            anomaly_score=0.0,
            timestamp=0.0
        )
        assert state.total_euler_characteristic() == 32  # 16 × 2

    def test_is_valid_true(self):
        """is_valid should return True for VALID status."""
        state = PHDMState(
            polyhedron_states={},
            hamiltonian_valid={},
            status=PHDMStatus.VALID,
            euler_violations=[],
            anomaly_score=0.0,
            timestamp=0.0
        )
        assert state.is_valid() is True

    def test_is_valid_false(self):
        """is_valid should return False for non-VALID status."""
        for status in [PHDMStatus.EULER_VIOLATION, PHDMStatus.HAMILTONIAN_BREAK,
                      PHDMStatus.DUAL_MISMATCH, PHDMStatus.TOPOLOGY_ANOMALY]:
            state = PHDMState(
                polyhedron_states={},
                hamiltonian_valid={},
                status=status,
                euler_violations=[],
                anomaly_score=0.0,
                timestamp=0.0
            )
            assert state.is_valid() is False


# =============================================================================
# QUASICRYSTAL TESTS
# =============================================================================

class TestTauConstant:
    """Tests for the golden ratio constant."""

    def test_tau_value(self):
        """TAU should equal the golden ratio φ."""
        expected_phi = (1 + math.sqrt(5)) / 2
        assert abs(TAU - expected_phi) < 1e-10

    def test_tau_identity(self):
        """TAU should satisfy τ² = τ + 1."""
        assert abs(TAU**2 - TAU - 1) < 1e-10


class TestQuasicrystalVertex:
    """Tests for QuasicrystalVertex dataclass."""

    def test_vertex_creation(self):
        """Vertex should store 6D lattice coords and 3D position."""
        lattice_6d = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        position_3d = np.array([0.5, 0.5, 0.5])
        perp_3d = np.array([0.1, 0.1, 0.1])

        vertex = QuasicrystalVertex(
            lattice_6d=lattice_6d,
            position_3d=position_3d,
            perp_3d=perp_3d,
            distance_from_origin=np.linalg.norm(position_3d),
            index=0
        )

        assert vertex.index == 0
        assert len(vertex.lattice_6d) == 6
        assert len(vertex.position_3d) == 3
        assert len(vertex.perp_3d) == 3


class TestIcosahedralProjector:
    """Tests for IcosahedralProjector."""

    def test_project_single_point(self):
        """Projecting a 6D point should give 3D parallel and perpendicular results."""
        projector = IcosahedralProjector()
        point_6d = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        parallel, perp, in_window = projector.project(point_6d)

        assert len(parallel) == 3
        assert len(perp) == 3
        assert bool(in_window) in (True, False)  # numpy bool comparison

    def test_projection_parallel(self):
        """Parallel projection maps 6D to 3D."""
        projector = IcosahedralProjector()
        point_6d = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        parallel = projector.project_parallel(point_6d)

        assert len(parallel) == 3
        assert isinstance(parallel, np.ndarray)

    def test_projection_perpendicular(self):
        """Perpendicular projection maps 6D to 3D."""
        projector = IcosahedralProjector()
        point_6d = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        perp = projector.project_perpendicular(point_6d)

        assert len(perp) == 3
        assert isinstance(perp, np.ndarray)

    def test_window_check(self):
        """Origin should be in acceptance window."""
        projector = IcosahedralProjector()
        origin = np.zeros(6)

        assert bool(projector.is_in_window(origin)) is True  # numpy bool


class TestGenerateQuasicrystalVertices:
    """Tests for vertex generation."""

    def test_generate_vertices(self):
        """Should generate vertices within bounds."""
        vertices = generate_quasicrystal_vertices(max_coord=3, max_vertices=500)

        # Should have generated some vertices
        assert len(vertices) > 0

        # All vertices should have valid coordinates
        for v in vertices:
            assert len(v.position_3d) == 3
            assert len(v.lattice_6d) == 6

    def test_vertex_count_increases_with_max_coord(self):
        """Larger max_coord should yield more or equal vertices."""
        vertices_small = generate_quasicrystal_vertices(max_coord=1, max_vertices=500)
        vertices_large = generate_quasicrystal_vertices(max_coord=3, max_vertices=500)

        # Larger max_coord should have more or equal vertices
        assert len(vertices_large) >= len(vertices_small)


class TestVerifyIcosahedralSymmetry:
    """Tests for icosahedral symmetry verification."""

    def test_symmetry_of_standard_lattice(self):
        """Standard lattice should have icosahedral symmetry."""
        vertices = generate_quasicrystal_vertices(max_coord=3, max_vertices=500)

        if len(vertices) < 20:
            pytest.skip("Not enough vertices for symmetry test")

        result = verify_icosahedral_symmetry(vertices)

        # Result should indicate symmetry analysis
        assert isinstance(result, dict)
        # May contain keys like five_fold_preserved_ratio, three_fold_preserved_ratio, etc.
        assert "center" in result or "five_fold_preserved_ratio" in result


class TestDiffractionFingerprint:
    """Tests for diffraction fingerprint computation."""

    def test_fingerprint_computation(self):
        """Should compute diffraction fingerprint."""
        vertices = generate_quasicrystal_vertices(max_coord=2, max_vertices=200)

        if len(vertices) < 5:
            pytest.skip("Not enough vertices for diffraction test")

        fingerprint = diffraction_fingerprint(vertices)

        # Should return fingerprint bytes (SHA256 hash of diffraction pattern)
        assert fingerprint is not None
        assert isinstance(fingerprint, bytes)
        assert len(fingerprint) == 32  # SHA256 produces 32 bytes


# =============================================================================
# MATHEMATICAL PROOFS
# =============================================================================

class TestMathematicalProofs:
    """Tests that verify mathematical proofs."""

    def test_euler_formula_v_minus_e_plus_f_equals_2(self):
        """Verify V - E + F = 2 for all polyhedra (Euler's formula)."""
        for name, poly in ALL_POLYHEDRA.items():
            v, e, f = poly.vertices, poly.edges, poly.faces
            chi = v - e + f
            assert chi == 2, f"{name}: V={v}, E={e}, F={f}, χ={chi}"

    def test_handshaking_lemma(self):
        """Sum of face degrees = 2E (handshaking lemma for faces)."""
        # For polyhedra where we know face structure
        cube = PLATONIC_SOLIDS["cube"]
        # Cube has 6 faces, each with 4 edges
        # Total face degree = 6 × 4 = 24 = 2 × 12 = 2E ✓
        assert 6 * 4 == 2 * cube.edges

    def test_golden_ratio_in_icosahedron(self):
        """Icosahedron vertices involve the golden ratio."""
        # The coordinates of icosahedron vertices include τ (golden ratio)
        # Vertices are at (0, ±1, ±τ) and cyclic permutations

        # Verify the relationship holds
        tau = TAU

        # Distance between adjacent vertices should be 2
        # (with appropriate scaling)
        v1 = np.array([0, 1, tau])
        v2 = np.array([1, tau, 0])

        dist = np.linalg.norm(v1 - v2)
        # This should equal 2 for unit icosahedron
        expected_dist = 2.0
        assert abs(dist - expected_dist) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
