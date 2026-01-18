"""
Quasicrystal Lattice Security Module

Provides geometric verification through aperiodic structures:
- Icosahedral quasicrystal (6D→3D projection)
- Polyhedral Hamiltonian Defense Manifold (PHDM)
- Integration with PQC audit chain

Quasicrystals have special properties that make them useful for security:
1. Aperiodic - no repeating pattern to exploit
2. Self-similar - structure preserved at all scales
3. 5-fold symmetry - impossible in periodic crystals
4. Mathematically precise - projection from higher dimensions

Document ID: AETHER-QC-LATTICE-2026-001
"""

from .quasicrystal import (
    # Core projection
    IcosahedralProjector,
    project_6d_to_3d,
    # Vertex generation
    QuasicrystalVertex,
    generate_quasicrystal_vertices,
    # Verification
    verify_icosahedral_symmetry,
    compute_diffraction_pattern,
    diffraction_fingerprint,
    quasicrystal_coherence,
    # Constants
    TAU, ICOSAHEDRAL_MATRIX,
)

from .phdm import (
    # Core PHDM
    PolyhedralDefenseManifold,
    PHDMState,
    PHDMStatus,
    PolyhedronDef,
    # Polyhedra
    PLATONIC_SOLIDS,
    ARCHIMEDEAN_SOLIDS,
    ALL_POLYHEDRA,
    # Verification
    verify_euler_characteristic,
    compute_hamiltonian_path,
    detect_topological_anomaly,
)

__all__ = [
    # Quasicrystal
    "IcosahedralProjector",
    "project_6d_to_3d",
    "QuasicrystalVertex",
    "generate_quasicrystal_vertices",
    "verify_icosahedral_symmetry",
    "compute_diffraction_pattern",
    "diffraction_fingerprint",
    "quasicrystal_coherence",
    "TAU",
    "ICOSAHEDRAL_MATRIX",
    # PHDM
    "PolyhedralDefenseManifold",
    "PHDMState",
    "PHDMStatus",
    "PolyhedronDef",
    "PLATONIC_SOLIDS",
    "ARCHIMEDEAN_SOLIDS",
    "ALL_POLYHEDRA",
    "verify_euler_characteristic",
    "compute_hamiltonian_path",
    "detect_topological_anomaly",
]
