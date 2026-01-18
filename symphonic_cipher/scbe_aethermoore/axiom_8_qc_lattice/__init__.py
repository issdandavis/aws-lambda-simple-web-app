"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  AXIOM 8: QUASICRYSTAL LATTICE                               ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Penrose-Tiled Quasicrystal Lattice for Aperiodic Key Spaces                 ║
║                                                                              ║
║  Based on 5-fold rotational symmetry with φ-proportioned tiles:              ║
║  - Thin rhombus: angles 36°, 144°                                            ║
║  - Thick rhombus: angles 72°, 108°                                           ║
║                                                                              ║
║  Key Properties:                                                             ║
║  - Aperiodic: Never repeats, preventing pattern-based attacks                ║
║  - Self-similar: Fractal structure at all scales                             ║
║  - φ-inflation: Scaling by Golden Ratio maintains structure                  ║
║                                                                              ║
║  PHDM (Penrose Harmonic Distribution Matrix):                                ║
║    Distributes key material across quasicrystal vertices                     ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 8 (Quasicrystal Lattice)                                           ║
║  Author: Issac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from qc_lattice modules
from ..qc_lattice.quasicrystal import (
    QuasicrystalLattice,
    PQCQuasicrystalLattice,
    LatticePoint,
    ValidationResult,
    ValidationStatus,
)

from ..qc_lattice.phdm import (
    # Classes
    PHDMHamiltonianPath,
    PHDMDeviationDetector,
    HamiltonianNode,
    Polyhedron,
    PolyhedronType,
    # Coord Generators
    create_tetrahedron_coords,
    create_cube_coords,
    create_octahedron_coords,
    create_dodecahedron_coords,
    create_icosahedron_coords,
    # Functions
    get_phdm_family,
    get_family_summary,
    validate_all_polyhedra,
)

from ..qc_lattice.integration import (
    # Classes
    IntegratedAuditChain,
    QuasicrystalHMACChain,
    IntegratedDecision,
    IntegratedValidation,
    # Functions
    create_integrated_chain,
    quick_validate,
)

__all__ = [
    # Quasicrystal
    'QuasicrystalLattice',
    'PQCQuasicrystalLattice',
    'LatticePoint',
    'ValidationResult',
    'ValidationStatus',
    # PHDM
    'PHDMHamiltonianPath',
    'PHDMDeviationDetector',
    'HamiltonianNode',
    'Polyhedron',
    'PolyhedronType',
    'create_tetrahedron_coords',
    'create_cube_coords',
    'create_octahedron_coords',
    'create_dodecahedron_coords',
    'create_icosahedron_coords',
    'get_phdm_family',
    'get_family_summary',
    'validate_all_polyhedra',
    # Integration
    'IntegratedAuditChain',
    'QuasicrystalHMACChain',
    'IntegratedDecision',
    'IntegratedValidation',
    'create_integrated_chain',
    'quick_validate',
]

AXIOM_ID = "8"
AXIOM_TITLE = "Quasicrystal Lattice (Penrose Tiling)"
AXIOM_FORMULA = "φ-inflation: T → φT preserves aperiodic structure"
