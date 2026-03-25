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
Quasicrystal Lattice Module for SCBE-AETHERMOORE

Provides geometric verification using:
- Icosahedral Quasicrystal (aperiodic 6D->3D projection)
- Polyhedral Hamiltonian Defense Manifold (16 canonical polyhedra)
- Integrated HMAC chain binding

Simple Usage:
    from symphonic_cipher.scbe_aethermoore.qc_lattice import quick_validate

    result = quick_validate("user123", "read_data")
    print(result.decision)  # ALLOW, DENY, QUARANTINE, or SNAP

Full Usage:
    from symphonic_cipher.scbe_aethermoore.qc_lattice import (
        IntegratedAuditChain,
        QuasicrystalLattice,
        PHDMHamiltonianPath
    )

    # Create audit chain with PQC
    chain = IntegratedAuditChain(use_pqc=True)

    # Add entries
    validation, signature = chain.add_entry("user", "action")

    # Verify everything
    is_valid, errors = chain.verify_all()
"""

# Quasicrystal Lattice
from .quasicrystal import (
    # Core classes
    QuasicrystalLattice,
    PQCQuasicrystalLattice,

    # Result types
    ValidationResult,
    ValidationStatus,
    LatticePoint,

    # Constants
    PHI,
    TAU,
)

# PHDM (Polyhedral Hamiltonian Defense Manifold)
from .phdm import (
    # Core classes
    Polyhedron,
    PolyhedronType,
    PHDMHamiltonianPath,
    PHDMDeviationDetector,
    HamiltonianNode,

    # Family functions
    get_phdm_family,
    get_family_summary,
    validate_all_polyhedra,

    # Coordinate generators
    create_tetrahedron_coords,
    create_cube_coords,
    create_octahedron_coords,
    create_dodecahedron_coords,
    create_icosahedron_coords,
)

# Integration with HMAC Chain
from .integration import (
    # Main classes
    QuasicrystalHMACChain,
    IntegratedAuditChain,

    # Result types
    IntegratedDecision,
    IntegratedValidation,

    # Convenience functions
    create_integrated_chain,
    quick_validate,

    # Constants
    NONCE_BYTES,
    KEY_LEN,
    AUDIT_CHAIN_IV,
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
    "QuasicrystalLattice",
    "PQCQuasicrystalLattice",
    "ValidationResult",
    "ValidationStatus",
    "LatticePoint",
    "PHI",
    "TAU",

    # PHDM
    "Polyhedron",
    "PolyhedronType",
    "PHDMHamiltonianPath",
    "PHDMDeviationDetector",
    "HamiltonianNode",
    "get_phdm_family",
    "get_family_summary",
    "validate_all_polyhedra",
    "create_tetrahedron_coords",
    "create_cube_coords",
    "create_octahedron_coords",
    "create_dodecahedron_coords",
    "create_icosahedron_coords",

    # Integration
    "QuasicrystalHMACChain",
    "IntegratedAuditChain",
    "IntegratedDecision",
    "IntegratedValidation",
    "create_integrated_chain",
    "quick_validate",
    "NONCE_BYTES",
    "KEY_LEN",
    "AUDIT_CHAIN_IV",
]

__version__ = "1.0.0"
