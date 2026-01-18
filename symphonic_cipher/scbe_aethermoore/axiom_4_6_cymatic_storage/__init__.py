"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  AXIOM 4.6: CYMATIC VOXEL STORAGE                            ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  HolographicQRCube: Data storage at nodal positions in standing wave fields  ║
║                                                                              ║
║  Key Concepts:                                                               ║
║  - Voxels stored at positions satisfying Chladni nodal equation              ║
║  - Access requires agent vector producing resonance at voxel position        ║
║  - KD-Tree spatial indexing using harmonic distance metric                   ║
║  - 6D vector-derived mode parameters for access control                      ║
║                                                                              ║
║  Storage Modes:                                                              ║
║  - PUBLIC: No resonance check required                                       ║
║  - RESONANCE: Requires cymatic resonance match                               ║
║  - ENCRYPTED: Requires resonance + decryption key                            ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 6 (Cymatic Voxel Storage)                                          ║
║  Author: Isaac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from cymatic_storage module
from ..cymatic_storage import (
    # Configuration & Types
    StorageMode,
    Voxel,
    CubeConfig,
    CubeStats,

    # Core Storage
    HolographicQRCube,

    # KD-Tree
    KDNode,
    KDTree,

    # Utilities
    compute_access_vector,
    create_voxel_grid,
    get_cymatic_storage_stats,
)

__all__ = [
    'StorageMode',
    'Voxel',
    'CubeConfig',
    'CubeStats',
    'HolographicQRCube',
    'KDNode',
    'KDTree',
    'compute_access_vector',
    'create_voxel_grid',
    'get_cymatic_storage_stats',
]

AXIOM_ID = "4.6"
AXIOM_TITLE = "Cymatic Voxel Storage"
AXIOM_FORMULA = "Access(v, agent) = resonance(v.position, mode(agent.vector))"
