"""
SCBE Production Package
=======================

Commercial-grade implementation of Spectral Context-Bound Encryption
with GeoSeal Geometric Trust Manifold and Spiralverse Protocol.

Version: 1.0.0
License: Proprietary
Copyright: 2026 SCBE Systems

Components:
- PQC: Post-quantum cryptography (ML-KEM, ML-DSA)
- QCLattice: Quasicrystal structures (Penrose, Fibonacci)
- GeoSeal: Geometric trust manifold
- SpiralSeal: SS1 cipher with Sacred Tongues
- Governance: Risk assessment with SNAP protocol
"""

__version__ = "1.0.0"
__author__ = "SCBE Systems"

from .config import ProductionConfig, get_config
from .logging import AuditLogger, get_logger
from .exceptions import (
    SCBEError,
    PQCError,
    GovernanceError,
    ValidationError,
    AuthenticationError,
)

__all__ = [
    "__version__",
    "ProductionConfig",
    "get_config",
    "AuditLogger",
    "get_logger",
    "SCBEError",
    "PQCError",
    "GovernanceError",
    "ValidationError",
    "AuthenticationError",
]
