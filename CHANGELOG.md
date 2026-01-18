# Changelog

All notable changes to the SCBE-AETHERMOORE Python SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-01-18

### Added

- **Production Service Package** (`scbe_production/`)
  - Unified `SCBEProductionService` API for all operations
  - Environment-aware configuration (development/staging/production)
  - Structured JSON audit logging with sensitive data redaction
  - Commercial-grade exception hierarchy with error codes
  - Health check endpoint for monitoring

- **GeoSeal Geometric Trust Manifold**
  - Dual-space projection (sphere S^n + hypercube [0,1]^m)
  - Interior/exterior path classification
  - Time dilation factor: τ = exp(-γ·r)
  - Configurable threshold (default 0.5)

- **HAL Attention Layer**
  - Harmonic coupling matrix: Λ[i,j] = R^(d_Q·d_K)
  - Query-key distance calculations
  - Golden ratio (φ) based scaling

- **Integrated Demo System**
  - 4 security scenarios (benign, stolen credentials, insider threat, hallucination)
  - JSON report output for CI/CD integration
  - Cymatic resonance physics visualization

- **Docker Support**
  - Multi-stage Dockerfile (builder, production, development)
  - Docker Compose with profiles (dev, cache, monitoring)
  - Health checks and resource limits

- **Documentation**
  - `DEPLOYMENT.md` - Comprehensive deployment guide
  - `docs/MATHEMATICAL_PROOFS.md` - Formal verification
  - `CONTRIBUTING.md` - Contribution guidelines
  - `API_REFERENCE.md` - Full API documentation

- **CI/CD**
  - GitHub Actions workflows for testing and linting
  - VS Code configuration for debugging

### Changed

- **GeoSeal Threshold**: Changed from 0.3 to 0.5 for proper interior/exterior classification
- **Governance Logging**: Moved `agent_id` to details dict to fix keyword argument conflict
- **Package Structure**: Reorganized for PyPI publishing compatibility

### Fixed

- Import error for `harmonic_scaling` function
- GeoSeal false exterior classifications for benign requests
- Audit logger `agent_id` parameter conflict

---

## [2.1.0] - 2026-01-15

### Added

- **Dual Lattice Consensus**
  - MLWE primal lattice verification
  - MSIS dual lattice signatures
  - Consensus state machine (PENDING → SETTLED/FAILED)

- **Quasicrystal Lattice Validation**
  - Fibonacci word generation
  - Icosahedral projection (E8 → 3D)
  - Crystallinity scoring

- **Sacred Tongue Tokenizer**
  - 6 Sacred Tongues (Aetheric Khöömei, Runic Verse, Ciphered Aramaic, Umbral Script, Draconic Syllabary, Kaothic Oracle)
  - 256 tokens per tongue
  - Bijective encoding/decoding

### Changed

- PQC backend abstraction for mock/real switching
- Improved error messages with context

---

## [2.0.0] - 2026-01-10

### Added

- **14-Layer Hyperbolic Governance Pipeline**
  1. Input Validation
  2. Agent Authentication
  3. Topic Binding
  4. Position Verification
  5. Hyperbolic Distance
  6. Langues Metric
  7. Spiral Seal SS1
  8. PQC Key Encapsulation
  9. Cymatic Storage
  10. Signature Generation
  11. Risk Assessment
  12. Harmonic Wall
  13. Governance Decision
  14. Audit Trail

- **Post-Quantum Cryptography**
  - ML-KEM-768 (Kyber) key encapsulation
  - ML-DSA-65 (Dilithium) signatures
  - Hybrid mode support

- **Harmonic Scaling Law**
  - Exponential cost: H(d, R) = R^(d²)
  - SNAP protocol failsafe
  - φ-based frequency harmonics

### Changed

- Complete architecture redesign
- Migrated from AES to PQC primitives

---

## [1.0.0] - 2026-01-01

### Added

- Initial release
- Basic SpiralSeal cipher
- Fibonacci position binding
- Simple governance (ALLOW/DENY)

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 3.0.0 | 2026-01-18 | Production package, GeoSeal, Docker |
| 2.1.0 | 2026-01-15 | Dual lattice, quasicrystal, Sacred Tongues |
| 2.0.0 | 2026-01-10 | 14-layer pipeline, PQC, harmonic scaling |
| 1.0.0 | 2026-01-01 | Initial release |

---

## Upgrade Guide

### From 2.x to 3.0

```python
# Old (2.x)
from symphonic_cipher import SCBESystem
system = SCBESystem()
result = system.encrypt(data)

# New (3.0)
from scbe_production.service import SCBEProductionService
service = SCBEProductionService()
shard = service.seal_memory(
    plaintext=data,
    agent_id="my-agent",
    topic="default",
    position=(1, 2, 3, 5, 8, 13),
)
```

### Configuration Migration

```python
# Old (2.x) - hardcoded values
THRESHOLD = 0.3

# New (3.0) - environment configuration
from scbe_production.config import get_config
config = get_config()
threshold = config.geoseal.interior_threshold  # 0.5 default
```

---

## Deprecation Notices

### 3.0.0

- `SCBESystem` class is deprecated; use `SCBEProductionService`
- Direct cipher access is deprecated; use service methods
- Hardcoded configuration is deprecated; use `ProductionConfig`

---

*For the complete commit history, see the [GitHub repository](https://github.com/issdandavis/aws-lambda-simple-web-app).*
