# SCBE-AETHERMOORE Python SDK - API Reference

Complete API documentation for the SCBE-AETHERMOORE Python SDK v3.0.0.

---

## Table of Contents

- [Quick Start](#quick-start)
- [SCBEProductionService](#scbeproductionservice)
- [Data Types](#data-types)
- [Configuration](#configuration)
- [Exceptions](#exceptions)
- [Logging](#logging)
- [Low-Level APIs](#low-level-apis)

---

## Quick Start

```python
from scbe_production.service import SCBEProductionService, AccessRequest

# Initialize service
service = SCBEProductionService()

# Seal a memory shard
shard = service.seal_memory(
    plaintext=b"Sensitive AI memory data",
    agent_id="agent-001",
    topic="reasoning",
    position=(1, 2, 3, 5, 8, 13),
)

# Access with governance check
request = AccessRequest(
    agent_id="agent-001",
    message="Retrieving context for task",
    features={"trust_level": 0.9},
    position=(1, 2, 3, 5, 8, 13),
)
response = service.access_memory(request)

if response.decision == "ALLOW":
    print("Access granted:", response.data)
```

---

## SCBEProductionService

The main service class providing a unified API for all SCBE operations.

### Constructor

```python
SCBEProductionService(
    config: Optional[ProductionConfig] = None,
    password: Optional[bytes] = None,
)
```

**Parameters:**
- `config`: Optional configuration object. Uses environment defaults if not provided.
- `password`: Optional master password for key derivation.

**Example:**
```python
from scbe_production.service import SCBEProductionService
from scbe_production.config import ProductionConfig, Environment

# Default configuration
service = SCBEProductionService()

# Custom configuration
config = ProductionConfig(environment=Environment.DEVELOPMENT)
service = SCBEProductionService(config=config, password=b"master-key")
```

---

### Methods

#### `seal_memory()`

Seal a memory shard with full SCBE protection.

```python
def seal_memory(
    self,
    plaintext: bytes,
    agent_id: str,
    topic: str,
    position: Tuple[int, ...],
    tags: Optional[List[str]] = None,
) -> MemoryShard
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `plaintext` | `bytes` | Raw data to seal |
| `agent_id` | `str` | Unique agent identifier |
| `topic` | `str` | Classification topic |
| `position` | `Tuple[int, ...]` | Fibonacci spiral position |
| `tags` | `Optional[List[str]]` | Optional metadata tags |

**Returns:** `MemoryShard` - Sealed memory with cryptographic protection.

**Raises:**
- `ValidationError` - Invalid inputs
- `PQCError` - Cryptographic operation failure

**Example:**
```python
shard = service.seal_memory(
    plaintext=b"Secret reasoning chain",
    agent_id="agent-alpha",
    topic="internal-reasoning",
    position=(1, 1, 2, 3, 5, 8),
    tags=["confidential", "reasoning"],
)
print(f"Sealed at: {shard.timestamp}")
print(f"Shard ID: {shard.shard_id}")
```

---

#### `access_memory()`

Request access to memory with governance verification.

```python
def access_memory(
    self,
    request: AccessRequest,
) -> AccessResponse
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `request` | `AccessRequest` | Access request with context |

**Returns:** `AccessResponse` - Governance decision and optional data.

**Example:**
```python
request = AccessRequest(
    agent_id="agent-beta",
    message="Need context for user query",
    features={
        "trust_level": 0.85,
        "session_duration": 3600,
        "previous_violations": 0,
    },
    position=(1, 2, 3, 5, 8, 13),
)

response = service.access_memory(request)
print(f"Decision: {response.decision}")  # ALLOW, QUARANTINE, DENY, or SNAP
print(f"Risk Score: {response.risk_score}")
print(f"Reason: {response.reason}")
```

---

#### `health_check()`

Check service health and component status.

```python
def health_check(self) -> Dict[str, Any]
```

**Returns:** Health status dictionary.

**Example:**
```python
health = service.health_check()
print(health)
# {
#     "status": "healthy",
#     "version": "3.0.0",
#     "environment": "production",
#     "components": {
#         "pqc": {"status": "healthy", "backend": "MOCK", "kem_level": 768},
#         "governance": {"status": "healthy", "thresholds": {...}},
#         "geoseal": {"status": "healthy", "interior_threshold": 0.5},
#     }
# }
```

---

## Data Types

### MemoryShard

Represents a sealed memory unit.

```python
@dataclass
class MemoryShard:
    shard_id: str              # Unique identifier
    agent_id: str              # Owning agent
    topic: str                 # Classification topic
    position: Tuple[int, ...]  # Fibonacci position
    sealed_data: bytes         # Encrypted payload
    signature: bytes           # PQC signature
    timestamp: str             # ISO 8601 timestamp
    tags: List[str]            # Metadata tags

    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
```

**Example:**
```python
# Serialize for storage
json_data = shard.to_json()

# Convert to dictionary
dict_data = shard.to_dict()
```

---

### AccessRequest

Request to access sealed memory.

```python
@dataclass
class AccessRequest:
    agent_id: str                    # Requesting agent
    message: str                     # Access justification
    features: Dict[str, float]       # Trust features
    position: Tuple[int, ...]        # Target position
```

**Trust Features:**
| Feature | Range | Description |
|---------|-------|-------------|
| `trust_level` | 0.0-1.0 | Agent's trust score |
| `session_duration` | seconds | Current session length |
| `previous_violations` | integer | Past violation count |
| `verification_score` | 0.0-1.0 | Identity verification |

---

### AccessResponse

Response from governance engine.

```python
@dataclass
class AccessResponse:
    decision: str           # ALLOW, QUARANTINE, DENY, SNAP
    risk_score: float       # Calculated risk (0.0-1.0)
    reason: str             # Human-readable explanation
    data: Optional[bytes]   # Decrypted data (if ALLOW)
    geometric_path: str     # interior or exterior
    time_dilation: float    # Temporal factor

    def to_dict(self) -> Dict[str, Any]
```

**Decision Types:**
| Decision | Risk Range | Description |
|----------|------------|-------------|
| `ALLOW` | 0.0 - 0.2 | Access granted |
| `QUARANTINE` | 0.2 - 0.4 | Limited access, flagged |
| `DENY` | 0.4 - 1.0 | Access denied |
| `SNAP` | Critical | Failsafe triggered |

---

## Configuration

### ProductionConfig

Main configuration class with environment support.

```python
from scbe_production.config import ProductionConfig, get_config

# Get default config (from environment)
config = get_config()

# Create custom config
config = ProductionConfig(
    environment=Environment.PRODUCTION,
    pqc=PQCConfig(kem_level=768, dsa_level=65),
    governance=GovernanceConfig(allow_threshold=0.2),
    geoseal=GeoSealConfig(interior_threshold=0.5),
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCBE_ENVIRONMENT` | `production` | Environment mode |
| `SCBE_LOG_LEVEL` | `INFO` | Logging level |
| `SCBE_LOG_FORMAT` | `json` | Log format (json/text) |
| `SCBE_AUDIT_ENABLED` | `true` | Enable audit logging |
| `SCBE_PQC_KEM_LEVEL` | `768` | ML-KEM level (512/768/1024) |
| `SCBE_PQC_DSA_LEVEL` | `65` | ML-DSA level (44/65/87) |
| `SCBE_GOV_ALLOW_THRESHOLD` | `0.20` | Risk threshold for ALLOW |
| `SCBE_GOV_QUARANTINE_THRESHOLD` | `0.40` | Risk threshold for QUARANTINE |
| `SCBE_GEOSEAL_INTERIOR_THRESHOLD` | `0.5` | Interior path threshold |

---

## Exceptions

### Exception Hierarchy

```
SCBEError (base)
├── ValidationError
├── PQCError
├── GovernanceError
├── AuthenticationError
├── ConfigurationError
└── CryptoError
```

### SCBEError

Base exception with error code and details.

```python
class SCBEError(Exception):
    code: str                # Error code
    message: str             # Human-readable message
    details: Dict[str, Any]  # Additional context
    http_status: int         # HTTP status code
```

### Factory Methods

```python
from scbe_production.exceptions import ValidationError, GovernanceError

# Validation errors
raise ValidationError.invalid_position((1, 2))
raise ValidationError.missing_field("agent_id")

# Governance errors
raise GovernanceError.access_denied("High risk score", risk_score=0.85)
raise GovernanceError.snap_triggered("Critical threat detected")
```

---

## Logging

### AuditLogger

Structured logging with redaction.

```python
from scbe_production.logging import get_logger, AuditEventType

logger = get_logger()

# Set request context
logger.set_request_context(
    request_id="req-123",
    agent_id="agent-001",
)

# Log events
logger.governance_decision(
    decision="ALLOW",
    risk_score=0.15,
    reason="Low risk, trusted agent",
)

logger.geoseal_result(
    path="interior",
    distance=0.32,
    time_dilation=0.95,
)

# Clear context
logger.clear_request_context()
```

### Log Output (JSON)

```json
{
  "event_id": "uuid",
  "timestamp": "2026-01-18T19:00:00Z",
  "event_type": "governance.allow",
  "service": "scbe-production",
  "request_id": "req-123",
  "agent_id": "hash:a1b2c3d4e5f6",
  "risk_score": 0.15,
  "decision": "ALLOW"
}
```

### TimedOperation

Context manager for performance tracking.

```python
from scbe_production.logging import TimedOperation

with TimedOperation("seal_memory") as op:
    shard = service.seal_memory(...)

print(f"Operation took {op.duration_ms:.2f}ms")
```

---

## Low-Level APIs

### PQC Module

Direct access to post-quantum cryptography.

```python
from symphonic_cipher.scbe_aethermoore.pqc import PQCCore

pqc = PQCCore(kem_level=768, dsa_level=65)

# Key generation
public_key, secret_key = pqc.keygen_kem()

# Encapsulation
ciphertext, shared_secret = pqc.encapsulate(public_key)

# Decapsulation
shared_secret = pqc.decapsulate(ciphertext, secret_key)

# Signing
signing_key, verify_key = pqc.keygen_dsa()
signature = pqc.sign(message, signing_key)
valid = pqc.verify(message, signature, verify_key)
```

### GeoSeal Manifold

```python
from symphonic_cipher.scbe_aethermoore.manifold import GeoSealManifold

manifold = GeoSealManifold(interior_threshold=0.5)

# Project to dual space
sphere_point = manifold.project_to_sphere(context_vector)
hypercube_point = manifold.project_to_hypercube(policy_vector)

# Calculate geometric trust distance
distance = manifold.geometric_distance(sphere_point, hypercube_point)

# Classify path
path = manifold.classify_path(distance)  # 'interior' or 'exterior'

# Time dilation
dilation = manifold.time_dilation(distance)  # τ = exp(-γ·r)
```

### Harmonic Scaling

```python
from symphonic_cipher.harmonic_scaling_law import quantum_resistant_harmonic_scaling

# Calculate harmonic cost
cost = quantum_resistant_harmonic_scaling(
    distance=1.5,
    harmonic_ratio=1.618,  # φ
)
# cost = φ^(1.5²) = φ^2.25
```

### Dual Lattice Consensus

```python
from symphonic_cipher.dual_lattice_consensus import DualLatticeConsensus

consensus = DualLatticeConsensus()

# Verify consensus
result = consensus.verify(
    primal_data=kem_ciphertext,
    dual_data=dsa_signature,
)

print(f"Consensus: {result.state}")  # SETTLED or FAILED
print(f"Primal OK: {result.primal_verified}")
print(f"Dual OK: {result.dual_verified}")
```

---

## Type Hints

Full type annotations are available:

```python
from scbe_production.service import (
    SCBEProductionService,
    MemoryShard,
    AccessRequest,
    AccessResponse,
)
from scbe_production.config import (
    ProductionConfig,
    PQCConfig,
    GovernanceConfig,
    GeoSealConfig,
    Environment,
)
from scbe_production.exceptions import (
    SCBEError,
    ValidationError,
    PQCError,
    GovernanceError,
)
```

---

## Thread Safety

`SCBEProductionService` is thread-safe for read operations. For write operations, use appropriate synchronization:

```python
import threading

service = SCBEProductionService()
lock = threading.Lock()

def seal_with_lock(data, agent_id):
    with lock:
        return service.seal_memory(
            plaintext=data,
            agent_id=agent_id,
            topic="concurrent",
            position=(1, 2, 3, 5, 8, 13),
        )
```

---

*For more examples, see the `/tests` directory and demo files.*
