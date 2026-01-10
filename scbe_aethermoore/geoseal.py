"""
GeoSeal — Geometric Trust Manifold for SCBE (v0.1)

Treats authorization and key-derivation as GEOMETRY:
- Sphere S^n = AI mind-state manifold (behavioral/telemetry coordinates)
- Hypercube [0,1]^m = Policy & governance space (rules/tiers/actors)
- Fields of influence (risk/trust potentials) ride on both
- Interior vs Exterior paths produce different signatures/keys

The result is a geometric access kernel that plugs under SCBE's PQ core
(Kyber/Dilithium) and over its chaotic/fractal gates.

Reference: GeoSeal Specification v0.1
"""

import math
import struct
import hashlib
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import numpy as np


class PathType(Enum):
    """Geometric path classification."""
    INTERIOR = "interior"  # Fast path - within allowed cells, low potential
    EXTERIOR = "exterior"  # Slow path - outside allowed cells or high potential


@dataclass
class SphereConfig:
    """Sphere manifold configuration."""
    n_dims: int = 3           # Dimension of sphere (2=S², 3=S³)
    level: int = 4            # HEALPix-like tiling level L_s
    # Feature indices to use for sphere projection
    feature_indices: Tuple[int, ...] = (3, 4, 5)  # entropy, load, stability


@dataclass
class CubeConfig:
    """Hypercube manifold configuration."""
    m_dims: int = 4           # Dimensions of policy hypercube
    level: int = 4            # Morton code tiling level L_c
    # Policy feature names
    policy_axes: Tuple[str, ...] = ("tier", "intent", "data_class", "safety")


@dataclass
class PotentialConfig:
    """Risk/Trust potential field configuration."""
    alpha: float = 1.0        # Risk weight
    beta: float = 1.0         # Trust weight
    theta_interior: float = 0.5  # Interior threshold


@dataclass
class TimeDilationConfig:
    """Time-dilation parameters based on geometry."""
    tau_0: float = 100.0      # Base latency budget (ms)
    gamma: float = 0.5        # Dilation rate for interior
    pow_0: int = 0            # Base PoW bits
    kappa: float = 2.0        # PoW scaling for exterior
    r_0: float = 0.5          # Radius threshold for PoW


@dataclass
class GeoSealConfig:
    """Complete GeoSeal configuration."""
    sphere: SphereConfig = field(default_factory=SphereConfig)
    cube: CubeConfig = field(default_factory=CubeConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    time_dilation: TimeDilationConfig = field(default_factory=TimeDilationConfig)
    margin_epsilon: float = 0.1  # Margin for interior classification


# =============================================================================
# SPHERE PROJECTION (S^n - AI Mind-State Manifold)
# =============================================================================

def project_to_sphere(
    context: np.ndarray,
    config: SphereConfig
) -> np.ndarray:
    """
    Project context features to unit sphere S^n.

    Uses z-score normalization then L2 normalization.

    Args:
        context: Full context vector
        config: Sphere configuration

    Returns:
        Unit vector u on S^n
    """
    # Extract relevant features
    indices = list(config.feature_indices[:config.n_dims])
    features = context[indices] if len(context) > max(indices) else context[:config.n_dims]

    # Z-score normalization (assuming roughly centered data)
    # In production, use running mean/std
    mean = np.mean(features)
    std = np.std(features) + 1e-10
    z_scored = (features - mean) / std

    # L2 normalize to unit sphere
    norm = np.linalg.norm(z_scored)
    if norm < 1e-10:
        # Default to north pole if zero
        u = np.zeros(config.n_dims)
        u[0] = 1.0
        return u

    return z_scored / norm


def sphere_to_angles(u: np.ndarray) -> Tuple[float, ...]:
    """
    Convert unit vector to spherical angles.

    For S²: returns (theta, phi) where theta ∈ [0,π], phi ∈ [0,2π)
    For S³: returns (theta, phi, psi)
    """
    n = len(u)

    if n == 2:
        # S¹ (circle)
        theta = math.atan2(u[1], u[0])
        return (theta,)
    elif n == 3:
        # S² (2-sphere)
        r = np.linalg.norm(u)
        theta = math.acos(np.clip(u[2] / (r + 1e-10), -1, 1))
        phi = math.atan2(u[1], u[0])
        return (theta, phi)
    else:
        # Higher dimensions - return first 3 angles
        angles = []
        for i in range(min(3, n-1)):
            denom = np.linalg.norm(u[i:])
            if denom > 1e-10:
                angles.append(math.acos(np.clip(u[i] / denom, -1, 1)))
            else:
                angles.append(0.0)
        return tuple(angles)


def healpix_index(u: np.ndarray, level: int) -> int:
    """
    Compute HEALPix-like cell index for sphere point.

    Simplified implementation using angular binning.
    For production, use healpy library.

    Args:
        u: Unit vector on sphere
        level: Tiling level (higher = finer)

    Returns:
        Cell index h
    """
    angles = sphere_to_angles(u)
    n_cells_per_dim = 2 ** level

    # Quantize angles to cell indices
    indices = []
    for i, angle in enumerate(angles):
        # Normalize angle to [0, 1]
        if i == 0:  # theta ∈ [0, π]
            normalized = angle / math.pi
        else:  # phi ∈ [-π, π] or [0, 2π]
            normalized = (angle + math.pi) / (2 * math.pi)

        idx = int(normalized * n_cells_per_dim) % n_cells_per_dim
        indices.append(idx)

    # Combine into single index (interleaved for locality)
    h = 0
    for i, idx in enumerate(indices):
        h += idx * (n_cells_per_dim ** i)

    return h


# =============================================================================
# HYPERCUBE PROJECTION ([0,1]^m - Policy Space)
# =============================================================================

def project_to_cube(
    policy_values: Dict[str, float],
    config: CubeConfig
) -> np.ndarray:
    """
    Project policy values to hypercube [0,1]^m.

    Args:
        policy_values: Dict of policy axis -> value
        config: Cube configuration

    Returns:
        Vector v in [0,1]^m
    """
    v = np.zeros(config.m_dims)

    for i, axis in enumerate(config.policy_axes[:config.m_dims]):
        if axis in policy_values:
            # Clip to [0, 1]
            v[i] = np.clip(policy_values[axis], 0.0, 1.0)
        else:
            v[i] = 0.5  # Default to center

    return v


def morton_encode(v: np.ndarray, level: int) -> int:
    """
    Compute Morton code (Z-order) for hypercube point.

    Interleaves bits of quantized coordinates for spatial locality.

    Args:
        v: Point in [0,1]^m
        level: Tiling level

    Returns:
        Morton code z
    """
    n_cells = 2 ** level
    m = len(v)

    # Quantize to integer coordinates
    coords = [int(v[i] * (n_cells - 1)) for i in range(m)]

    # Interleave bits
    z = 0
    for bit in range(level):
        for dim in range(m):
            if coords[dim] & (1 << bit):
                z |= 1 << (bit * m + dim)

    return z


def morton_decode(z: int, m: int, level: int) -> np.ndarray:
    """Decode Morton code back to coordinates."""
    n_cells = 2 ** level
    coords = [0] * m

    for bit in range(level):
        for dim in range(m):
            if z & (1 << (bit * m + dim)):
                coords[dim] |= 1 << bit

    return np.array([c / (n_cells - 1) for c in coords])


# =============================================================================
# FIELDS OF INFLUENCE (Risk/Trust Potentials)
# =============================================================================

@dataclass
class PotentialResult:
    """Result of potential field computation."""
    risk: float
    trust: float
    potential: float  # P = α·R - β·T
    margin: float     # Distance to threshold


def compute_potentials(
    u: np.ndarray,
    v: np.ndarray,
    risk_factors: Dict[str, float],
    trust_factors: Dict[str, float],
    config: PotentialConfig
) -> PotentialResult:
    """
    Compute risk and trust potentials at geometric coordinates.

    Args:
        u: Sphere coordinates
        v: Cube coordinates
        risk_factors: Dict of risk signals (phase_skew, oracle_delta, etc.)
        trust_factors: Dict of trust signals (approvals, uptime, etc.)
        config: Potential configuration

    Returns:
        PotentialResult with combined potential
    """
    # Aggregate risk (0 = safe, higher = dangerous)
    risk_weights = {
        "phase_skew": 0.2,
        "oracle_delta": 0.2,
        "retry_count": 0.15,
        "inter_arrival_var": 0.15,
        "ledger_delta": 0.15,
        "anomaly_score": 0.15
    }

    risk = 0.0
    for factor, weight in risk_weights.items():
        if factor in risk_factors:
            risk += weight * min(1.0, risk_factors[factor])

    # Aggregate trust (0 = untrusted, 1 = fully trusted)
    trust_weights = {
        "prior_approvals": 0.25,
        "operator_attestation": 0.25,
        "uptime": 0.2,
        "relationship_age": 0.15,
        "verification_history": 0.15
    }

    trust = 0.0
    for factor, weight in trust_weights.items():
        if factor in trust_factors:
            trust += weight * min(1.0, trust_factors[factor])

    # Combined potential
    potential = config.alpha * risk - config.beta * trust
    margin = config.theta_interior - potential

    return PotentialResult(
        risk=risk,
        trust=trust,
        potential=potential,
        margin=margin
    )


# =============================================================================
# INTERIOR VS EXTERIOR CLASSIFICATION
# =============================================================================

@dataclass
class CellMembership:
    """Cell membership result."""
    h: int           # Sphere cell index
    z: int           # Cube cell index (Morton code)
    L_s: int         # Sphere level
    L_c: int         # Cube level
    in_allowed_sphere: bool
    in_allowed_cube: bool
    margin: float    # Potential margin


def check_cell_membership(
    h: int,
    z: int,
    allowed_sphere_cells: set,
    allowed_cube_cells: set,
    margin: float,
    epsilon: float
) -> Tuple[bool, bool]:
    """
    Check if cells are in allowed sets.

    Args:
        h: Sphere cell index
        z: Cube cell index
        allowed_sphere_cells: Set of allowed h values
        allowed_cube_cells: Set of allowed z values
        margin: Potential margin
        epsilon: Margin threshold

    Returns:
        Tuple of (in_sphere, in_cube)
    """
    in_sphere = h in allowed_sphere_cells if allowed_sphere_cells else True
    in_cube = z in allowed_cube_cells if allowed_cube_cells else True

    # Also check margin
    if margin < epsilon:
        in_sphere = False
        in_cube = False

    return (in_sphere, in_cube)


def classify_path(
    u: np.ndarray,
    v: np.ndarray,
    config: GeoSealConfig,
    risk_factors: Dict[str, float],
    trust_factors: Dict[str, float],
    allowed_sphere_cells: Optional[set] = None,
    allowed_cube_cells: Optional[set] = None
) -> Tuple[PathType, CellMembership, PotentialResult]:
    """
    Classify geometric path as interior or exterior.

    Args:
        u: Sphere point
        v: Cube point
        config: Full GeoSeal config
        risk_factors: Risk signals
        trust_factors: Trust signals
        allowed_sphere_cells: Optional set of allowed sphere cells
        allowed_cube_cells: Optional set of allowed cube cells

    Returns:
        Tuple of (path_type, cell_membership, potential_result)
    """
    # Compute cell indices
    h = healpix_index(u, config.sphere.level)
    z = morton_encode(v, config.cube.level)

    # Compute potentials
    potentials = compute_potentials(
        u, v, risk_factors, trust_factors, config.potential
    )

    # Check membership
    in_sphere, in_cube = check_cell_membership(
        h, z,
        allowed_sphere_cells or set(),
        allowed_cube_cells or set(),
        potentials.margin,
        config.margin_epsilon
    )

    membership = CellMembership(
        h=h,
        z=z,
        L_s=config.sphere.level,
        L_c=config.cube.level,
        in_allowed_sphere=in_sphere,
        in_allowed_cube=in_cube,
        margin=potentials.margin
    )

    # Classify path
    if in_sphere and in_cube and potentials.margin >= config.margin_epsilon:
        path = PathType.INTERIOR
    else:
        path = PathType.EXTERIOR

    return (path, membership, potentials)


# =============================================================================
# TIME DILATION
# =============================================================================

def compute_time_dilation(
    path: PathType,
    radial_distance: float,
    config: TimeDilationConfig
) -> Tuple[float, int]:
    """
    Compute time dilation parameters based on geometry.

    τ_allow = τ₀ · exp(-γ · r)  for interior
    pow_bits = pow₀ + κ · max(0, r - r₀)  for exterior

    Args:
        path: Interior or exterior
        radial_distance: Distance from center (0=center, 1=edge)
        config: Time dilation config

    Returns:
        Tuple of (latency_budget_ms, pow_bits)
    """
    if path == PathType.INTERIOR:
        # Faster inside trust wells
        latency = config.tau_0 * math.exp(-config.gamma * radial_distance)
        pow_bits = config.pow_0
    else:
        # Slower/harder outside
        latency = config.tau_0 * math.exp(config.gamma * radial_distance)
        pow_bits = config.pow_0 + int(
            config.kappa * max(0, radial_distance - config.r_0)
        )

    return (latency, pow_bits)


# =============================================================================
# CRYPTO BINDING (Domain-Separated Keys)
# =============================================================================

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract using HMAC-SHA256."""
    import hmac
    return hmac.new(salt, ikm, 'sha256').digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-Expand using HMAC-SHA256."""
    import hmac

    hash_len = 32
    n = (length + hash_len - 1) // hash_len

    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), 'sha256').digest()
        okm += t

    return okm[:length]


def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """Full HKDF using HMAC-SHA256."""
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)


def derive_region_keys(
    shared_secret: bytes,
    membership: CellMembership,
    path: PathType
) -> Dict[str, bytes]:
    """
    Derive region-specific keys from shared secret.

    K_sphere = HKDF(ss, "geo:sphere|h|L_s")
    K_cube   = HKDF(ss, "geo:cube|z|L_c")
    K_msg    = HKDF(K_sphere ⊕ K_cube, "geo:msg")

    Args:
        shared_secret: Kyber shared secret
        membership: Cell membership info
        path: Interior or exterior

    Returns:
        Dict with K_sphere, K_cube, K_msg
    """
    # Sphere key
    info_sphere = f"geo:sphere|{membership.h}|{membership.L_s}".encode()
    K_sphere = hkdf(shared_secret, b"geoseal", info_sphere)

    # Cube key
    info_cube = f"geo:cube|{membership.z}|{membership.L_c}".encode()
    K_cube = hkdf(shared_secret, b"geoseal", info_cube)

    # Combined message key
    K_combined = bytes(a ^ b for a, b in zip(K_sphere, K_cube))
    info_msg = f"geo:msg|{path.value}".encode()
    K_msg = hkdf(K_combined, b"geoseal", info_msg)

    return {
        "K_sphere": K_sphere,
        "K_cube": K_cube,
        "K_msg": K_msg
    }


# =============================================================================
# ATTESTATION
# =============================================================================

@dataclass
class GeoAttestation:
    """Geometric attestation for signing."""
    h: int
    z: int
    L_s: int
    L_c: int
    P: float
    margin: float
    timestamp: float
    path: str

    def to_bytes(self) -> bytes:
        """Serialize attestation for signing."""
        data = struct.pack(
            '<IIIIffd',
            self.h, self.z, self.L_s, self.L_c,
            self.P, self.margin, self.timestamp
        )
        return data + self.path.encode()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "h": self.h,
            "z": self.z,
            "L_s": self.L_s,
            "L_c": self.L_c,
            "P": self.P,
            "margin": self.margin,
            "timestamp": self.timestamp,
            "path": self.path
        }


def create_attestation(
    membership: CellMembership,
    potentials: PotentialResult,
    path: PathType,
    timestamp: float
) -> GeoAttestation:
    """Create geometric attestation."""
    return GeoAttestation(
        h=membership.h,
        z=membership.z,
        L_s=membership.L_s,
        L_c=membership.L_c,
        P=potentials.potential,
        margin=potentials.margin,
        timestamp=timestamp,
        path=path.value
    )


def sign_attestation(
    attestation: GeoAttestation,
    payload_hash: bytes,
    signing_key: bytes
) -> bytes:
    """
    Sign attestation with payload.

    In production, use Dilithium. Here we use HMAC for demo.
    """
    import hmac

    message = attestation.to_bytes() + payload_hash
    return hmac.new(signing_key, message, 'sha256').digest()


def verify_attestation(
    attestation: GeoAttestation,
    payload_hash: bytes,
    signature: bytes,
    verification_key: bytes
) -> bool:
    """Verify attestation signature."""
    expected = sign_attestation(attestation, payload_hash, verification_key)
    return signature == expected


# =============================================================================
# COMPLETE GEOSEAL OPERATIONS
# =============================================================================

@dataclass
class GeoSealEnvelope:
    """Complete GeoSeal encrypted envelope."""
    ct_kem: bytes           # Kyber ciphertext (simulated as hash)
    ct_spectral: bytes      # Chaos-encrypted payload
    attestation: GeoAttestation
    signature: bytes
    path: PathType


def geoseal_encrypt(
    plaintext: bytes,
    context: np.ndarray,
    policy: Dict[str, float],
    risk_factors: Dict[str, float],
    trust_factors: Dict[str, float],
    master_key: bytes,
    signing_key: bytes,
    config: GeoSealConfig = GeoSealConfig(),
    allowed_sphere_cells: Optional[set] = None,
    allowed_cube_cells: Optional[set] = None
) -> GeoSealEnvelope:
    """
    GeoSeal encryption with geometric binding.

    Args:
        plaintext: Data to encrypt
        context: Runtime context vector
        policy: Policy values for cube projection
        risk_factors: Risk signals
        trust_factors: Trust signals
        master_key: Master encryption key
        signing_key: Signing key
        config: GeoSeal configuration
        allowed_sphere_cells: Allowed sphere cells
        allowed_cube_cells: Allowed cube cells

    Returns:
        GeoSealEnvelope
    """
    import time

    # Project to geometric spaces
    u = project_to_sphere(context, config.sphere)
    v = project_to_cube(policy, config.cube)

    # Classify path
    path, membership, potentials = classify_path(
        u, v, config, risk_factors, trust_factors,
        allowed_sphere_cells, allowed_cube_cells
    )

    # Simulate Kyber encapsulation
    # In production: ss, ct_kem = Kyber.encaps(pk)
    ct_kem = hashlib.sha256(master_key + b"kem").digest()
    ss = hashlib.sha256(master_key + ct_kem).digest()

    # Derive region keys
    keys = derive_region_keys(ss, membership, path)

    # Chaos diffusion encryption
    from .chaos import chaos_diffusion
    from .context import ContextVector

    # Derive chaos params from geometric context
    ctx_for_chaos = ContextVector(
        time=context[0] if len(context) > 0 else 0,
        device_id=context[1] if len(context) > 1 else 0,
        threat_level=context[2] if len(context) > 2 else 0,
        entropy=context[3] if len(context) > 3 else 0.5,
        server_load=context[4] if len(context) > 4 else 0.5,
        behavior_stability=context[5] if len(context) > 5 else 0.9
    )

    # Use path-dependent seed for chaos
    chaos_seed = hashlib.sha256(
        keys["K_msg"] + path.value.encode()
    ).digest()

    # Derive chaos parameters
    r = 3.97 + (chaos_seed[0] / 255) * 0.029  # [3.97, 3.999]
    x0 = 0.1 + (chaos_seed[1] / 255) * 0.8    # [0.1, 0.9]

    ct_spectral = chaos_diffusion(plaintext, r, x0, security_dimension=6)

    # Create attestation
    attestation = create_attestation(
        membership, potentials, path, time.time()
    )

    # Sign
    payload_hash = hashlib.sha256(ct_spectral).digest()
    signature = sign_attestation(attestation, payload_hash, signing_key)

    return GeoSealEnvelope(
        ct_kem=ct_kem,
        ct_spectral=ct_spectral,
        attestation=attestation,
        signature=signature,
        path=path
    )


def geoseal_decrypt(
    envelope: GeoSealEnvelope,
    context: np.ndarray,
    master_key: bytes,
    verification_key: bytes,
    config: GeoSealConfig = GeoSealConfig()
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    GeoSeal decryption with verification.

    Args:
        envelope: Encrypted envelope
        context: Current context (must match encryption context)
        master_key: Master key
        verification_key: Signature verification key
        config: GeoSeal configuration

    Returns:
        Tuple of (plaintext or None, status dict)
    """
    # Verify signature
    payload_hash = hashlib.sha256(envelope.ct_spectral).digest()
    if not verify_attestation(
        envelope.attestation, payload_hash,
        envelope.signature, verification_key
    ):
        return (None, {"error": "Signature verification failed"})

    # Reconstruct shared secret
    ss = hashlib.sha256(master_key + envelope.ct_kem).digest()

    # Reconstruct membership from attestation
    membership = CellMembership(
        h=envelope.attestation.h,
        z=envelope.attestation.z,
        L_s=envelope.attestation.L_s,
        L_c=envelope.attestation.L_c,
        in_allowed_sphere=True,
        in_allowed_cube=True,
        margin=envelope.attestation.margin
    )

    # Derive keys
    path = PathType(envelope.attestation.path)
    keys = derive_region_keys(ss, membership, path)

    # Chaos decryption
    from .chaos import chaos_undiffusion

    chaos_seed = hashlib.sha256(
        keys["K_msg"] + path.value.encode()
    ).digest()

    r = 3.97 + (chaos_seed[0] / 255) * 0.029
    x0 = 0.1 + (chaos_seed[1] / 255) * 0.8

    plaintext = chaos_undiffusion(
        envelope.ct_spectral, r, x0, security_dimension=6
    )

    return (plaintext, {
        "path": path.value,
        "h": membership.h,
        "z": membership.z,
        "verified": True
    })


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def visualize_geometry(
    u: np.ndarray,
    v: np.ndarray,
    path: PathType,
    membership: CellMembership
) -> str:
    """Generate ASCII visualization of geometric state."""
    lines = []
    lines.append("┌─────────────────────────────────────────┐")
    lines.append("│         GEOSEAL GEOMETRY                │")
    lines.append("├─────────────────────────────────────────┤")

    # Sphere visualization (simplified 2D projection)
    sphere_char = "●" if path == PathType.INTERIOR else "○"
    angles = sphere_to_angles(u)
    lines.append(f"│ Sphere S^{len(u)-1}:                              │")
    lines.append(f"│   Point: {sphere_char} ({u[0]:.2f}, {u[1]:.2f}, {u[2] if len(u)>2 else 0:.2f})  │")
    lines.append(f"│   Cell h={membership.h}, Level={membership.L_s}           │")

    # Cube visualization
    cube_char = "■" if path == PathType.INTERIOR else "□"
    lines.append(f"│ Cube [0,1]^{len(v)}:                            │")
    lines.append(f"│   Point: {cube_char} ({v[0]:.2f}, {v[1]:.2f}, ...)       │")
    lines.append(f"│   Morton z={membership.z}, Level={membership.L_c}        │")

    # Path
    path_symbol = "→→→" if path == PathType.INTERIOR else "⇢⇢⇢"
    lines.append(f"│ Path: {path.value.upper()} {path_symbol}              │")
    lines.append(f"│ Margin: {membership.margin:.4f}                      │")
    lines.append("└─────────────────────────────────────────┘")

    return "\n".join(lines)
