"""
Topology-Gated Two-Channel KEM Handshake

Uses geometric manifold intersection (Sphere S² ∩ Torus T²) to derive an
unforgeable control bit for policy enforcement.

Core Concept:
- Two parallel Kyber channels: Brain (inside) and Steward (outside)
- Map public transcripts to manifold coordinates via XOF
- Sphere point u ∈ S² represents "brain's point"
- Torus point τ ∈ T² represents "steward's point"
- ON_SPHERE: τ intersects unit sphere AND u ≈ τ̂
- OFF_SPHERE: otherwise → requires Tier-3 elevation

This creates a signed control bit that neither party can forge alone.

Reference: SCBE-AETHER-UNIFIED-2026-001 Extension
"""

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List
import numpy as np


class ManifoldLabel(Enum):
    """Manifold intersection classification."""
    ON_SPHERE = "ON_SPHERE"      # Inside the brain - fast path
    OFF_SPHERE = "OFF_SPHERE"    # Outside the box - requires Roundtable


@dataclass
class TorusParams:
    """Torus geometry parameters."""
    R: float = 1.0    # Major radius
    r: float = 0.25   # Minor radius (choose so torus sometimes kisses unit sphere)


@dataclass
class ManifoldThresholds:
    """Intersection detection thresholds."""
    epsilon: float = 1e-3  # Radius error tolerance: | ||τ|| - 1 | ≤ ε
    delta: float = 1e-2    # Angular proximity: ||u - τ̂|| ≤ δ


@dataclass
class ManifoldResult:
    """Result of manifold classification."""
    label: ManifoldLabel
    manifold_tag: bytes
    sphere_point: Tuple[float, float, float]      # u ∈ S²
    torus_point: Tuple[float, float, float]       # τ ∈ T²
    torus_angles: Tuple[float, float]             # (α, β)
    sphere_delta: float                           # ||u - τ̂||
    radius_error: float                           # | ||τ|| - 1 |
    torus_params: TorusParams


def xof_expand(seed: bytes, length: int) -> bytes:
    """
    Extensible Output Function using SHAKE256.

    Args:
        seed: Input seed bytes
        length: Number of output bytes

    Returns:
        Expanded bytes
    """
    return hashlib.shake_256(seed).digest(length)


def bytes_to_float3(data: bytes) -> Tuple[float, float, float]:
    """
    Convert 24 bytes to 3 float64 values in [-1, 1].

    Args:
        data: At least 24 bytes

    Returns:
        Tuple of 3 floats normalized to [-1, 1]
    """
    if len(data) < 24:
        raise ValueError(f"Need at least 24 bytes, got {len(data)}")

    # Unpack as 3 unsigned 64-bit integers
    v1, v2, v3 = struct.unpack('<QQQ', data[:24])

    # Normalize to [-1, 1]
    max_val = 2**64 - 1
    x = 2 * (v1 / max_val) - 1
    y = 2 * (v2 / max_val) - 1
    z = 2 * (v3 / max_val) - 1

    return (x, y, z)


def normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector to unit length."""
    x, y, z = v
    norm = math.sqrt(x*x + y*y + z*z)
    if norm < 1e-10:
        return (1.0, 0.0, 0.0)  # Default if near-zero
    return (x/norm, y/norm, z/norm)


def bytes_to_angles(data: bytes) -> Tuple[float, float]:
    """
    Convert 16 bytes to 2 angles in [0, 2π).

    Args:
        data: At least 16 bytes

    Returns:
        Tuple of (α, β) angles
    """
    if len(data) < 16:
        raise ValueError(f"Need at least 16 bytes, got {len(data)}")

    v1, v2 = struct.unpack('<QQ', data[:16])
    max_val = 2**64 - 1

    alpha = 2 * math.pi * (v1 / max_val)
    beta = 2 * math.pi * (v2 / max_val)

    return (alpha, beta)


def torus_point(
    alpha: float,
    beta: float,
    params: TorusParams
) -> Tuple[float, float, float]:
    """
    Compute a point on the torus T².

    τ(α, β) = ((R + r·cos(α))·cos(β), (R + r·cos(α))·sin(β), r·sin(α))

    Args:
        alpha: Angle around minor circle
        beta: Angle around major circle
        params: Torus geometry parameters

    Returns:
        3D point on torus surface
    """
    R, r = params.R, params.r

    x = (R + r * math.cos(alpha)) * math.cos(beta)
    y = (R + r * math.cos(alpha)) * math.sin(beta)
    z = r * math.sin(alpha)

    return (x, y, z)


def torus_radius_squared(alpha: float, params: TorusParams) -> float:
    """
    Compute ||τ||² for a given α.

    ||τ||² = R² + 2Rr·cos(α) + r²
    """
    R, r = params.R, params.r
    return R*R + 2*R*r*math.cos(alpha) + r*r


def can_intersect_sphere(params: TorusParams) -> Tuple[bool, Optional[float]]:
    """
    Check if torus can intersect the unit sphere.

    Intersection when |cos(α*)| ≤ 1 where:
    cos(α*) = (1 - R² - r²) / (2Rr)

    Args:
        params: Torus parameters

    Returns:
        Tuple of (can_intersect, cos_alpha_star)
    """
    R, r = params.R, params.r
    if R * r == 0:
        return (False, None)

    cos_alpha_star = (1 - R*R - r*r) / (2 * R * r)
    can_intersect = abs(cos_alpha_star) <= 1

    return (can_intersect, cos_alpha_star)


def derive_sphere_point(
    ctx: bytes,
    ct_out: bytes,
    pk_out: bytes
) -> Tuple[float, float, float]:
    """
    Derive sphere point u ∈ S² from OUT transcript.

    u = normalize(H(ct_out || pk_out || ctx)[:24] as float3)

    The point is derived from the steward's transcript so both
    parties contribute to the final position.
    """
    seed = ct_out + pk_out + ctx
    expanded = xof_expand(seed, 24)
    raw = bytes_to_float3(expanded)
    return normalize(raw)


def derive_torus_point(
    ctx: bytes,
    ct_in: bytes,
    pk_in: bytes,
    params: TorusParams
) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
    """
    Derive torus point τ ∈ T² from IN transcript.

    (α, β) = angles_from(H(ct_in || pk_in || ctx))
    τ = torus(R, r, α, β)

    Returns:
        Tuple of (torus_point, (alpha, beta))
    """
    seed = ct_in + pk_in + ctx
    expanded = xof_expand(seed, 16)
    alpha, beta = bytes_to_angles(expanded)
    tau = torus_point(alpha, beta, params)
    return (tau, (alpha, beta))


def vector_distance(
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float]
) -> float:
    """Compute Euclidean distance between two 3D vectors."""
    dx = v1[0] - v2[0]
    dy = v1[1] - v2[1]
    dz = v1[2] - v2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def vector_norm(v: Tuple[float, float, float]) -> float:
    """Compute norm of a 3D vector."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def classify_manifold(
    ctx: bytes,
    pk_in: bytes,
    ct_in: bytes,
    pk_out: bytes,
    ct_out: bytes,
    params: TorusParams = TorusParams(),
    thresholds: ManifoldThresholds = ManifoldThresholds()
) -> ManifoldResult:
    """
    Classify the manifold intersection to derive the control bit.

    Procedure:
    1. u = unit(float3(XOF(ctx||ct_out||pk_out)))
    2. (α, β) = angles(XOF(ctx||ct_in||pk_in))
    3. τ = torus(R, r, α, β); ρ = ||τ||; τ̂ = τ/ρ
    4. inside = (|ρ - 1| ≤ ε) ∧ (||u - τ̂|| ≤ δ)
    5. label = inside ? ON_SPHERE : OFF_SPHERE
    6. manifold_tag = H(ctx || pk_in || ct_in || pk_out || ct_out || u || τ || label)

    Args:
        ctx: Context hash (your 6D SCBE context)
        pk_in: Brain channel public key
        ct_in: Brain channel ciphertext
        pk_out: Steward channel public key
        ct_out: Steward channel ciphertext
        params: Torus geometry parameters
        thresholds: Intersection detection thresholds

    Returns:
        ManifoldResult with classification and cryptographic tag
    """
    # Step 1: Derive sphere point from OUT transcript
    u = derive_sphere_point(ctx, ct_out, pk_out)

    # Step 2-3: Derive torus point from IN transcript
    tau, (alpha, beta) = derive_torus_point(ctx, ct_in, pk_in, params)

    # Compute torus radius and normalized point
    rho = vector_norm(tau)
    if rho > 1e-10:
        tau_hat = (tau[0]/rho, tau[1]/rho, tau[2]/rho)
    else:
        tau_hat = (1.0, 0.0, 0.0)

    # Step 4: Check intersection conditions
    radius_error = abs(rho - 1.0)
    sphere_delta = vector_distance(u, tau_hat)

    inside = (radius_error <= thresholds.epsilon) and (sphere_delta <= thresholds.delta)

    # Step 5: Assign label
    label = ManifoldLabel.ON_SPHERE if inside else ManifoldLabel.OFF_SPHERE

    # Step 6: Compute manifold tag
    # Pack geometric data for hashing
    u_bytes = struct.pack('<3d', *u)
    tau_bytes = struct.pack('<3d', *tau)
    label_bytes = label.value.encode('utf-8')

    tag_input = (
        ctx + pk_in + ct_in + pk_out + ct_out +
        u_bytes + tau_bytes + label_bytes
    )
    manifold_tag = hashlib.sha256(tag_input).digest()

    return ManifoldResult(
        label=label,
        manifold_tag=manifold_tag,
        sphere_point=u,
        torus_point=tau,
        torus_angles=(alpha, beta),
        sphere_delta=sphere_delta,
        radius_error=radius_error,
        torus_params=params
    )


@dataclass
class SignedManifoldResult:
    """Manifold result with cryptographic signatures."""
    result: ManifoldResult
    sig_manager: bytes
    sig_steward: Optional[bytes] = None


def sign_manifold_result(
    result: ManifoldResult,
    sk_manager: bytes,
    sk_steward: Optional[bytes] = None
) -> SignedManifoldResult:
    """
    Sign the manifold classification.

    For OFF_SPHERE, steward signature is required.

    Args:
        result: Classification result
        sk_manager: Manager's signing key (simulated as HMAC key here)
        sk_steward: Steward's signing key (required if OFF_SPHERE)

    Returns:
        SignedManifoldResult with signatures

    Note:
        In production, use Dilithium.sign() instead of HMAC.
        This is a simplified demonstration.
    """
    # Simulate Dilithium signature with HMAC (for demo purposes)
    import hmac

    sig_manager = hmac.new(sk_manager, result.manifold_tag, 'sha256').digest()

    sig_steward = None
    if result.label == ManifoldLabel.OFF_SPHERE:
        if sk_steward is None:
            raise ValueError("OFF_SPHERE requires steward signature")
        sig_steward = hmac.new(sk_steward, result.manifold_tag, 'sha256').digest()
    elif sk_steward is not None:
        # Optional steward signature even for ON_SPHERE
        sig_steward = hmac.new(sk_steward, result.manifold_tag, 'sha256').digest()

    return SignedManifoldResult(
        result=result,
        sig_manager=sig_manager,
        sig_steward=sig_steward
    )


class TierPolicy(Enum):
    """Roundtable tier classification."""
    TIER_1 = 1  # Manager only
    TIER_2 = 2  # Manager + optional Steward
    TIER_3 = 3  # Full Roundtable (KO + RU + UM)


def determine_tier(result: ManifoldResult, risk_score: float) -> TierPolicy:
    """
    Determine Roundtable tier based on manifold label and risk.

    Policy:
    - ON_SPHERE + high risk (≥0.80): Tier-1 (allow)
    - ON_SPHERE + medium risk: Tier-2 (review)
    - OFF_SPHERE: Always Tier-3+ (require full Roundtable)
    - Low risk (<0.50): Deny

    Args:
        result: Manifold classification
        risk_score: Risk assessment (0-1)

    Returns:
        Appropriate tier level
    """
    if result.label == ManifoldLabel.OFF_SPHERE:
        return TierPolicy.TIER_3

    if risk_score >= 0.80:
        return TierPolicy.TIER_1
    elif risk_score >= 0.50:
        return TierPolicy.TIER_2
    else:
        return TierPolicy.TIER_3  # Deny → escalate to full review


def gate_scbe_context(
    ctx: bytes,
    result: ManifoldResult
) -> bytes:
    """
    Gate SCBE context with manifold label.

    Feeds label and manifold_tag into context seed so wrong label
    causes chaos-diffusion to fail noisily.

    Args:
        ctx: Original SCBE context
        result: Manifold classification

    Returns:
        Gated context for SCBE encryption
    """
    # Combine original context with manifold binding
    gated = hashlib.sha256(
        ctx + result.manifold_tag + result.label.value.encode()
    ).digest()
    return gated


def analyze_label_distribution(
    num_samples: int = 10000,
    params: TorusParams = TorusParams(),
    thresholds: ManifoldThresholds = ManifoldThresholds()
) -> dict:
    """
    Analyze the distribution of ON_SPHERE vs OFF_SPHERE labels.

    Used to tune (R, r, ε, δ) for desired prior probability.

    Args:
        num_samples: Number of random samples
        params: Torus parameters
        thresholds: Detection thresholds

    Returns:
        Statistics dictionary
    """
    import os

    on_sphere_count = 0
    radius_errors = []
    sphere_deltas = []

    for _ in range(num_samples):
        # Generate random transcripts
        ctx = os.urandom(32)
        pk_in = os.urandom(32)
        ct_in = os.urandom(32)
        pk_out = os.urandom(32)
        ct_out = os.urandom(32)

        result = classify_manifold(
            ctx, pk_in, ct_in, pk_out, ct_out,
            params, thresholds
        )

        if result.label == ManifoldLabel.ON_SPHERE:
            on_sphere_count += 1

        radius_errors.append(result.radius_error)
        sphere_deltas.append(result.sphere_delta)

    on_sphere_ratio = on_sphere_count / num_samples

    return {
        "samples": num_samples,
        "on_sphere_count": on_sphere_count,
        "off_sphere_count": num_samples - on_sphere_count,
        "on_sphere_ratio": on_sphere_ratio,
        "mean_radius_error": sum(radius_errors) / len(radius_errors),
        "mean_sphere_delta": sum(sphere_deltas) / len(sphere_deltas),
        "min_radius_error": min(radius_errors),
        "max_radius_error": max(radius_errors),
        "params": {"R": params.R, "r": params.r},
        "thresholds": {"epsilon": thresholds.epsilon, "delta": thresholds.delta}
    }


def tune_parameters_for_ratio(
    target_ratio: float = 0.5,
    tolerance: float = 0.05,
    samples_per_test: int = 1000
) -> Tuple[TorusParams, ManifoldThresholds]:
    """
    Find parameters that achieve target ON_SPHERE ratio.

    Args:
        target_ratio: Desired P(ON_SPHERE)
        tolerance: Acceptable deviation
        samples_per_test: Samples per parameter set

    Returns:
        Tuned (params, thresholds)
    """
    # Search over reasonable parameter ranges
    best_params = TorusParams()
    best_thresholds = ManifoldThresholds()
    best_error = float('inf')

    for R in [0.9, 1.0, 1.1]:
        for r in [0.1, 0.2, 0.25, 0.3]:
            for eps in [1e-3, 5e-3, 1e-2]:
                for delta in [1e-2, 5e-2, 1e-1]:
                    params = TorusParams(R=R, r=r)
                    thresholds = ManifoldThresholds(epsilon=eps, delta=delta)

                    stats = analyze_label_distribution(
                        samples_per_test, params, thresholds
                    )

                    error = abs(stats["on_sphere_ratio"] - target_ratio)
                    if error < best_error:
                        best_error = error
                        best_params = params
                        best_thresholds = thresholds

                        if error <= tolerance:
                            return (best_params, best_thresholds)

    return (best_params, best_thresholds)


# Telemetry structure for logging
@dataclass
class ManifoldTelemetry:
    """Telemetry fields for manifold classification."""
    manifold_label: str
    sphere_delta: float
    radius_error: float
    R: float
    r: float
    alpha: float
    beta: float
    tier: int
    manifold_tag_hex: str


def create_telemetry(
    result: ManifoldResult,
    tier: TierPolicy
) -> ManifoldTelemetry:
    """Create telemetry record for logging."""
    return ManifoldTelemetry(
        manifold_label=result.label.value,
        sphere_delta=result.sphere_delta,
        radius_error=result.radius_error,
        R=result.torus_params.R,
        r=result.torus_params.r,
        alpha=result.torus_angles[0],
        beta=result.torus_angles[1],
        tier=tier.value,
        manifold_tag_hex=result.manifold_tag.hex()
    )
