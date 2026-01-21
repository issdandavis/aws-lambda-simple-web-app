"""
Harmonic Scaling Law Implementation

The core mathematical foundation shared by SCBE and AETHERMOORE:
    H(d, R) = R^(d²)

This provides super-exponential growth for security scaling.

Reference: Section 0.1 of SCBE-AETHER-UNIFIED-2026-001
Claims: 18, 51, 53
"""

import math
from typing import Tuple, List, Optional
from .constants import PERFECT_FIFTH, HARMONIC_METRIC_TENSOR


def harmonic_scaling(d: int, R: float = PERFECT_FIFTH) -> float:
    """
    Compute the Harmonic Scaling Law: H(d, R) = R^(d²)

    This is the core formula shared by SCBE (Claim 18) and AETHERMOORE (Claim 51).

    Args:
        d: Dimension/depth parameter (positive integer, typically 1-7)
        R: Harmonic ratio (default: 1.5, the Perfect Fifth)

    Returns:
        The harmonic scaling factor R^(d²)

    Properties:
        - Super-exponential growth: O(R^(d²)) >> O(e^d)
        - Dimensional separability: H(d₁+d₂, R) includes cross-term R^(2d₁d₂)
        - Inverse duality: H(d, R) × H(d, 1/R) = 1

    Example:
        >>> harmonic_scaling(6, 1.5)
        2184164.409...  # 1.5^36
    """
    if d < 1:
        raise ValueError(f"Dimension d must be positive, got {d}")
    if R <= 0:
        raise ValueError(f"Ratio R must be positive, got {R}")

    return R ** (d ** 2)


def security_bits(d: int, base_bits: int = 128, R: float = PERFECT_FIFTH) -> float:
    """
    Calculate effective security bits with harmonic scaling.

    The harmonic scaling adds log₂(H(d, R)) bits of security.

    Args:
        d: Security dimension (1-7)
        base_bits: Base cryptographic strength (default: AES-128)
        R: Harmonic ratio (default: 1.5)

    Returns:
        Total effective security bits

    Reference: Section 0.1 Validation Table

    Example:
        >>> security_bits(6)
        149.06  # AES-149 equivalent
    """
    H = harmonic_scaling(d, R)
    added_bits = math.log2(H)
    return base_bits + added_bits


def harmonic_scaling_table(max_d: int = 7, R: float = PERFECT_FIFTH) -> List[dict]:
    """
    Generate the harmonic security scaling table.

    Args:
        max_d: Maximum dimension to compute
        R: Harmonic ratio

    Returns:
        List of dicts with d, d², H(d,R), security bits added, total effective

    Reference: Section 0.1 Validation Table
    """
    results = []
    for d in range(1, max_d + 1):
        H = harmonic_scaling(d, R)
        added = math.log2(H)
        results.append({
            "d": d,
            "d_squared": d ** 2,
            "H": H,
            "bits_added": added,
            "total_effective": 128 + added,
            "aes_equivalent": f"AES-{int(128 + added)}"
        })
    return results


def harmonic_metric_distance(
    c1: Tuple[float, ...],
    c2: Tuple[float, ...],
    metric: Tuple[float, ...] = HARMONIC_METRIC_TENSOR
) -> float:
    """
    Compute distance using the 6D harmonic metric tensor.

    The metric tensor g = diag(1, 1, 1, R₅, R₅², R₅³) weights the
    security dimension (index 5) 3.375× more than position (index 0).

    D_H = √(Σ gᵢᵢ × Δcᵢ²)

    Args:
        c1: First context vector (6 components)
        c2: Second context vector (6 components)
        metric: Diagonal metric tensor (default: harmonic metric)

    Returns:
        Harmonic distance between contexts

    Reference: Section 0.5 6D Vector Space Isomorphism, Claim 53
    """
    if len(c1) != len(c2):
        raise ValueError(f"Vectors must have same length: {len(c1)} vs {len(c2)}")
    if len(c1) != len(metric):
        raise ValueError(f"Vector length {len(c1)} doesn't match metric {len(metric)}")

    squared_sum = 0.0
    for i in range(len(c1)):
        delta = c1[i] - c2[i]
        squared_sum += metric[i] * (delta ** 2)

    return math.sqrt(squared_sum)


def dimensional_separability(d1: int, d2: int, R: float = PERFECT_FIFTH) -> dict:
    """
    Demonstrate dimensional separability property.

    H(d₁+d₂, R) includes cross-term R^(2d₁d₂)

    Args:
        d1, d2: Two dimensions to combine
        R: Harmonic ratio

    Returns:
        Dict showing the separability computation
    """
    combined_d = d1 + d2
    combined_H = harmonic_scaling(combined_d, R)

    # H(d1+d2) = R^((d1+d2)²) = R^(d1² + 2d1d2 + d2²)
    # = R^(d1²) × R^(2d1d2) × R^(d2²)
    # = H(d1) × R^(2d1d2) × H(d2)
    H1 = harmonic_scaling(d1, R)
    H2 = harmonic_scaling(d2, R)
    cross_term = R ** (2 * d1 * d2)

    return {
        "d1": d1,
        "d2": d2,
        "combined_d": combined_d,
        "H_combined": combined_H,
        "H_d1": H1,
        "H_d2": H2,
        "cross_term": cross_term,
        "product": H1 * cross_term * H2,
        "verification": abs(combined_H - H1 * cross_term * H2) < 1e-10
    }


def inverse_duality(d: int, R: float = PERFECT_FIFTH) -> dict:
    """
    Verify the inverse duality property: H(d, R) × H(d, 1/R) = 1

    Args:
        d: Dimension
        R: Harmonic ratio

    Returns:
        Dict showing the duality verification
    """
    H_R = harmonic_scaling(d, R)
    H_inv_R = harmonic_scaling(d, 1 / R)
    product = H_R * H_inv_R

    return {
        "d": d,
        "R": R,
        "H(d, R)": H_R,
        "H(d, 1/R)": H_inv_R,
        "product": product,
        "verification": abs(product - 1.0) < 1e-10
    }


def chaos_iterations(d: int, base_iterations: int = 50, R: float = PERFECT_FIFTH) -> int:
    """
    Calculate chaos diffusion iterations scaled by harmonic depth.

    iterations = base × H(d, R)^(1/3)

    This creates exponentially more chaos at higher security levels.

    Args:
        d: Security dimension
        base_iterations: Base number of iterations (default: 50)
        R: Harmonic ratio

    Returns:
        Number of chaos iterations for the given dimension

    Reference: Section 1.4 AETHERMOORE Enhancement
    """
    H = harmonic_scaling(d, R)
    return int(base_iterations * (H ** (1/3)))


def generate_validation_table() -> str:
    """Generate the validation table from Section 0.1."""
    lines = [
        "| d | d² | H(d, 1.5) | Security Bits Added | Total Effective |",
        "|---|----|-----------:|--------------------:|----------------:|",
    ]

    for d in range(1, 8):
        H = harmonic_scaling(d)
        bits_added = math.log2(H)
        total = 128 + bits_added

        if H < 1000:
            h_str = f"{H:.2f}"
        elif H < 1e6:
            h_str = f"{H:,.2f}"
        else:
            h_str = f"{H:.2e}"

        lines.append(
            f"| {d} | {d**2} | {h_str} | {bits_added:.2f} | AES-{int(total)} |"
        )

    return "\n".join(lines)
