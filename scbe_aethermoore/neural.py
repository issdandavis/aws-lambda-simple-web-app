"""
Neural Defense Module (AXIS 2)

Implements behavioral authorization using Hopfield networks and HAL-Attention.
Learns what "normal" behavior looks like; abnormal = high energy = rejected.

Reference: Section AXIS 2 of SCBE-AETHER-UNIFIED-2026-001
Claims: 10, 14, 59
"""

import math
from typing import List, Tuple, Optional, Union
import numpy as np
from .constants import PERFECT_FIFTH, PARAMETER_RANGES
from .harmonic import harmonic_scaling


def hopfield_energy(
    c: np.ndarray,
    W: np.ndarray,
    theta: np.ndarray
) -> float:
    """
    Compute standard Hopfield network energy.

    E(c) = -½ cᵀWc + θᵀc

    Lower energy = more stable/valid pattern.
    Higher energy = abnormal/rejected.

    Args:
        c: Context/state vector (N,)
        W: Weight matrix (N, N) - learned from valid patterns
        theta: Bias vector (N,)

    Returns:
        Energy value

    Reference: Section 2.1
    Claim: 10
    """
    c = np.asarray(c)
    W = np.asarray(W)
    theta = np.asarray(theta)

    quadratic = -0.5 * c @ W @ c
    linear = theta @ c

    return float(quadratic + linear)


def hal_energy(
    c: np.ndarray,
    W: np.ndarray,
    theta: np.ndarray,
    d: int = 6,
    R: float = PERFECT_FIFTH
) -> float:
    """
    Compute HAL-Attention energy (Harmonic Attention Layer).

    E_H(c) = -½ cᵀWc / H(d, R₅) + θᵀc

    The harmonic scaling creates tighter tolerances at higher security levels.

    At d=1: Energy nearly unchanged
    At d=6: Energy divided by 2.18M (much tighter tolerance)

    Args:
        c: Context/state vector
        W: Weight matrix
        theta: Bias vector
        d: Security dimension (1-7)
        R: Harmonic ratio

    Returns:
        Harmonic-scaled energy

    Reference: Section 0.7 HAL-Attention Integration
    Claim: 59
    """
    H = harmonic_scaling(d, R)

    c = np.asarray(c)
    W = np.asarray(W)
    theta = np.asarray(theta)

    quadratic = -0.5 * (c @ W @ c) / H
    linear = theta @ c

    return float(quadratic + linear)


def learn_hopfield_weights(
    patterns: List[np.ndarray],
    normalize: bool = True
) -> np.ndarray:
    """
    Learn Hopfield weight matrix from valid patterns.

    Uses Hebbian learning: W = (1/P) Σ pᵢpᵢᵀ

    Args:
        patterns: List of valid pattern vectors
        normalize: Whether to normalize weights

    Returns:
        Weight matrix W
    """
    if not patterns:
        raise ValueError("Need at least one pattern")

    N = len(patterns[0])
    W = np.zeros((N, N))

    for p in patterns:
        p = np.asarray(p)
        W += np.outer(p, p)

    if normalize:
        W /= len(patterns)

    # Zero diagonal (Hopfield constraint)
    np.fill_diagonal(W, 0)

    return W


def gradient_margin(
    c: np.ndarray,
    W: np.ndarray,
    theta: np.ndarray,
    epsilon: float = 0.01
) -> float:
    """
    Compute adversarial gradient margin.

    Measures how much the energy changes with small perturbations.
    Small margin = vulnerable to adversarial attack.
    Large margin = robust.

    Args:
        c: Context vector
        W: Weight matrix
        theta: Bias vector
        epsilon: Perturbation size

    Returns:
        Gradient margin (larger = more robust)

    Claim: 14
    """
    c = np.asarray(c, dtype=float)
    base_energy = hopfield_energy(c, W, theta)

    max_gradient = 0.0
    for i in range(len(c)):
        # Perturb each dimension
        c_plus = c.copy()
        c_plus[i] += epsilon

        c_minus = c.copy()
        c_minus[i] -= epsilon

        e_plus = hopfield_energy(c_plus, W, theta)
        e_minus = hopfield_energy(c_minus, W, theta)

        gradient = abs(e_plus - e_minus) / (2 * epsilon)
        max_gradient = max(max_gradient, gradient)

    # Margin is inverse of max gradient (bounded)
    return 1.0 / (1.0 + max_gradient)


def hal_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    d: int = 6,
    R: float = PERFECT_FIFTH
) -> np.ndarray:
    """
    Harmonic Attention Layer (HAL).

    HAL(Q, K, V) = (QKᵀ / H(d, R₅)) · V

    Unlike standard softmax attention, this is:
    - No exponential computation (faster)
    - Gradient stable (divisor grows with depth)
    - Deterministic (reproducible across nodes)

    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        d: Security dimension
        R: Harmonic ratio

    Returns:
        Attention output (seq_len, d_v)

    Reference: Section 0.7
    """
    H = harmonic_scaling(d, R)

    # Scaled dot-product (no softmax)
    scores = (Q @ K.T) / H

    # Apply to values
    return scores @ V


def energy_threshold(
    k: int = 3,
    base_threshold: float = 1.0
) -> float:
    """
    Compute energy rejection threshold.

    Patterns with energy above this threshold are rejected.

    Args:
        k: Threshold multiplier (default: 3)
        base_threshold: Base energy level

    Returns:
        Energy threshold

    Reference: Appendix C, energy_threshold_k parameter
    """
    return k * base_threshold


def is_valid_pattern(
    c: np.ndarray,
    W: np.ndarray,
    theta: np.ndarray,
    threshold: float,
    use_hal: bool = True,
    d: int = 6
) -> Tuple[bool, float]:
    """
    Check if a context pattern is valid.

    Args:
        c: Context vector to check
        W: Learned weight matrix
        theta: Bias vector
        threshold: Energy threshold
        use_hal: Use HAL energy (recommended)
        d: Security dimension

    Returns:
        Tuple of (is_valid, energy)
    """
    if use_hal:
        energy = hal_energy(c, W, theta, d)
    else:
        energy = hopfield_energy(c, W, theta)

    return (energy < threshold, energy)


def hopfield_update(
    c: np.ndarray,
    W: np.ndarray,
    theta: np.ndarray,
    steps: int = 10
) -> np.ndarray:
    """
    Run Hopfield network update dynamics.

    Updates converge to nearest stored pattern (attractor).

    Args:
        c: Initial state
        W: Weight matrix
        theta: Bias vector
        steps: Update iterations

    Returns:
        Converged state
    """
    c = np.asarray(c, dtype=float).copy()

    for _ in range(steps):
        # Asynchronous update (random order)
        order = np.random.permutation(len(c))
        for i in order:
            h = W[i] @ c - theta[i]
            c[i] = 1.0 if h > 0 else -1.0 if h < 0 else c[i]

    return c


def cymatic_null_check(
    c: np.ndarray,
    n_mode: int,
    m_mode: int,
    L: float = 1.0,
    tolerance: float = 0.01
) -> bool:
    """
    Check if context sits at a cymatic nodal point (Claim 58).

    N(x, y; n, m) = cos(nπx/L)cos(mπy/L) - cos(mπx/L)cos(nπy/L) = 0

    Data is readable only when the context creates a cymatic null.

    Args:
        c: Context vector (uses first two components as x, y)
        n_mode: Mode parameter (derived from entropy)
        m_mode: Mode parameter (derived from security dimension)
        L: Domain size
        tolerance: How close to zero counts as null

    Returns:
        True if at nodal point

    Reference: Section 0.6 Cymatic Voxel Storage
    """
    if len(c) < 2:
        raise ValueError("Context must have at least 2 components")

    x, y = c[0], c[1]

    # Chladni equation
    term1 = math.cos(n_mode * math.pi * x / L) * math.cos(m_mode * math.pi * y / L)
    term2 = math.cos(m_mode * math.pi * x / L) * math.cos(n_mode * math.pi * y / L)
    N = term1 - term2

    return abs(N) < tolerance


class NeuralDefense:
    """
    Complete neural defense system combining Hopfield and HAL.
    """

    def __init__(
        self,
        pattern_dim: int = 6,
        security_dimension: int = 6,
        energy_k: int = 3
    ):
        """
        Initialize neural defense.

        Args:
            pattern_dim: Dimension of context vectors
            security_dimension: Security level (1-7)
            energy_k: Energy threshold multiplier
        """
        self.pattern_dim = pattern_dim
        self.security_dimension = security_dimension
        self.energy_k = energy_k

        self.W: Optional[np.ndarray] = None
        self.theta: np.ndarray = np.zeros(pattern_dim)
        self.threshold: float = energy_k

        self.valid_patterns: List[np.ndarray] = []

    def learn(self, patterns: List[Union[list, np.ndarray]]) -> None:
        """Learn valid patterns."""
        self.valid_patterns = [np.asarray(p) for p in patterns]
        self.W = learn_hopfield_weights(self.valid_patterns)

        # Set threshold based on training energies
        if self.valid_patterns:
            energies = [
                hal_energy(p, self.W, self.theta, self.security_dimension)
                for p in self.valid_patterns
            ]
            mean_energy = sum(energies) / len(energies)
            self.threshold = self.energy_k * abs(mean_energy) + 1.0

    def authorize(self, context: Union[list, np.ndarray]) -> Tuple[bool, dict]:
        """
        Authorize a context vector.

        Returns:
            Tuple of (authorized, details)
        """
        if self.W is None:
            return (False, {"error": "Model not trained"})

        c = np.asarray(context)
        energy = hal_energy(c, self.W, self.theta, self.security_dimension)
        margin = gradient_margin(c, self.W, self.theta)

        authorized = energy < self.threshold and margin > 0.1

        return (authorized, {
            "energy": energy,
            "threshold": self.threshold,
            "margin": margin,
            "security_dimension": self.security_dimension
        })
