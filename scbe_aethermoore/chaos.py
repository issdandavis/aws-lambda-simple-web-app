"""
Chaos Diffusion Module

Implements the logistic map chaos diffusion - the "secret sauce" of SCBE.
Wrong parameters = unscrambling gives noise (fail-to-noise property).

Reference: Section 1.4 of SCBE-AETHER-UNIFIED-2026-001
Claims: 4, 50
"""

import math
from typing import List, Tuple, Optional, Generator
from .constants import PERFECT_FIFTH, PARAMETER_RANGES
from .harmonic import harmonic_scaling, chaos_iterations


def logistic_map(x: float, r: float) -> float:
    """
    Single iteration of the logistic map: x_{n+1} = r × x_n × (1 - x_n)

    Args:
        x: Current value in (0, 1)
        r: Control parameter in [3.97, 4.0) for chaos

    Returns:
        Next value in the sequence

    Note:
        For r in [3.97, 4.0), the map exhibits chaotic behavior.
        Small changes in initial conditions lead to completely different sequences.
    """
    return r * x * (1 - x)


def chaos_sequence(
    x0: float,
    r: float,
    length: int,
    skip: int = 100
) -> List[float]:
    """
    Generate a chaotic sequence from the logistic map.

    Args:
        x0: Initial value in (0, 1)
        r: Control parameter in [3.97, 4.0)
        length: Number of values to generate
        skip: Iterations to skip (transient removal)

    Returns:
        List of chaotic values

    Reference: Section 1.4
    """
    if not 0 < x0 < 1:
        raise ValueError(f"x0 must be in (0, 1), got {x0}")
    if not 3.97 <= r < 4.0:
        raise ValueError(f"r must be in [3.97, 4.0), got {r}")

    x = x0

    # Skip transient
    for _ in range(skip):
        x = logistic_map(x, r)

    # Generate sequence
    sequence = []
    for _ in range(length):
        x = logistic_map(x, r)
        sequence.append(x)

    return sequence


def chaos_sequence_generator(
    x0: float,
    r: float,
    skip: int = 100
) -> Generator[float, None, None]:
    """
    Infinite generator for chaotic sequence values.

    Args:
        x0: Initial value
        r: Control parameter
        skip: Transient iterations to skip

    Yields:
        Chaotic values indefinitely
    """
    x = x0
    for _ in range(skip):
        x = logistic_map(x, r)

    while True:
        x = logistic_map(x, r)
        yield x


def chaos_diffusion(
    plaintext: bytes,
    r: float,
    x0: float,
    security_dimension: int = 6
) -> bytes:
    """
    Encrypt data using chaos diffusion.

    Each byte is XORed with a byte derived from the chaotic sequence.
    The number of iterations scales with the security dimension.

    Args:
        plaintext: Data to encrypt
        r: Chaos control parameter
        x0: Initial chaos value
        security_dimension: Security level (1-7)

    Returns:
        Encrypted data

    Reference: Section 1.4
    Claim: 4
    """
    # Scale iterations by harmonic depth
    iterations_per_byte = chaos_iterations(security_dimension)

    generator = chaos_sequence_generator(x0, r)
    ciphertext = bytearray()

    for byte in plaintext:
        # Generate chaos value through multiple iterations
        chaos_val = 0.0
        for _ in range(iterations_per_byte):
            chaos_val = next(generator)

        # Convert to byte and XOR
        chaos_byte = int(chaos_val * 256) % 256
        ciphertext.append(byte ^ chaos_byte)

    return bytes(ciphertext)


def chaos_undiffusion(
    ciphertext: bytes,
    r: float,
    x0: float,
    security_dimension: int = 6
) -> bytes:
    """
    Decrypt chaos-diffused data.

    Since XOR is its own inverse, this is identical to encryption
    with the same parameters.

    Args:
        ciphertext: Encrypted data
        r: Chaos control parameter (must match encryption)
        x0: Initial chaos value (must match encryption)
        security_dimension: Security level (must match encryption)

    Returns:
        Decrypted data

    Note:
        Wrong parameters produce noise (fail-to-noise property, Claim 50)
    """
    return chaos_diffusion(ciphertext, r, x0, security_dimension)


def lyapunov_exponent(r: float, iterations: int = 1000) -> float:
    """
    Compute the Lyapunov exponent for the logistic map.

    A positive Lyapunov exponent indicates chaos.
    For r ≈ 4, λ ≈ ln(2) ≈ 0.693.

    Args:
        r: Control parameter
        iterations: Number of iterations for estimation

    Returns:
        Estimated Lyapunov exponent
    """
    x = 0.5  # Starting point
    lyap_sum = 0.0

    for _ in range(iterations):
        # Derivative of logistic map: f'(x) = r(1 - 2x)
        derivative = abs(r * (1 - 2 * x))
        if derivative > 0:
            lyap_sum += math.log(derivative)
        x = logistic_map(x, r)

    return lyap_sum / iterations


def sensitivity_test(
    r: float,
    x0: float,
    perturbation: float = 1e-10,
    iterations: int = 100
) -> List[Tuple[int, float]]:
    """
    Demonstrate sensitive dependence on initial conditions.

    Two trajectories starting from x0 and x0+perturbation diverge
    exponentially (chaos signature).

    Args:
        r: Control parameter
        x0: Initial value
        perturbation: Small difference in initial condition
        iterations: Number of iterations to track

    Returns:
        List of (iteration, separation) pairs
    """
    x1 = x0
    x2 = x0 + perturbation
    separations = []

    for i in range(iterations):
        separation = abs(x1 - x2)
        separations.append((i, separation))
        x1 = logistic_map(x1, r)
        x2 = logistic_map(x2, r)

    return separations


def validate_chaos_regime(r: float) -> dict:
    """
    Validate that r produces proper chaotic behavior.

    Args:
        r: Control parameter to test

    Returns:
        Dict with validation results
    """
    lyap = lyapunov_exponent(r)

    # Generate sequence and check distribution
    seq = chaos_sequence(0.5, r, 10000)
    mean_val = sum(seq) / len(seq)
    variance = sum((x - mean_val) ** 2 for x in seq) / len(seq)

    # Check sensitivity
    sens = sensitivity_test(r, 0.5)
    max_separation = max(s[1] for s in sens)

    return {
        "r": r,
        "lyapunov_exponent": lyap,
        "is_chaotic": lyap > 0,
        "mean": mean_val,
        "variance": variance,
        "max_sensitivity_divergence": max_separation,
        "in_valid_range": 3.97 <= r < 4.0
    }


def fail_to_noise_demo(
    plaintext: bytes,
    correct_r: float,
    correct_x0: float,
    wrong_r: Optional[float] = None,
    wrong_x0: Optional[float] = None
) -> dict:
    """
    Demonstrate the fail-to-noise property (Claim 50).

    Wrong parameters produce noise, not corrupted plaintext.

    Args:
        plaintext: Original data
        correct_r, correct_x0: Correct chaos parameters
        wrong_r: Incorrect r value (or None for small perturbation)
        wrong_x0: Incorrect x0 value (or None for small perturbation)

    Returns:
        Dict showing correct vs incorrect decryption results
    """
    # Encrypt with correct parameters
    ciphertext = chaos_diffusion(plaintext, correct_r, correct_x0)

    # Decrypt with correct parameters
    correct_decrypt = chaos_undiffusion(ciphertext, correct_r, correct_x0)

    # Decrypt with wrong parameters
    r_wrong = wrong_r if wrong_r else correct_r + 0.0001
    x0_wrong = wrong_x0 if wrong_x0 else correct_x0 + 0.0001

    wrong_decrypt = chaos_undiffusion(ciphertext, r_wrong, x0_wrong)

    # Analyze results
    correct_match = correct_decrypt == plaintext

    # Check if wrong decryption looks like noise
    # Noise should have roughly uniform byte distribution
    wrong_bytes = list(wrong_decrypt)
    byte_counts = [wrong_bytes.count(i) for i in range(256)]
    expected_count = len(wrong_bytes) / 256
    chi_squared = sum((c - expected_count) ** 2 / expected_count
                      for c in byte_counts if expected_count > 0)

    # For truly random data, chi-squared should be ~255 with std ~22
    looks_like_noise = 180 < chi_squared < 330

    return {
        "original_length": len(plaintext),
        "correct_decryption_matches": correct_match,
        "wrong_decryption_looks_like_noise": looks_like_noise,
        "chi_squared": chi_squared,
        "wrong_params": {"r": r_wrong, "x0": x0_wrong},
        "correct_params": {"r": correct_r, "x0": correct_x0}
    }


def bifurcation_data(
    r_min: float = 3.5,
    r_max: float = 4.0,
    r_steps: int = 1000,
    iterations: int = 500,
    last_n: int = 100
) -> List[Tuple[float, List[float]]]:
    """
    Generate bifurcation diagram data.

    Shows the transition from periodic to chaotic behavior.

    Args:
        r_min, r_max: Range of r values
        r_steps: Number of r values to sample
        iterations: Total iterations per r
        last_n: Last n values to record (after transient)

    Returns:
        List of (r, [values]) pairs for plotting
    """
    data = []
    r_step = (r_max - r_min) / r_steps

    for i in range(r_steps):
        r = r_min + i * r_step
        x = 0.5

        # Skip transient
        for _ in range(iterations - last_n):
            x = logistic_map(x, r)

        # Collect last_n values
        values = []
        for _ in range(last_n):
            x = logistic_map(x, r)
            values.append(x)

        data.append((r, values))

    return data
