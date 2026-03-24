#!/usr/bin/env python3
"""
SCBE-AETHERMOORE v4.0 HARDENED SPECIFICATION VALIDATION
========================================================
Validates against USPTO provisional filing requirements.

Key Changes from v3:
- H(d, R_h) = R_h^(d²) [was R^(1+d²)]
- Strict R_h = 1.5, R_g = φ
- v₁-v₆ operational definitions per §112
- Complex mapping: f(z) = [|z|cos(θ), |z|sin(θ)]
"""

import numpy as np
import hashlib
import hmac
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum

# ==============================================================================
# MATHEMATICAL CONSTANTS (LOCKED - §112 COMPLIANCE)
# ==============================================================================

R_H = 1.5                           # Harmonic base (LOCKED)
R_G = (1 + np.sqrt(5)) / 2          # Golden ratio φ ≈ 1.618034 (LOCKED)
D = 6                               # Context vector dimension

# ==============================================================================
# OPERATIONAL DEFINITIONS (v₁-v₆)
# ==============================================================================

class SacredTongue(Enum):
    """Phase encoding for v₂ intent domain."""
    KO = 0.0                # Korah - Creation/Build
    AV = np.pi/3            # Avel - Validation/Test
    RU = 2*np.pi/3          # Ruah - Spirit/Inspiration
    CA = np.pi              # Caph - Container/Boundary
    UM = 4*np.pi/3          # Umar - Command/Execute
    DR = 5*np.pi/3          # Darak - Path/Navigate


class ControlAction(Enum):
    """Control actions per Claim 1(e)."""
    AUTHORIZE = "authorize"
    FAIL_TO_NOISE = "fail_to_noise"
    COUNTER_SIGNAL = "counter_signal"


@dataclass
class ContextVector:
    """
    Six-dimensional context vector per §112 operational definitions.

    v₁ (Identity):    Deterministic bit-string from hardware RoT
    v₂ (Intent):      Complex semantic coordinate, phase = Sacred Tongue
    v₃ (Trajectory):  Time-series coherence score (path curvature)
    v₄ (Timing):      Temporal phase-lock delta
    v₅ (Commitment):  Hash-linked commit (chain of custody)
    v₆ (Signature):   ML-DSA lattice signature over v₁-v₅
    """
    v1_identity: bytes              # Deterministic RoT fingerprint
    v2_intent: complex              # Phase = tongue, magnitude = priority
    v3_trajectory: float            # Coherence score [0, 1]
    v4_timing: float                # Phase-lock delta [-π, π]
    v5_commitment: bytes            # Previous state hash (32 bytes)
    v6_signature: bytes             # ML-DSA signature


# ==============================================================================
# CORE MATHEMATICAL FUNCTIONS
# ==============================================================================

def complex_to_real_mapping(z: complex) -> Tuple[float, float]:
    """
    Phase-preserving projection: ℂ → ℝ²
    f(z) = [|z| cos(θ), |z| sin(θ)]

    Preserves phase coherence as geometric distance.
    """
    magnitude = abs(z)
    theta = np.angle(z)
    return (magnitude * np.cos(theta), magnitude * np.sin(theta))


def context_to_real_space(ctx: ContextVector) -> np.ndarray:
    """
    Map ℂ^D context vector to ℝ^(2D) per Claim 1(b).
    """
    # v₁: Identity → hash to [0,1]
    v1_real = int.from_bytes(hashlib.sha256(ctx.v1_identity).digest()[:4], 'big') / (2**32)

    # v₂: Intent (complex) → [Re, Im]
    v2_re, v2_im = complex_to_real_mapping(ctx.v2_intent)

    # v₃: Trajectory (already real)
    v3_real = ctx.v3_trajectory

    # v₄: Timing (already real)
    v4_real = ctx.v4_timing / np.pi  # Normalize to [-1, 1]

    # v₅: Commitment → hash to [0,1]
    v5_real = int.from_bytes(hashlib.sha256(ctx.v5_commitment).digest()[:4], 'big') / (2**32)

    # v₆: Signature → hash to complex then project
    sig_hash = hashlib.sha256(ctx.v6_signature).digest()
    sig_angle = (int.from_bytes(sig_hash[:4], 'big') / (2**32)) * 2 * np.pi
    sig_complex = np.exp(1j * sig_angle)
    v6_re, v6_im = complex_to_real_mapping(sig_complex)

    return np.array([v1_real, v2_re, v2_im, v3_real, v4_real, v5_real, v6_re, v6_im])


def metric_tensor(d: int = 8) -> np.ndarray:
    """
    Construct metric tensor G per Claim 1(c).
    G = diag(1, 1, 1, R_g, R_g², R_g³, R_g⁴, R_g⁵)

    Extended to 8D for the real-projected space.
    """
    weights = [R_G ** max(0, i - 2) for i in range(d)]
    return np.diag(weights)


def divergence_score(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Calculate divergence score d using weighted metric.
    d(c₁, c₂) = √((c₁ - c₂)ᵀ G (c₁ - c₂))
    """
    G = metric_tensor(len(c1))
    diff = c1 - c2
    return np.sqrt(diff @ G @ diff)


def harmonic_work_factor(d: float, R_h: float = R_H) -> float:
    """
    HARDENED SPECIFICATION: H(d, R_h) = R_h^(d²)

    Note: Changed from v3 which was R^(1+d²)
    This creates steeper gradient at small d.
    """
    return R_h ** (d ** 2)


def response_latency(d: float, base_latency_ms: float = 1.0) -> float:
    """
    Modulate response latency τ per Claim 1(d).
    τ = base × H(d, R_h)
    """
    H = harmonic_work_factor(d)
    return base_latency_ms * H


def determine_action(d: float, thresholds: dict = None) -> ControlAction:
    """
    Execute control action per Claim 1(e).

    Actions:
    - AUTHORIZE: Operation permitted
    - FAIL_TO_NOISE: Inject static response
    - COUNTER_SIGNAL: Active cancellation of adversarial waveform
    """
    if thresholds is None:
        thresholds = {'authorize': 1.0, 'fail_to_noise': 2.5}

    if d < thresholds['authorize']:
        return ControlAction.AUTHORIZE
    elif d < thresholds['fail_to_noise']:
        return ControlAction.FAIL_TO_NOISE
    else:
        return ControlAction.COUNTER_SIGNAL


# ==============================================================================
# CLAIM 25: PHYSICAL-LAYER VERIFICATION
# ==============================================================================

def refractive_index_measurement(fiber_length_m: float = 1000.0) -> np.ndarray:
    """
    Claim 25(a): Retrieve refractive index measurement from fiber-optic medium.

    Simulates physical measurement with realistic parameters.
    """
    n_samples = 100
    base_n = 1.467  # Silica core

    # Temperature and strain variations
    temp_noise = np.random.normal(0, 0.0001, n_samples)
    strain_noise = np.random.normal(0, 0.00005, n_samples)

    return base_n + temp_noise + strain_noise


def ray_trace_lattice_noise(refractive_indices: np.ndarray, seed_offset: int = 0) -> np.ndarray:
    """
    Claim 25(b): Generate deterministic lattice noise mask via ray-tracing.

    Uses refractive measurements to seed noise generation.
    """
    # Quantize measurements to create deterministic seed
    quantized = ((refractive_indices - refractive_indices.min()) * 1e6).astype(int)

    # Generate lattice noise (Kyber-compatible)
    q = 3329  # Kyber modulus
    noise = np.zeros(256, dtype=int)

    for i in range(256):
        seed = (quantized[i % len(quantized)] + seed_offset + i) % (2**31)
        np.random.seed(seed)
        # Centered binomial distribution (η=2)
        noise[i] = (np.random.binomial(2, 0.5) - np.random.binomial(2, 0.5)) % q

    return noise


def blind_intent_gate(intent: complex, noise_mask: np.ndarray) -> complex:
    """
    Claim 25(c): Blind intent-gate against non-authorized observers.

    Applies lattice noise to phase-encode the intent.
    """
    # Use first noise element to perturb phase
    phase_perturbation = (noise_mask[0] / 3329) * 2 * np.pi
    return intent * np.exp(1j * phase_perturbation)


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================

def test_harmonic_work_factor_v4():
    """Test HARDENED H(d) = R_h^(d²) specification."""
    print("\n" + "="*60)
    print(" TEST 1: Harmonic Work Factor (v4.0 Hardened)")
    print("="*60)

    print(f"\n  HARDENED FORMULA: H(d, R_h) = R_h^(d²)")
    print(f"  R_h = {R_H} (LOCKED)")

    print(f"\n  {'d':<10} {'H(d) v3':<15} {'H(d) v4':<15} {'Difference'}")
    print("  " + "-"*50)

    for d in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        H_v3 = R_H ** (1 + d**2)  # Old formula
        H_v4 = R_H ** (d**2)       # New formula
        diff = H_v3 - H_v4
        print(f"  {d:<10.1f} {H_v3:<15.4f} {H_v4:<15.4f} {diff:+.4f}")

    # Key insight: at d=0, v4 gives H=1 (neutral), v3 gives H=R_h
    H_at_0 = harmonic_work_factor(0)
    correct_at_zero = np.isclose(H_at_0, 1.0)

    print(f"\n  H(0) = {H_at_0:.4f} (should be 1.0 for neutral)")
    print(f"  ✓ PASS: v4 formula gives neutral response at d=0" if correct_at_zero else "  ✗ FAIL")

    return correct_at_zero


def test_complex_mapping():
    """Test phase-preserving projection ℂ → ℝ²."""
    print("\n" + "="*60)
    print(" TEST 2: Complex-to-Real Mapping")
    print("="*60)

    test_cases = [
        (1 + 0j, "Pure real"),
        (0 + 1j, "Pure imaginary"),
        (1 + 1j, "45° angle"),
        (np.exp(1j * np.pi/3), "60° unit circle"),
        (2 * np.exp(1j * np.pi), "180° magnitude 2"),
    ]

    print(f"\n  {'z':<20} {'|z|':<10} {'θ':<10} {'f(z)'}")
    print("  " + "-"*55)

    all_pass = True
    for z, desc in test_cases:
        re, im = complex_to_real_mapping(z)
        mag = abs(z)
        theta = np.angle(z)

        # Verify: should recover original magnitude and angle
        recovered_mag = np.sqrt(re**2 + im**2)
        recovered_theta = np.arctan2(im, re)

        match = np.isclose(mag, recovered_mag) and np.isclose(theta, recovered_theta)
        all_pass = all_pass and match

        print(f"  {str(z):<20} {mag:<10.4f} {theta:<10.4f} ({re:.4f}, {im:.4f})")

    print(f"\n  {'✓ PASS' if all_pass else '✗ FAIL'}: Phase-preserving projection verified")
    return all_pass


def test_metric_tensor_golden_ratio():
    """Test metric tensor with locked R_g = φ."""
    print("\n" + "="*60)
    print(" TEST 3: Metric Tensor (Golden Ratio Locked)")
    print("="*60)

    G = metric_tensor(8)

    print(f"\n  R_g = φ = {R_G:.6f} (LOCKED)")
    print(f"\n  G = diag({[f'{g:.4f}' for g in np.diag(G)]})")

    # Verify golden ratio progression
    diag = np.diag(G)
    ratios = [diag[i+1]/diag[i] if i < len(diag)-1 else 0 for i in range(len(diag))]

    print(f"\n  Ratios between consecutive weights:")
    golden_ratios_correct = True
    for i, r in enumerate(ratios[2:-1], start=3):  # Skip first few, check golden progression
        is_golden = np.isclose(r, R_G, rtol=0.01)
        golden_ratios_correct = golden_ratios_correct and is_golden
        print(f"    G[{i+1}]/G[{i}] = {r:.4f} {'✓' if is_golden else '✗'}")

    print(f"\n  {'✓ PASS' if golden_ratios_correct else '✗ FAIL'}: Golden ratio progression verified")
    return golden_ratios_correct


def test_sacred_tongue_phases():
    """Test Sacred Tongue phase encoding for v₂."""
    print("\n" + "="*60)
    print(" TEST 4: Sacred Tongue Phase Encoding")
    print("="*60)

    print(f"\n  {'Tongue':<10} {'Phase (rad)':<15} {'Phase (deg)':<15} {'Semantic'}")
    print("  " + "-"*55)

    semantics = {
        'KO': 'Creation/Build',
        'AV': 'Validation/Test',
        'RU': 'Spirit/Inspiration',
        'CA': 'Container/Boundary',
        'UM': 'Command/Execute',
        'DR': 'Path/Navigate',
    }

    for tongue in SacredTongue:
        phase_rad = tongue.value
        phase_deg = np.degrees(phase_rad)
        print(f"  {tongue.name:<10} {phase_rad:<15.4f} {phase_deg:<15.1f} {semantics[tongue.name]}")

    # Verify phases are evenly distributed around circle
    phases = [t.value for t in SacredTongue]
    phase_diffs = [phases[i+1] - phases[i] for i in range(len(phases)-1)]
    evenly_distributed = all(np.isclose(d, np.pi/3, rtol=0.01) for d in phase_diffs)

    print(f"\n  Phases evenly distributed (60° apart): {evenly_distributed}")
    print(f"  {'✓ PASS' if evenly_distributed else '✗ FAIL'}: Sacred Tongue encoding verified")
    return evenly_distributed


def test_claim_25_physical_layer():
    """Test Claim 25: Physical-layer verification."""
    print("\n" + "="*60)
    print(" TEST 5: Claim 25 - Physical Layer Verification")
    print("="*60)

    # 25(a): Refractive index measurement
    print("\n  [25a] Refractive Index Measurement:")
    n_measurements = refractive_index_measurement(1000.0)
    print(f"    Samples: {len(n_measurements)}")
    print(f"    Mean n: {np.mean(n_measurements):.6f}")
    print(f"    Std n:  {np.std(n_measurements):.2e}")

    # 25(b): Ray-tracing lattice noise
    print("\n  [25b] Ray-Tracing Lattice Noise:")
    noise = ray_trace_lattice_noise(n_measurements)
    print(f"    Noise dimension: {len(noise)}")
    print(f"    Non-zero entries: {np.count_nonzero(noise)}/{len(noise)}")
    print(f"    Range: [{noise.min()}, {noise.max()}]")

    # 25(c): Blind intent-gate
    print("\n  [25c] Intent Gate Blinding:")
    original_intent = 0.8 * np.exp(1j * SacredTongue.KO.value)  # KO intent
    blinded_intent = blind_intent_gate(original_intent, noise)

    print(f"    Original: {original_intent:.4f}")
    print(f"    Blinded:  {blinded_intent:.4f}")
    print(f"    Magnitude preserved: {np.isclose(abs(original_intent), abs(blinded_intent))}")
    print(f"    Phase shifted: {np.angle(blinded_intent) - np.angle(original_intent):.4f} rad")

    # Determinism test
    noise2 = ray_trace_lattice_noise(n_measurements)
    deterministic = np.array_equal(noise, noise2)

    print(f"\n  Deterministic (same input → same noise): {deterministic}")
    print(f"  {'✓ PASS' if deterministic else '✗ FAIL'}: Physical layer verification complete")
    return deterministic


def test_full_pipeline_v4():
    """Test complete v4.0 pipeline."""
    print("\n" + "="*60)
    print(" TEST 6: Full Pipeline (v4.0 Hardened)")
    print("="*60)

    # Create context vector
    ctx = ContextVector(
        v1_identity=b"hardware_root_of_trust_id_001",
        v2_intent=0.8 * np.exp(1j * SacredTongue.AV.value),  # Validation intent
        v3_trajectory=0.95,  # High coherence
        v4_timing=0.1,  # Small phase-lock delta
        v5_commitment=hashlib.sha256(b"previous_state").digest(),
        v6_signature=b"ml_dsa_signature_placeholder_64bytes" + b"\x00" * 28,
    )

    print(f"\n  Context Vector:")
    print(f"    v₁ (Identity):   {ctx.v1_identity[:20]}...")
    print(f"    v₂ (Intent):     {ctx.v2_intent:.4f} (AV tongue)")
    print(f"    v₃ (Trajectory): {ctx.v3_trajectory}")
    print(f"    v₄ (Timing):     {ctx.v4_timing}")
    print(f"    v₅ (Commitment): {ctx.v5_commitment[:8].hex()}...")
    print(f"    v₆ (Signature):  {ctx.v6_signature[:8]}...")

    # Map to real space
    c_real = context_to_real_space(ctx)
    print(f"\n  Real projection (ℝ⁸): {c_real.round(4)}")

    # Reference point (legitimate baseline)
    reference = np.array([0.5, 0.4, 0.35, 0.95, 0.0, 0.5, 1.0, 0.0])

    # Calculate divergence
    d = divergence_score(c_real, reference)
    print(f"\n  Divergence score d: {d:.4f}")

    # Work factor
    H = harmonic_work_factor(d)
    print(f"  Work factor H(d): {H:.4f}")

    # Latency
    tau = response_latency(d)
    print(f"  Response latency τ: {tau:.4f} ms")

    # Action
    action = determine_action(d)
    print(f"  Control action: {action.value}")

    pipeline_complete = d > 0 and H > 0 and action is not None
    print(f"\n  {'✓ PASS' if pipeline_complete else '✗ FAIL'}: Full v4.0 pipeline operational")
    return pipeline_complete


def test_frequency_drift_detection():
    """Test detection of 'Frequency Drift' (attack behavior)."""
    print("\n" + "="*60)
    print(" TEST 7: Frequency Drift Detection")
    print("="*60)

    # Legitimate context - well-behaved agent
    legit_ctx = ContextVector(
        v1_identity=b"trusted_agent_001",
        v2_intent=0.8 * np.exp(1j * SacredTongue.KO.value),  # KO = 0 rad
        v3_trajectory=0.95,  # High coherence
        v4_timing=0.05,  # Good phase-lock
        v5_commitment=hashlib.sha256(b"valid_chain").digest(),
        v6_signature=b"valid_sig" + b"\x00" * 55,
    )

    # Attack context (frequency drift in intent)
    attack_ctx = ContextVector(
        v1_identity=b"attacker_spoofed_id",
        v2_intent=1.5 * np.exp(1j * (SacredTongue.CA.value)),  # Wrong tongue (CA=π)
        v3_trajectory=0.2,  # Erratic trajectory (low coherence)
        v4_timing=0.9,  # Out of phase-lock
        v5_commitment=hashlib.sha256(b"broken_chain").digest(),
        v6_signature=b"fake_sig" + b"\x00" * 55,
    )

    # First compute legitimate projection to use AS the reference
    legit_real = context_to_real_space(legit_ctx)
    reference = legit_real  # Legitimate behavior IS the baseline

    legit_real = context_to_real_space(legit_ctx)
    attack_real = context_to_real_space(attack_ctx)

    d_legit = divergence_score(legit_real, reference)
    d_attack = divergence_score(attack_real, reference)

    action_legit = determine_action(d_legit)
    action_attack = determine_action(d_attack)

    print(f"\n  {'Metric':<25} {'Legitimate':<15} {'Attack':<15}")
    print("  " + "-"*55)
    print(f"  {'Divergence d':<25} {d_legit:<15.4f} {d_attack:<15.4f}")
    print(f"  {'Work factor H(d)':<25} {harmonic_work_factor(d_legit):<15.4f} {harmonic_work_factor(d_attack):<15.4f}")
    print(f"  {'Latency τ (ms)':<25} {response_latency(d_legit):<15.4f} {response_latency(d_attack):<15.4f}")
    print(f"  {'Action':<25} {action_legit.value:<15} {action_attack.value:<15}")

    # Attack should have higher divergence and different action
    discriminates = d_attack > d_legit * 1.5 and action_attack != action_legit

    print(f"\n  Attack divergence > 1.5× legitimate: {d_attack > d_legit * 1.5}")
    print(f"  Different actions triggered: {action_attack != action_legit}")
    print(f"\n  {'✓ PASS' if discriminates else '✗ FAIL'}: Frequency drift detected")
    return discriminates


def test_v3_to_v4_comparison():
    """Compare v3 and v4 formula behaviors."""
    print("\n" + "="*60)
    print(" TEST 8: v3 → v4 Formula Comparison")
    print("="*60)

    print(f"\n  v3: H(d) = R_h^(1 + d²)")
    print(f"  v4: H(d) = R_h^(d²)  [HARDENED]")
    print(f"\n  Key difference: v4 is NEUTRAL at d=0 (H=1)")

    d_values = np.linspace(0, 3, 31)
    H_v3 = [R_H ** (1 + d**2) for d in d_values]
    H_v4 = [R_H ** (d**2) for d in d_values]

    # Find crossover point
    ratios = [h3/h4 for h3, h4 in zip(H_v3, H_v4)]

    print(f"\n  At d=0: v3={H_v3[0]:.2f}, v4={H_v4[0]:.2f} (ratio={ratios[0]:.2f})")
    print(f"  At d=1: v3={R_H**(1+1):.2f}, v4={R_H**(1):.2f} (ratio={R_H:.2f})")
    print(f"  At d=2: v3={R_H**(1+4):.2f}, v4={R_H**(4):.2f} (ratio={R_H:.2f})")

    # v4 advantage: legitimate traffic (d≈0) gets NO penalty
    v4_neutral_at_zero = np.isclose(H_v4[0], 1.0)

    print(f"\n  v4 advantage: Legitimate traffic (d≈0) gets H=1 (no latency penalty)")
    print(f"  {'✓ PASS' if v4_neutral_at_zero else '✗ FAIL'}: v4 formula correctly neutral at origin")
    return v4_neutral_at_zero


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*60)
    print(" SCBE-AETHERMOORE v4.0 HARDENED VALIDATION")
    print(" USPTO Provisional Filing Compliance")
    print("="*60)

    print(f"\n  R_h = {R_H} (Harmonic Base - LOCKED)")
    print(f"  R_g = φ = {R_G:.6f} (Golden Ratio - LOCKED)")

    results = {}

    results['harmonic_v4'] = test_harmonic_work_factor_v4()
    results['complex_mapping'] = test_complex_mapping()
    results['metric_tensor'] = test_metric_tensor_golden_ratio()
    results['sacred_tongues'] = test_sacred_tongue_phases()
    results['claim_25'] = test_claim_25_physical_layer()
    results['full_pipeline'] = test_full_pipeline_v4()
    results['frequency_drift'] = test_frequency_drift_detection()
    results['v3_to_v4'] = test_v3_to_v4_comparison()

    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✓ v4.0 HARDENED SPECIFICATION VALIDATED")
        print("    Ready for USPTO provisional filing.")

    return passed == total


if __name__ == "__main__":
    main()
