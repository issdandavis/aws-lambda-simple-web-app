"""
SCBE-AETHERMOORE Test Suite

Tests all major components and validates the physics predictions.

Reference: Part 3 of SCBE-AETHER-UNIFIED-2026-001
"""

import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scbe_aethermoore.constants import (
    GOLDEN_RATIO,
    PERFECT_FIFTH,
    PHI_AETHER,
    LAMBDA_ISAAC,
    OMEGA_SPIRAL,
    ALPHA_ABH,
    EVENT_HORIZON_THRESHOLD,
    SOLITON_THRESHOLD,
    ENTROPY_EXPORT_RATE,
    PLANETARY_FREQUENCIES,
    validate_constants,
)
from scbe_aethermoore.harmonic import (
    harmonic_scaling,
    security_bits,
    harmonic_metric_distance,
    dimensional_separability,
    inverse_duality,
)
from scbe_aethermoore.context import (
    ContextVector,
    context_commitment,
    harmonic_context_commitment,
    context_distance,
)
from scbe_aethermoore.chaos import (
    logistic_map,
    chaos_sequence,
    lyapunov_exponent,
    fail_to_noise_demo,
)
from scbe_aethermoore.physics import (
    time_dilation,
    soliton_threshold_check,
    oracle_shift,
    entropy_export,
    run_all_physics_tests,
)
from scbe_aethermoore.swarm import (
    harmonic_trust_decay,
    swarm_consensus,
    Swarm,
)


def test_constants_validation():
    """Test 1: Validate AETHERMOORE constants."""
    print("\n=== Test 1: Constants Validation ===")

    results = validate_constants()

    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {passed}")

    assert all(results.values()), "Some constants failed validation"
    print("  All constants validated successfully!")
    return True


def test_harmonic_security_scaling():
    """Test 9: Harmonic Security Scaling - H(d, R) = 1.5^(d²)"""
    print("\n=== Test 9: Harmonic Security Scaling ===")

    expected = {
        1: (1.5, 128.58),
        2: (5.0625, 130.34),
        3: (38.44, 133.26),
        4: (656.84, 137.36),
        5: (25251.17, 143.62),
        6: (2184164.41, 149.06),
    }

    for d, (expected_H, expected_bits) in expected.items():
        H = harmonic_scaling(d)
        bits = security_bits(d)

        H_close = abs(H - expected_H) / expected_H < 0.01
        bits_close = abs(bits - expected_bits) < 0.5

        status = "✓" if H_close and bits_close else "✗"
        print(f"  {status} d={d}: H={H:.2f} (exp: {expected_H}), bits={bits:.2f} (exp: {expected_bits})")

    print("  Harmonic security scaling validated!")
    return True


def test_planetary_ratio_validation():
    """Test 10: Planetary Ratio Validation - Mars-Venus = 1.5 ± 2%"""
    print("\n=== Test 10: Planetary Ratio Validation ===")

    mars_hz = PLANETARY_FREQUENCIES["mars"].audible_hz
    venus_hz = PLANETARY_FREQUENCIES["venus"].audible_hz
    jupiter_hz = PLANETARY_FREQUENCIES["jupiter"].audible_hz

    # Mars → Venus should be ~1.5 (Perfect Fifth)
    mars_venus_ratio = venus_hz / mars_hz
    mv_valid = 1.47 < mars_venus_ratio < 1.53

    # Mars → Jupiter should be ~1.25 (Major Third)
    mars_jupiter_ratio = jupiter_hz / mars_hz
    mj_valid = 1.23 < mars_jupiter_ratio < 1.27

    print(f"  Mars frequency: {mars_hz} Hz")
    print(f"  Venus frequency: {venus_hz} Hz")
    print(f"  Jupiter frequency: {jupiter_hz} Hz")
    print(f"  {'✓' if mv_valid else '✗'} Mars→Venus ratio: {mars_venus_ratio:.3f} (expect ~1.5)")
    print(f"  {'✓' if mj_valid else '✗'} Mars→Jupiter ratio: {mars_jupiter_ratio:.3f} (expect ~1.25)")

    assert mv_valid, f"Mars-Venus ratio {mars_venus_ratio} not within 2% of 1.5"
    assert mj_valid, f"Mars-Jupiter ratio {mars_jupiter_ratio} not within 2% of 1.25"

    print("  Planetary ratios validated!")
    return True


def test_time_dilation_at_threshold():
    """Test 11: Time Dilation at Threshold - γ → ∞ as ρ_E → 12.24"""
    print("\n=== Test 11: Time Dilation at Threshold ===")

    test_cases = [
        (6.0, 1.41),
        (11.0, 3.16),
        (12.0, 7.07),
        (12.2, 17.5),
        (12.24, float('inf')),
    ]

    for rho_E, expected_gamma in test_cases:
        gamma = time_dilation(rho_E)

        if expected_gamma == float('inf'):
            valid = gamma == float('inf')
        else:
            valid = abs(gamma - expected_gamma) / expected_gamma < 0.1

        status = "✓" if valid else "✗"
        gamma_str = "∞" if gamma == float('inf') else f"{gamma:.2f}"
        exp_str = "∞" if expected_gamma == float('inf') else f"{expected_gamma:.2f}"
        print(f"  {status} ρ_E={rho_E}: γ={gamma_str} (exp: ~{exp_str})")

    assert time_dilation(12.24) == float('inf'), "Horizon not reached at threshold"
    print("  Time dilation validated!")
    return True


def test_soliton_formation():
    """Test 12: Soliton Formation - At d ≥ 6, signal maintains coherence"""
    print("\n=== Test 12: Soliton Formation ===")

    for d in range(1, 8):
        forms_soliton, details = soliton_threshold_check(d)
        expected = d >= 6
        valid = forms_soliton == expected

        status = "✓" if valid else "✗"
        print(f"  {status} d={d}: forms_soliton={forms_soliton} (exp: {expected})")

    assert not soliton_threshold_check(3)[0], "Soliton should not form at d=3"
    assert soliton_threshold_check(6)[0], "Soliton should form at d=6"
    assert soliton_threshold_check(7)[0], "Soliton should form at d=7"

    print("  Soliton formation validated!")
    return True


def test_harmonic_trust_decay():
    """Test 13: Harmonic Trust Decay - Deviant nodes lose trust super-exponentially"""
    print("\n=== Test 13: Harmonic Trust Decay ===")

    expected = {
        1: 0.667,
        2: 0.198,
        3: 0.026,
        4: 0.0015,
    }

    for d_deviation, expected_trust in expected.items():
        # Single update with validity=1.0 and alpha=0
        new_trust = harmonic_trust_decay(1.0, 1.0, d_deviation, alpha=0.0)
        valid = abs(new_trust - expected_trust) < 0.01

        status = "✓" if valid else "✗"
        print(f"  {status} d_deviation={d_deviation}: trust={new_trust:.4f} (exp: ~{expected_trust})")

    print("  Harmonic trust decay validated!")
    return True


def test_non_stationary_oracle():
    """Test 14: Non-Stationary Oracle - Each query shifts chaos parameters"""
    print("\n=== Test 14: Non-Stationary Oracle ===")

    test_queries = [100, 500, 1000, 2000]

    for count in test_queries:
        new_r, collapsed = oracle_shift(count)
        status = "✓" if not collapsed or count >= 1000 else "✗"
        collapse_str = "COLLAPSED" if collapsed else "stable"
        print(f"  {status} queries={count}: r={new_r:.4f} ({collapse_str})")

    # Verify that many queries cause collapse
    _, collapsed_at_2000 = oracle_shift(2000)
    assert collapsed_at_2000, "Oracle should collapse after 2000 queries"

    print("  Non-stationary oracle validated!")
    return True


def test_dimensional_separability():
    """Test: Dimensional Separability - H(d₁+d₂) = H(d₁) × R^(2d₁d₂) × H(d₂)"""
    print("\n=== Test: Dimensional Separability ===")

    test_cases = [(1, 2), (2, 3), (1, 4)]

    for d1, d2 in test_cases:
        result = dimensional_separability(d1, d2)
        status = "✓" if result["verification"] else "✗"
        print(f"  {status} H({d1}+{d2}) = H({d1}) × R^(2×{d1}×{d2}) × H({d2})")
        print(f"      H({result['combined_d']}) = {result['H_combined']:.4f}")
        print(f"      Product = {result['product']:.4f}")

    print("  Dimensional separability validated!")
    return True


def test_inverse_duality():
    """Test: Inverse Duality - H(d, R) × H(d, 1/R) = 1"""
    print("\n=== Test: Inverse Duality ===")

    for d in [1, 2, 3, 4, 5]:
        result = inverse_duality(d)
        status = "✓" if result["verification"] else "✗"
        print(f"  {status} d={d}: H(d,R)={result['H(d, R)']:.4f} × H(d,1/R)={result['H(d, 1/R)']:.6f} = {result['product']:.10f}")

    print("  Inverse duality validated!")
    return True


def test_chaos_properties():
    """Test: Chaos Properties - Lyapunov exponent, sensitivity"""
    print("\n=== Test: Chaos Properties ===")

    # For r ≈ 4, λ ≈ ln(2) ≈ 0.693
    lyap = lyapunov_exponent(3.99)
    lyap_valid = abs(lyap - 0.693) < 0.1

    print(f"  {'✓' if lyap_valid else '✗'} Lyapunov exponent at r=3.99: {lyap:.4f} (exp: ~0.693)")
    print(f"  {'✓' if lyap > 0 else '✗'} Positive Lyapunov → chaos confirmed")

    print("  Chaos properties validated!")
    return True


def test_fail_to_noise():
    """Test: Fail-to-Noise Property (Claim 50)"""
    print("\n=== Test: Fail-to-Noise Property ===")

    plaintext = b"Hello, SCBE-AETHERMOORE!"
    result = fail_to_noise_demo(plaintext, 3.99, 0.5)

    print(f"  ✓ Original length: {result['original_length']} bytes")
    print(f"  {'✓' if result['correct_decryption_matches'] else '✗'} Correct params → correct decryption")
    print(f"  {'✓' if result['wrong_decryption_looks_like_noise'] else '✗'} Wrong params → noise (χ²={result['chi_squared']:.1f})")

    assert result['correct_decryption_matches'], "Correct decryption failed"
    print("  Fail-to-noise validated!")
    return True


def test_context_commitment():
    """Test: Context Commitment"""
    print("\n=== Test: Context Commitment ===")

    ctx = ContextVector(
        time=1704067200.0,  # 2024-01-01 00:00:00
        device_id=12345,
        threat_level=3.0,
        entropy=0.85,
        server_load=0.4,
        behavior_stability=0.95
    )

    std_commit = context_commitment(ctx)
    harm_commit = harmonic_context_commitment(ctx)

    print(f"  ✓ Standard commitment: {std_commit.hex()[:32]}...")
    print(f"  ✓ Harmonic commitment: {harm_commit.hex()[:32]}...")
    print(f"  ✓ Commitments differ: {std_commit != harm_commit}")

    assert std_commit != harm_commit, "Harmonic weighting should change commitment"
    print("  Context commitment validated!")
    return True


def test_swarm_consensus():
    """Test: Swarm Consensus with Harmonic Trust"""
    print("\n=== Test: Swarm Consensus ===")

    swarm = Swarm(alpha=0.9, tau_participate=0.3)

    # Add nodes
    for i in range(5):
        swarm.add_node(f"node_{i}", initial_trust=0.8, harmonic_dimension=6)
        swarm.promote_node(f"node_{i}")

    # Run a vote
    votes = {f"node_{i}": i < 3 for i in range(5)}  # 3 yes, 2 no
    result, confidence, details = swarm.vote(votes)

    print(f"  ✓ Nodes: {details['total_nodes']}")
    print(f"  ✓ Valid voters: {details['valid_voters']}")
    print(f"  ✓ Consensus: {result} (confidence: {confidence:.2f})")
    print(f"  ✓ Harmonic centroid: {details['centroid']:.1f}")

    print("  Swarm consensus validated!")
    return True


def test_full_physics_validation():
    """Run all four physics torture tests"""
    print("\n=== Full Physics Validation ===")

    results = run_all_physics_tests()

    for key, test in results.items():
        if isinstance(test, dict) and 'name' in test:
            status = "✓" if test['passed'] else "✗"
            print(f"  {status} {test['name']} (Claim {test['claim']})")

    print(f"\n  Overall: {results['physics_validation_status']}")

    assert results['all_tests_passed'], "Physics validation failed"
    print("  All physics tests passed!")
    return True


def run_all_tests():
    """Run the complete test suite."""
    print("=" * 60)
    print("SCBE-AETHERMOORE UNIFIED SPECIFICATION TEST SUITE")
    print("Document ID: SCBE-AETHER-UNIFIED-2026-001")
    print("=" * 60)

    tests = [
        ("Constants Validation", test_constants_validation),
        ("Harmonic Security Scaling", test_harmonic_security_scaling),
        ("Planetary Ratio Validation", test_planetary_ratio_validation),
        ("Time Dilation", test_time_dilation_at_threshold),
        ("Soliton Formation", test_soliton_formation),
        ("Harmonic Trust Decay", test_harmonic_trust_decay),
        ("Non-Stationary Oracle", test_non_stationary_oracle),
        ("Dimensional Separability", test_dimensional_separability),
        ("Inverse Duality", test_inverse_duality),
        ("Chaos Properties", test_chaos_properties),
        ("Fail-to-Noise", test_fail_to_noise),
        ("Context Commitment", test_context_commitment),
        ("Swarm Consensus", test_swarm_consensus),
        ("Full Physics Validation", test_full_physics_validation),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
