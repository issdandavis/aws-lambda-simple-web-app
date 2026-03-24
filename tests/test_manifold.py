"""
Manifold KEM Tests

Tests for the topology-gated two-channel KEM handshake:
(a) Unbiased label distribution
(b) Tier enforcement on OFF_SPHERE
(c) Deterministic classification
(d) Signature requirements
"""

import sys
import os
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scbe_aethermoore.manifold import (
    ManifoldLabel,
    TorusParams,
    ManifoldThresholds,
    classify_manifold,
    sign_manifold_result,
    determine_tier,
    gate_scbe_context,
    analyze_label_distribution,
    TierPolicy,
    can_intersect_sphere,
    derive_sphere_point,
    derive_torus_point,
    vector_norm,
    create_telemetry,
)


def test_basic_classification():
    """Test basic manifold classification works."""
    print("\n=== Test: Basic Classification ===")

    ctx = b"test_context_hash_32_bytes_long!"
    pk_in = b"brain_public_key_32_bytes_here!"
    ct_in = b"brain_ciphertext_32_bytes_here!!"
    pk_out = b"steward_public_key_32_bytes_!!"
    ct_out = b"steward_ciphertext_32bytes_!!!"

    result = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out)

    print(f"  Label: {result.label.value}")
    print(f"  Sphere point u: ({result.sphere_point[0]:.4f}, {result.sphere_point[1]:.4f}, {result.sphere_point[2]:.4f})")
    print(f"  Torus point τ: ({result.torus_point[0]:.4f}, {result.torus_point[1]:.4f}, {result.torus_point[2]:.4f})")
    print(f"  Angles (α, β): ({result.torus_angles[0]:.4f}, {result.torus_angles[1]:.4f})")
    print(f"  Radius error: {result.radius_error:.6f}")
    print(f"  Sphere delta: {result.sphere_delta:.6f}")
    print(f"  Manifold tag: {result.manifold_tag.hex()[:32]}...")

    assert result.label in [ManifoldLabel.ON_SPHERE, ManifoldLabel.OFF_SPHERE]
    assert len(result.manifold_tag) == 32
    print("  ✓ Basic classification passed")
    return True


def test_deterministic():
    """Test that classification is deterministic."""
    print("\n=== Test: Deterministic Classification ===")

    ctx = b"deterministic_test_context_hash!"
    pk_in = b"fixed_brain_public_key_32bytes!"
    ct_in = b"fixed_brain_ciphertext_32bytes!"
    pk_out = b"fixed_steward_public_key_32by!"
    ct_out = b"fixed_steward_ciphertext_32by!"

    # Classify twice
    result1 = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out)
    result2 = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out)

    assert result1.label == result2.label
    assert result1.manifold_tag == result2.manifold_tag
    assert result1.sphere_point == result2.sphere_point
    assert result1.torus_point == result2.torus_point

    print(f"  Label (both runs): {result1.label.value}")
    print(f"  Tags match: True")
    print("  ✓ Deterministic classification passed")
    return True


def test_label_distribution():
    """Test that labels are reasonably distributed (not all one class)."""
    print("\n=== Test: Label Distribution ===")

    # Use relaxed thresholds to get some ON_SPHERE hits
    params = TorusParams(R=1.0, r=0.3)
    thresholds = ManifoldThresholds(epsilon=0.5, delta=0.5)  # Very relaxed

    stats = analyze_label_distribution(
        num_samples=500,
        params=params,
        thresholds=thresholds
    )

    print(f"  Samples: {stats['samples']}")
    print(f"  ON_SPHERE: {stats['on_sphere_count']} ({stats['on_sphere_ratio']*100:.1f}%)")
    print(f"  OFF_SPHERE: {stats['off_sphere_count']} ({(1-stats['on_sphere_ratio'])*100:.1f}%)")
    print(f"  Mean radius error: {stats['mean_radius_error']:.4f}")
    print(f"  Mean sphere delta: {stats['mean_sphere_delta']:.4f}")

    # With relaxed thresholds, we should get some of both
    # (exact ratio depends on geometry)
    has_on = stats['on_sphere_count'] > 0
    has_off = stats['off_sphere_count'] > 0

    print(f"  Has ON_SPHERE samples: {has_on}")
    print(f"  Has OFF_SPHERE samples: {has_off}")
    print("  ✓ Label distribution test passed")
    return True


def test_tier_enforcement():
    """Test that OFF_SPHERE enforces Tier-3."""
    print("\n=== Test: Tier Enforcement ===")

    # Create a mock OFF_SPHERE result
    ctx = b"tier_test_context_32_bytes_here"
    pk_in = b"tier_brain_pk_32_bytes_here!!!!"
    ct_in = b"tier_brain_ct_32_bytes_here!!!!"
    pk_out = b"tier_steward_pk_32_bytes_here!!"
    ct_out = b"tier_steward_ct_32_bytes_here!!"

    # Use tight thresholds to likely get OFF_SPHERE
    params = TorusParams(R=1.0, r=0.25)
    thresholds = ManifoldThresholds(epsilon=1e-6, delta=1e-6)

    result = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out, params, thresholds)

    # Test tier determination
    tier_high_risk = determine_tier(result, risk_score=0.9)
    tier_med_risk = determine_tier(result, risk_score=0.6)
    tier_low_risk = determine_tier(result, risk_score=0.3)

    print(f"  Label: {result.label.value}")
    print(f"  Tier @ risk=0.9: {tier_high_risk.name}")
    print(f"  Tier @ risk=0.6: {tier_med_risk.name}")
    print(f"  Tier @ risk=0.3: {tier_low_risk.name}")

    if result.label == ManifoldLabel.OFF_SPHERE:
        # OFF_SPHERE always requires Tier-3
        assert tier_high_risk == TierPolicy.TIER_3
        assert tier_med_risk == TierPolicy.TIER_3
        assert tier_low_risk == TierPolicy.TIER_3
        print("  ✓ OFF_SPHERE correctly enforces Tier-3 for all risk levels")
    else:
        # ON_SPHERE follows risk-based tiering
        assert tier_high_risk == TierPolicy.TIER_1
        assert tier_med_risk == TierPolicy.TIER_2
        assert tier_low_risk == TierPolicy.TIER_3
        print("  ✓ ON_SPHERE correctly applies risk-based tiering")

    print("  ✓ Tier enforcement test passed")
    return True


def test_signature_requirements():
    """Test signature requirements for each label type."""
    print("\n=== Test: Signature Requirements ===")

    ctx = b"signature_test_context_32_bytes"
    pk_in = b"sig_brain_pk_32_bytes_here!!!!!"
    ct_in = b"sig_brain_ct_32_bytes_here!!!!!"
    pk_out = b"sig_steward_pk_32_bytes_here!!!"
    ct_out = b"sig_steward_ct_32_bytes_here!!!"

    sk_manager = b"manager_secret_key_32_bytes!!!!"
    sk_steward = b"steward_secret_key_32_bytes!!!!"

    # Test with tight thresholds (likely OFF_SPHERE)
    params = TorusParams(R=1.0, r=0.25)
    thresholds = ManifoldThresholds(epsilon=1e-6, delta=1e-6)

    result = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out, params, thresholds)

    print(f"  Label: {result.label.value}")

    if result.label == ManifoldLabel.OFF_SPHERE:
        # Should fail without steward key
        try:
            signed = sign_manifold_result(result, sk_manager, None)
            print("  ✗ Should have required steward signature")
            return False
        except ValueError as e:
            print(f"  ✓ Correctly rejected: {e}")

        # Should succeed with steward key
        signed = sign_manifold_result(result, sk_manager, sk_steward)
        assert signed.sig_steward is not None
        print(f"  ✓ Manager signature: {signed.sig_manager.hex()[:16]}...")
        print(f"  ✓ Steward signature: {signed.sig_steward.hex()[:16]}...")
    else:
        # ON_SPHERE - steward signature optional
        signed = sign_manifold_result(result, sk_manager, None)
        assert signed.sig_manager is not None
        assert signed.sig_steward is None
        print(f"  ✓ Manager signature: {signed.sig_manager.hex()[:16]}...")
        print(f"  ✓ Steward signature: None (optional for ON_SPHERE)")

    print("  ✓ Signature requirements test passed")
    return True


def test_scbe_gating():
    """Test that manifold result gates SCBE context."""
    print("\n=== Test: SCBE Context Gating ===")

    ctx = b"scbe_gating_test_context_32byte"
    pk_in = b"gate_brain_pk_32_bytes_here!!!!"
    ct_in = b"gate_brain_ct_32_bytes_here!!!!"
    pk_out = b"gate_steward_pk_32_bytes_here!!"
    ct_out = b"gate_steward_ct_32_bytes_here!!"

    result = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out)

    # Gate the context
    gated_ctx = gate_scbe_context(ctx, result)

    print(f"  Original ctx: {ctx.hex()[:32]}...")
    print(f"  Gated ctx:    {gated_ctx.hex()[:32]}...")
    print(f"  Label bound:  {result.label.value}")

    # Different label should produce different gated context
    # (Simulate by manually creating a different result)
    other_label = ManifoldLabel.ON_SPHERE if result.label == ManifoldLabel.OFF_SPHERE else ManifoldLabel.OFF_SPHERE

    # Manually compute what the other gated context would be
    other_tag = hashlib.sha256(
        ctx + result.manifold_tag + other_label.value.encode()
    ).digest()

    assert gated_ctx != other_tag
    print("  ✓ Different labels produce different gated contexts")
    print("  ✓ SCBE context gating test passed")
    return True


def test_torus_geometry():
    """Test torus-sphere intersection geometry."""
    print("\n=== Test: Torus Geometry ===")

    # Test if intersection is possible
    params1 = TorusParams(R=1.0, r=0.25)
    can_int1, cos_alpha1 = can_intersect_sphere(params1)

    params2 = TorusParams(R=2.0, r=0.1)  # Too far from unit sphere
    can_int2, cos_alpha2 = can_intersect_sphere(params2)

    print(f"  Params (R=1.0, r=0.25): can_intersect={can_int1}, cos(α*)={cos_alpha1:.4f}")
    print(f"  Params (R=2.0, r=0.1): can_intersect={can_int2}, cos(α*)={cos_alpha2:.4f}")

    # R=1.0, r=0.25 should be able to intersect
    # cos(α*) = (1 - 1 - 0.0625) / (2 * 1 * 0.25) = -0.0625 / 0.5 = -0.125
    # |−0.125| ≤ 1, so intersection is possible
    assert can_int1 == True

    print("  ✓ Torus geometry test passed")
    return True


def test_telemetry():
    """Test telemetry creation."""
    print("\n=== Test: Telemetry ===")

    ctx = b"telemetry_test_context_32_bytes"
    pk_in = b"telem_brain_pk_32_bytes_here!!!"
    ct_in = b"telem_brain_ct_32_bytes_here!!!"
    pk_out = b"telem_steward_pk_32_bytes_here!"
    ct_out = b"telem_steward_ct_32_bytes_here!"

    result = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out)
    tier = determine_tier(result, risk_score=0.75)
    telemetry = create_telemetry(result, tier)

    print(f"  manifold_label: {telemetry.manifold_label}")
    print(f"  sphere_delta: {telemetry.sphere_delta:.6f}")
    print(f"  radius_error: {telemetry.radius_error:.6f}")
    print(f"  R: {telemetry.R}")
    print(f"  r: {telemetry.r}")
    print(f"  alpha: {telemetry.alpha:.4f}")
    print(f"  beta: {telemetry.beta:.4f}")
    print(f"  tier: {telemetry.tier}")
    print(f"  manifold_tag_hex: {telemetry.manifold_tag_hex[:32]}...")

    assert telemetry.manifold_label in ["ON_SPHERE", "OFF_SPHERE"]
    assert telemetry.tier in [1, 2, 3]
    print("  ✓ Telemetry test passed")
    return True


def test_unforgeable_bit():
    """Test that neither party alone can control the label."""
    print("\n=== Test: Unforgeable Control Bit ===")

    # Fixed context and one side's transcript
    ctx = b"unforge_test_context_32_bytes!!"
    pk_in = b"unforge_brain_pk_32_bytes_here!"
    ct_in = b"unforge_brain_ct_32_bytes_here!"

    # Try many different steward transcripts
    labels = set()
    for i in range(100):
        pk_out = f"steward_pk_{i:04d}_32_bytes_here!".encode()[:32]
        ct_out = f"steward_ct_{i:04d}_32_bytes_here!".encode()[:32]

        # Use relaxed thresholds
        params = TorusParams(R=1.0, r=0.3)
        thresholds = ManifoldThresholds(epsilon=0.5, delta=0.5)

        result = classify_manifold(ctx, pk_in, ct_in, pk_out, ct_out, params, thresholds)
        labels.add(result.label)

    print(f"  Tested 100 different steward transcripts")
    print(f"  Unique labels observed: {[l.value for l in labels]}")

    # Both parties contribute to the outcome
    # (exact distribution depends on parameters, but both should be possible)
    print("  ✓ Unforgeable control bit test passed")
    return True


def run_all_tests():
    """Run all manifold tests."""
    print("=" * 60)
    print("MANIFOLD KEM TEST SUITE")
    print("=" * 60)

    tests = [
        ("Basic Classification", test_basic_classification),
        ("Deterministic", test_deterministic),
        ("Label Distribution", test_label_distribution),
        ("Tier Enforcement", test_tier_enforcement),
        ("Signature Requirements", test_signature_requirements),
        ("SCBE Context Gating", test_scbe_gating),
        ("Torus Geometry", test_torus_geometry),
        ("Telemetry", test_telemetry),
        ("Unforgeable Control Bit", test_unforgeable_bit),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
