"""
Triadic Temporal Manifold for SCBE
===================================

Three parallel time axes for adaptive threat containment:
- Time¹: Linear (immediate reactivity)
- Time²: Quadratic (long-term memory)
- Time^G: Gravitational (critical containment)

Patent Claims: 21 (proposed)
"""

import numpy as np

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
G_SCBE = 1 / PHI            # Gravitational constant ≈ 0.618
EPSILON = 1e-9              # Singularity avoidance


def linear_time(t):
    """
    Time¹: Linear sequencing axis.

    Used for: trajectory ordering, short-term delta, immediate coherence.
    """
    return t


def quadratic_time(t, alpha=2.0):
    """
    Time²: Powered/accelerating time axis.

    τ = t^α (α=2 default)

    Used for: long-term memory weighting, exponential drift amplification.
    """
    return t ** alpha


def gravitational_time(t, divergence, hyperbolic_radius, k=0.5):
    """
    Time^G: Gravitational time dilation.

    t_g = t × √(1 - k×d / (r + ε))

    Inspired by Schwarzschild metric:
    - divergence (d) → "mass" (threat/intent energy density)
    - hyperbolic_radius (r) → distance from core
    - k → gravitational strength (tunable)

    When d is large and r is small → t_g → 0 (time freezes, ultimate sink)
    """
    # Compute dilation factor
    dilation_arg = 1 - (k * divergence) / (hyperbolic_radius + EPSILON)

    # Clamp to avoid negative sqrt (would indicate "event horizon" crossed)
    dilation_arg = max(dilation_arg, EPSILON)

    return t * np.sqrt(dilation_arg)


def triadic_time_state(t, divergence, hyperbolic_radius, alpha=2.0, k=0.5):
    """
    Compute the triadic temporal state (t, τ, t_g).

    Returns all three time coordinates for a given moment.
    """
    t1 = linear_time(t)
    t2 = quadratic_time(t, alpha)
    tg = gravitational_time(t, divergence, hyperbolic_radius, k)

    return {
        'linear': t1,
        'quadratic': t2,
        'gravitational': tg,
        'dilation_factor': tg / (t + EPSILON)  # How much time has slowed
    }


def triadic_divergence(c_t, c_tau, c_tg, reference, lambda1=0.5, lambda2=0.3, lambda3=0.2):
    """
    Combined divergence metric across all three time axes.

    d_triadic = √(λ₁×d₁² + λ₂×d₂² + λ₃×d₃²)

    Args:
        c_t: Context vector at linear time
        c_tau: Context vector at quadratic time
        c_tg: Context vector at gravitational time
        reference: Reference/baseline context vector
        lambda1-3: Weighting factors (must sum to 1.0)

    Returns:
        Combined triadic divergence score
    """
    d1 = np.linalg.norm(c_t - reference)    # Linear divergence
    d2 = np.linalg.norm(c_tau - reference)  # Quadratic divergence
    d3 = np.linalg.norm(c_tg - reference)   # Gravitational divergence

    return np.sqrt(lambda1 * d1**2 + lambda2 * d2**2 + lambda3 * d3**2)


def time_dilation_trap_depth(divergence, hyperbolic_radius, k=0.5):
    """
    Compute how deep into the "gravitational trap" an agent is.

    Returns value in [0, 1]:
    - 0: No dilation (normal time flow)
    - 1: Event horizon (time stopped, maximum sink)
    """
    trap_depth = (k * divergence) / (hyperbolic_radius + EPSILON)
    return min(trap_depth, 1.0)


def classify_temporal_regime(dilation_factor):
    """
    Classify which temporal regime dominates based on dilation.
    """
    if dilation_factor > 0.9:
        return "LINEAR_DOMINANT"      # Normal operation
    elif dilation_factor > 0.5:
        return "QUADRATIC_ACTIVE"     # Long-term memory kicking in
    elif dilation_factor > 0.1:
        return "GRAVITATIONAL_TRAP"   # Significant slowdown
    else:
        return "EVENT_HORIZON"        # Critical containment (near-frozen)


def evaluate_triadic_threat(trajectory, times, reference, alpha=2.0, k=0.5):
    """
    Full triadic temporal evaluation of a trajectory.

    Args:
        trajectory: Array of context vectors over time
        times: Corresponding time values
        reference: Baseline context vector
        alpha: Quadratic time exponent
        k: Gravitational strength

    Returns:
        Evaluation dict with all temporal metrics
    """
    results = []

    for i, (c, t) in enumerate(zip(trajectory, times)):
        # Compute divergence at linear time
        d_linear = np.linalg.norm(c - reference)

        # Hyperbolic radius (use divergence as proxy for distance from "safe" core)
        r = 1.0 / (d_linear + EPSILON)  # High divergence = close to danger = small r

        # Get triadic time state
        time_state = triadic_time_state(t, d_linear, r, alpha, k)

        # Trap depth
        trap = time_dilation_trap_depth(d_linear, r, k)

        # Regime classification
        regime = classify_temporal_regime(time_state['dilation_factor'])

        results.append({
            'step': i,
            'linear_time': time_state['linear'],
            'quadratic_time': time_state['quadratic'],
            'gravitational_time': time_state['gravitational'],
            'dilation_factor': time_state['dilation_factor'],
            'trap_depth': trap,
            'regime': regime,
            'divergence': d_linear
        })

    # Aggregate statistics
    mean_dilation = np.mean([r['dilation_factor'] for r in results])
    max_trap_depth = max([r['trap_depth'] for r in results])

    # Count regime occurrences
    regime_counts = {}
    for r in results:
        regime_counts[r['regime']] = regime_counts.get(r['regime'], 0) + 1

    # Determine dominant regime
    dominant_regime = max(regime_counts, key=regime_counts.get)

    # Final classification
    if max_trap_depth > 0.9:
        status = "CRITICAL_CONTAINMENT"
    elif max_trap_depth > 0.5:
        status = "GRAVITATIONAL_SINK"
    elif mean_dilation < 0.7:
        status = "TEMPORAL_ANOMALY"
    else:
        status = "NORMAL"

    return {
        'status': status,
        'mean_dilation_factor': mean_dilation,
        'max_trap_depth': max_trap_depth,
        'dominant_regime': dominant_regime,
        'regime_counts': regime_counts,
        'step_details': results
    }


# =============================================================================
# TESTS
# =============================================================================

def test_triadic_temporal():
    """Test triadic temporal manifold."""
    print("=" * 60)
    print("TRIADIC TEMPORAL MANIFOLD TESTS")
    print("=" * 60)

    # Test 1: Linear time is identity
    t = 5.0
    assert linear_time(t) == t, "Linear time should be identity"
    print(f"[PASS] Linear time: t={t} → {linear_time(t)}")

    # Test 2: Quadratic time grows super-linearly
    t2 = quadratic_time(t, alpha=2.0)
    assert t2 == 25.0, f"Quadratic time t²: expected 25, got {t2}"
    print(f"[PASS] Quadratic time: t={t}, α=2 → τ={t2}")

    # Test 3: Gravitational time dilates under threat
    # Low threat (small d, large r) → minimal dilation
    tg_safe = gravitational_time(t, divergence=0.1, hyperbolic_radius=10.0, k=0.5)
    assert tg_safe > 0.99 * t, "Low threat should have minimal dilation"
    print(f"[PASS] Gravitational time (safe): d=0.1, r=10 → t_g={tg_safe:.4f} (≈{t})")

    # High threat (large d, small r) → severe dilation
    tg_danger = gravitational_time(t, divergence=5.0, hyperbolic_radius=0.5, k=0.5)
    assert tg_danger < 0.5 * t, "High threat should have severe dilation"
    print(f"[PASS] Gravitational time (danger): d=5.0, r=0.5 → t_g={tg_danger:.4f} (<<{t})")

    # Test 4: Triadic time state
    state = triadic_time_state(t, divergence=1.0, hyperbolic_radius=2.0, alpha=2.0, k=0.5)
    assert 'linear' in state and 'quadratic' in state and 'gravitational' in state
    print(f"[PASS] Triadic state: t¹={state['linear']}, t²={state['quadratic']}, t^G={state['gravitational']:.4f}")
    print(f"       Dilation factor: {state['dilation_factor']:.4f}")

    # Test 5: Trap depth calculation
    trap_low = time_dilation_trap_depth(0.1, 10.0, k=0.5)
    trap_high = time_dilation_trap_depth(5.0, 0.5, k=0.5)
    assert trap_low < trap_high, "Higher threat should have deeper trap"
    print(f"[PASS] Trap depth: low threat={trap_low:.4f}, high threat={trap_high:.4f}")

    # Test 6: Regime classification
    assert classify_temporal_regime(0.95) == "LINEAR_DOMINANT"
    assert classify_temporal_regime(0.7) == "QUADRATIC_ACTIVE"
    assert classify_temporal_regime(0.3) == "GRAVITATIONAL_TRAP"
    assert classify_temporal_regime(0.05) == "EVENT_HORIZON"
    print("[PASS] Regime classification: all regimes correctly identified")

    # Test 7: Full trajectory evaluation
    print("\n" + "-" * 40)
    print("Trajectory Evaluation Test")
    print("-" * 40)

    np.random.seed(42)
    reference = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Legitimate trajectory (stays near reference)
    legit_traj = [reference + np.random.normal(0, 0.1, 6) for _ in range(20)]
    times = np.linspace(0.1, 10, 20)

    legit_result = evaluate_triadic_threat(legit_traj, times, reference)
    print(f"\nLEGIT trajectory:")
    print(f"  Status: {legit_result['status']}")
    print(f"  Mean dilation: {legit_result['mean_dilation_factor']:.4f}")
    print(f"  Max trap depth: {legit_result['max_trap_depth']:.4f}")
    print(f"  Dominant regime: {legit_result['dominant_regime']}")

    # Attack trajectory (diverges from reference)
    attack_traj = [reference + np.array([i*0.3, i*0.2, i*0.25, i*0.1, i*0.15, i*0.2]) for i in range(20)]

    attack_result = evaluate_triadic_threat(attack_traj, times, reference)
    print(f"\nATTACK trajectory:")
    print(f"  Status: {attack_result['status']}")
    print(f"  Mean dilation: {attack_result['mean_dilation_factor']:.4f}")
    print(f"  Max trap depth: {attack_result['max_trap_depth']:.4f}")
    print(f"  Dominant regime: {attack_result['dominant_regime']}")

    # Verify differentiation
    assert legit_result['mean_dilation_factor'] > attack_result['mean_dilation_factor'], \
        "Attack should have more time dilation"
    assert legit_result['max_trap_depth'] < attack_result['max_trap_depth'], \
        "Attack should have deeper trap"

    print("\n" + "=" * 60)
    print("ALL TRIADIC TEMPORAL TESTS PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_triadic_temporal()
