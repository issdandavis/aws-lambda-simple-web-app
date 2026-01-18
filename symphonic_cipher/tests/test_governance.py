"""
Tests for Phase-Breath Hyperbolic Governance Module.

Tests cover:
- Phase transforms (Mobius addition, rotations)
- Breathing transforms
- Snap protocol (discontinuity detection)
- Causality verification
- Grand Unified Equation enforcement
- Integrated GovernanceEngine
"""

import numpy as np
import pytest
from symphonic_cipher.scbe_aethermoore.governance import (
    # Enums
    GovernanceDecision,
    BreathPhase,
    # Phase transforms
    mobius_add,
    clamp_ball,
    phase_transform,
    rotation_matrix_2d,
    rotation_matrix_nd,
    # Breathing transforms
    breathing_factor,
    get_breath_phase,
    breathing_transform,
    breathing_transform_timed,
    # Snap protocol
    SnapEvent,
    SnapProtocol,
    # Causality
    CausalityRecord,
    CausalityVerifier,
    # GUE
    GUEState,
    harmonic_scaling,
    evaluate_gue,
    gue_decision,
    # Engine
    GovernanceEngine,
    # Constants
    SNAP_THRESHOLD,
    B_BREATH_MAX,
)


class TestMobiusAddition:
    """Tests for Mobius addition on Poincare ball."""

    def test_identity(self):
        """Adding zero vector is identity."""
        u = np.array([0.3, 0.2, 0.1])
        zero = np.zeros(3)
        result = mobius_add(zero, u)
        np.testing.assert_array_almost_equal(result, u, decimal=10)

    def test_commutativity_near_origin(self):
        """Mobius addition is approximately commutative near origin."""
        a = np.array([0.01, 0.01])
        b = np.array([0.02, -0.01])
        ab = mobius_add(a, b)
        ba = mobius_add(b, a)
        # Near origin, should be approximately commutative
        np.testing.assert_array_almost_equal(ab, ba, decimal=2)

    def test_stays_in_ball(self):
        """Result stays in unit ball."""
        a = np.array([0.9, 0.0])
        b = np.array([0.9, 0.0])
        result = mobius_add(a, b)
        assert np.linalg.norm(result) < 1.0

    def test_inverse(self):
        """a ⊕ (-a) ≈ 0 for points in ball."""
        a = np.array([0.3, 0.4])
        neg_a = -a
        result = mobius_add(a, neg_a)
        np.testing.assert_array_almost_equal(result, np.zeros(2), decimal=10)


class TestClampBall:
    """Tests for ball clamping."""

    def test_inside_unchanged(self):
        """Points inside ball are unchanged."""
        u = np.array([0.3, 0.4, 0.2])
        result = clamp_ball(u, eps_ball=0.1)
        np.testing.assert_array_equal(result, u)

    def test_outside_clamped(self):
        """Points outside ball are clamped."""
        u = np.array([0.8, 0.8])  # norm > 1
        result = clamp_ball(u, eps_ball=0.001)
        assert np.linalg.norm(result) < 1.0

    def test_zero_unchanged(self):
        """Zero vector unchanged."""
        zero = np.zeros(4)
        result = clamp_ball(zero)
        np.testing.assert_array_equal(result, zero)


class TestPhaseTransform:
    """Tests for phase transform."""

    def test_translation_only(self):
        """Phase transform with no rotation."""
        u = np.array([0.1, 0.2])
        a = np.array([0.05, 0.0])
        result = phase_transform(u, a, Q=None)
        # Result should be different from u
        assert not np.allclose(result, u)
        # But still in ball
        assert np.linalg.norm(result) < 1.0

    def test_with_rotation(self):
        """Phase transform with rotation."""
        u = np.array([0.3, 0.0])
        a = np.zeros(2)
        Q = rotation_matrix_2d(np.pi / 2)  # 90 degree rotation
        result = phase_transform(u, a, Q=Q)
        expected = np.array([0.0, 0.3])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestRotationMatrices:
    """Tests for rotation matrix generation."""

    def test_2d_orthogonal(self):
        """2D rotation is orthogonal."""
        R = rotation_matrix_2d(0.5)
        # R^T R = I
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(2), decimal=10)

    def test_2d_determinant(self):
        """2D rotation has determinant 1."""
        R = rotation_matrix_2d(1.2)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_nd_orthogonal(self):
        """ND rotation is orthogonal."""
        R = rotation_matrix_nd(4, 0, 2, 0.7)
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(4), decimal=10)

    def test_nd_only_affects_plane(self):
        """ND rotation only affects specified plane."""
        R = rotation_matrix_nd(4, 1, 3, np.pi / 4)
        v = np.array([1, 0, 0, 0])
        # v is in axis 0, rotation is in plane (1,3), so v unchanged
        np.testing.assert_array_almost_equal(R @ v, v, decimal=10)


class TestBreathingTransform:
    """Tests for breathing transform."""

    def test_identity_at_b_equals_1(self):
        """b=1 is identity transform."""
        u = np.array([0.3, 0.4, 0.2])
        result = breathing_transform(u, b=1.0)
        np.testing.assert_array_almost_equal(result, u, decimal=5)

    def test_expansion(self):
        """b > 1 expands (pushes toward boundary)."""
        u = np.array([0.3, 0.0])
        result = breathing_transform(u, b=1.5)
        assert np.linalg.norm(result) > np.linalg.norm(u)

    def test_contraction(self):
        """b < 1 contracts (pulls toward origin)."""
        u = np.array([0.5, 0.0])
        result = breathing_transform(u, b=0.5)
        assert np.linalg.norm(result) < np.linalg.norm(u)

    def test_zero_unchanged(self):
        """Zero vector unchanged by any b."""
        zero = np.zeros(3)
        result = breathing_transform(zero, b=2.0)
        np.testing.assert_array_equal(result, zero)

    def test_stays_in_ball(self):
        """Result stays in ball."""
        u = np.array([0.9, 0.3])
        result = breathing_transform(u, b=2.0)
        assert np.linalg.norm(result) < 1.0


class TestBreathingFactor:
    """Tests for breathing factor calculation."""

    def test_oscillates(self):
        """Factor oscillates around 1."""
        factors = [breathing_factor(t) for t in np.linspace(0, 120, 100)]
        assert min(factors) < 1.0
        assert max(factors) > 1.0

    def test_bounded(self):
        """Factor is bounded by 1 ± b_max."""
        for t in np.linspace(0, 300, 50):
            b = breathing_factor(t)
            assert 1.0 - B_BREATH_MAX - 0.01 <= b <= 1.0 + B_BREATH_MAX + 0.01


class TestBreathPhase:
    """Tests for breath phase detection."""

    def test_phases_detected(self):
        """All phases can be detected at different times."""
        phases = set()
        for t in np.linspace(0, 120, 100):
            phases.add(get_breath_phase(t))
        assert BreathPhase.EXPANSION in phases or BreathPhase.CONTRACTION in phases


class TestSnapProtocol:
    """Tests for discontinuity detection."""

    def test_first_state_allowed(self):
        """First state is always allowed."""
        sp = SnapProtocol()
        state = np.array([0.1, 0.2, 0.3, 0.4])
        decision, event = sp.validate_and_record(0.0, state)
        assert decision == GovernanceDecision.ALLOW
        assert event is None

    def test_smooth_transition_allowed(self):
        """Smooth transitions are allowed."""
        sp = SnapProtocol(threshold=1.0)
        sp.validate_and_record(0.0, np.array([0.1, 0.0]))
        decision, _ = sp.validate_and_record(1.0, np.array([0.15, 0.0]))
        assert decision == GovernanceDecision.ALLOW

    def test_discontinuity_rejected(self):
        """Large jumps are rejected."""
        sp = SnapProtocol(threshold=0.1)
        sp.validate_and_record(0.0, np.array([0.1, 0.0]))
        decision, event = sp.validate_and_record(0.01, np.array([0.9, 0.0]))
        assert decision == GovernanceDecision.SNAP_VIOLATION
        assert event is not None
        assert event.magnitude > 0.1

    def test_negative_time_rejected(self):
        """Negative time delta is rejected."""
        sp = SnapProtocol()
        sp.validate_and_record(1.0, np.array([0.1, 0.0]))
        decision, event = sp.validate_and_record(0.5, np.array([0.1, 0.0]))
        assert decision == GovernanceDecision.SNAP_VIOLATION
        assert "causality" in event.reason.lower()


class TestCausalityVerifier:
    """Tests for causal chain verification."""

    def test_first_event_valid(self):
        """First event is always valid."""
        cv = CausalityVerifier()
        valid, reason = cv.verify_causality(1.0, np.array([0.1, 0.2]), None)
        assert valid
        assert reason == "OK"

    def test_increasing_timestamps(self):
        """Timestamps must increase."""
        cv = CausalityVerifier()
        cv.record_event(1.0, np.array([0.1, 0.2]))
        valid, reason = cv.verify_causality(0.5, np.array([0.2, 0.3]), None)
        assert not valid
        assert "Timestamp" in reason

    def test_chain_integrity(self):
        """Chain integrity is maintained."""
        cv = CausalityVerifier()
        for t in range(5):
            state = np.array([0.1 * t, 0.0])
            parent = cv.records[-1].event_hash if cv.records else None
            cv.record_event(float(t), state, parent)

        valid, reason = cv.verify_chain_integrity()
        assert valid
        assert cv.get_chain_length() == 5


class TestGrandUnifiedEquation:
    """Tests for GUE enforcement."""

    def test_harmonic_scaling_at_zero(self):
        """H(0) = 1."""
        H = harmonic_scaling(0.0)
        assert abs(H - 1.0) < 1e-10

    def test_harmonic_scaling_monotonic(self):
        """H is monotonically increasing in d."""
        d_values = [0, 0.5, 1.0, 2.0, 5.0]
        H_values = [harmonic_scaling(d) for d in d_values]
        for i in range(len(H_values) - 1):
            assert H_values[i] < H_values[i + 1]

    def test_harmonic_scaling_bounded(self):
        """H is bounded."""
        for d in [0, 1, 10, 100]:
            H = harmonic_scaling(d)
            assert 1.0 <= H <= 11.0  # 1 + alpha where alpha=10

    def test_evaluate_gue(self):
        """GUE evaluation produces valid state."""
        state = evaluate_gue(risk_base=0.2, d_star=1.0)
        assert state.risk_base == 0.2
        assert state.d_star == 1.0
        assert state.H_factor >= 1.0
        assert state.risk_prime == state.risk_base * state.H_factor

    def test_gue_decision_allow(self):
        """Low risk results in ALLOW."""
        state = GUEState(risk_base=0.1, d_star=0.1, H_factor=1.1, risk_prime=0.11)
        assert gue_decision(state) == GovernanceDecision.ALLOW

    def test_gue_decision_deny(self):
        """High risk results in DENY."""
        state = GUEState(risk_base=0.5, d_star=2.0, H_factor=5.0, risk_prime=2.5)
        assert gue_decision(state) == GovernanceDecision.DENY

    def test_gue_decision_quarantine(self):
        """Medium risk results in QUARANTINE."""
        state = GUEState(risk_base=0.3, d_star=0.5, H_factor=1.5, risk_prime=0.45)
        assert gue_decision(state) == GovernanceDecision.QUARANTINE


class TestGovernanceEngine:
    """Tests for integrated governance engine."""

    def test_initialization(self):
        """Engine initializes correctly."""
        engine = GovernanceEngine()
        assert engine.snap_protocol is not None
        assert engine.causality_verifier is not None

    def test_first_evaluation(self):
        """First evaluation works."""
        engine = GovernanceEngine()
        state = np.array([0.1, 0.2, 0.1, 0.2])
        decision, details = engine.evaluate(1.0, state, 0.1)
        assert decision in [GovernanceDecision.ALLOW, GovernanceDecision.QUARANTINE]
        assert "timestamp" in details
        assert "breath_phase" in details

    def test_smooth_trajectory_allowed(self):
        """Smooth trajectory is allowed."""
        engine = GovernanceEngine(snap_threshold=1.0)
        decisions = []
        for t in range(5):
            state = np.array([0.1 + 0.01 * t, 0.2, 0.1, 0.2])
            decision, _ = engine.evaluate(float(t), state, 0.1)
            decisions.append(decision)
        # All should be ALLOW (low risk, smooth)
        assert all(d == GovernanceDecision.ALLOW for d in decisions)

    def test_with_realm_centers(self):
        """Evaluation with realm centers works."""
        engine = GovernanceEngine()
        state = np.array([0.1, 0.2])
        centers = [np.array([0.0, 0.0]), np.array([0.5, 0.0])]
        decision, details = engine.evaluate(1.0, state, 0.1, realm_centers=centers)
        assert "d_star" in details
        assert details["d_star"] >= 0

    def test_statistics(self):
        """Statistics are tracked."""
        engine = GovernanceEngine()
        for t in range(3):
            state = np.array([0.1 * t, 0.0])
            engine.evaluate(float(t), state, 0.1)

        stats = engine.get_statistics()
        assert stats["causal_chain_length"] == 3
        assert stats["snap_count"] >= 0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete governance workflow."""
        engine = GovernanceEngine(snap_threshold=0.5, causality_window=10.0)

        # Define realm centers
        centers = [np.zeros(4), np.array([0.3, 0.0, 0.0, 0.0])]

        # Simulate trajectory
        results = []
        for i in range(10):
            t = float(i)
            state = np.array([0.05 * i, 0.02 * i, 0.01 * i, 0.0])
            state = clamp_ball(state)
            risk_base = 0.1 + 0.02 * i

            decision, details = engine.evaluate(t, state, risk_base, centers)
            results.append((decision, details))

        # Verify trajectory was processed
        assert len(results) == 10
        # Early low-risk states should be ALLOW
        assert results[0][0] == GovernanceDecision.ALLOW

    def test_attack_detection(self):
        """Test that sudden jumps (attacks) are detected."""
        engine = GovernanceEngine(snap_threshold=0.1)

        # Normal start
        engine.evaluate(0.0, np.array([0.1, 0.0, 0.0, 0.0]), 0.1)
        engine.evaluate(1.0, np.array([0.11, 0.0, 0.0, 0.0]), 0.1)

        # Sudden jump (simulated attack)
        decision, details = engine.evaluate(1.5, np.array([0.9, 0.5, 0.3, 0.2]), 0.1)

        # Should be rejected as snap violation
        assert decision == GovernanceDecision.SNAP_VIOLATION
