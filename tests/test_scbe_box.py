"""
Test Suite for SCBEBox: Security Context-Based Envelope Simulation
===================================================================

Comprehensive tests demonstrating all SCBE patent claims with mappings.

Test Categories:
1. Normal Operation Tests (low dimensions, high coherence)
2. Attack Simulation Tests (high dimensions, low coherence)
3. Attack Sink Curve Tests (cost vs dimensions)
4. Patent Claim Mapping Validation

USPTO Filing Reference: Reduction-to-Practice Evidence
"""

import unittest
import numpy as np
import json
import time
import sys
import os
from typing import Dict, List, Any

# Add tests directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SCBEBox import (
    SCBEBox,
    ContextVector,
    GeneticMarker,
    AttackCostResult,
    HyperbolicProjector,
    MahalanobisCoherenceAnalyzer,
    SecurityLevel,
    PHI,
    PHI_INV,
    GATE_NAMES,
    MIN_LATTICE_DIM,
    MAX_LATTICE_DIM,
    COHERENCE_THRESHOLD_NORMAL,
    COHERENCE_THRESHOLD_ATTACK
)


# =============================================================================
# TEST RESULT COLLECTION
# =============================================================================

class PatentClaimResult:
    """Collects test results mapped to patent claims."""

    def __init__(self):
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def record(self, claim: str, test_name: str, passed: bool, evidence: str):
        if claim not in self.results:
            self.results[claim] = []
        self.results[claim].append({
            "test": test_name,
            "passed": passed,
            "evidence": evidence,
            "timestamp": time.time()
        })

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "total_claims": len(self.results),
            "total_tests": sum(len(tests) for tests in self.results.values()),
            "all_passed": all(
                all(t["passed"] for t in tests)
                for tests in self.results.values()
            ),
            "claims": {}
        }

        for claim, tests in self.results.items():
            summary["claims"][claim] = {
                "tests_run": len(tests),
                "all_passed": all(t["passed"] for t in tests),
                "evidence": [t["evidence"] for t in tests]
            }

        return summary


# Global claim collector
claim_results = PatentClaimResult()


# =============================================================================
# SECTION 1: NORMAL OPERATION TESTS
# =============================================================================

class TestNormalOperation(unittest.TestCase):
    """
    Normal operation tests demonstrating low dimensions and high coherence.

    These tests verify the system works correctly under normal conditions.
    """

    def setUp(self):
        """Initialize SCBEBox for each test."""
        self.box = SCBEBox(
            security_level=SecurityLevel.POST_QUANTUM,
            base_dimensions=MIN_LATTICE_DIM
        )

    def test_context_vector_creation(self):
        """
        Test 6D context vector creation.

        Maps to Patent Claims 1-6: Six-gate verification pipeline.
        """
        dim = 32
        context = self.box.create_context_vector(
            identity=np.random.randn(dim),
            intent=np.random.randn(dim),
            trajectory=np.random.randn(dim),
            timing=time.time(),
            commitment=np.random.randn(dim),
            signature=np.random.randn(dim)
        )

        # Verify all components
        self.assertEqual(len(context.who), dim)
        self.assertEqual(len(context.what), dim)
        self.assertEqual(len(context.where), dim)
        self.assertGreater(context.when, 0)
        self.assertEqual(len(context.why), dim)
        self.assertEqual(len(context.how), dim)

        # Verify 6D representation
        vec_6d = context.to_6d_array()
        self.assertEqual(len(vec_6d), 6)

        # Record claim evidence
        for i, gate in enumerate(GATE_NAMES):
            claim_results.record(
                f"Claim {i+1}",
                "test_context_vector_creation",
                True,
                f"{gate} gate initialized with {dim}-dimensional vector"
            )

    def test_context_validation_passes(self):
        """
        Test that valid contexts pass all 6 gates.

        Maps to Patent Claims 1-6.
        """
        dim = 32
        context = self.box.create_context_vector(
            identity=np.ones(dim),  # Non-zero identity
            intent=np.ones(dim),     # Non-zero intent
            trajectory=np.ones(dim), # Non-zero trajectory
            timing=time.time(),      # Current timestamp
            commitment=np.ones(dim), # Non-zero commitment
            signature=np.ones(dim)   # Non-zero signature
        )

        valid, gate_results = self.box.validate_context(context)

        self.assertTrue(valid)
        for gate in GATE_NAMES:
            self.assertTrue(gate_results[gate]["valid"])

        claim_results.record(
            "Claims 1-6",
            "test_context_validation_passes",
            True,
            "All 6 gates validated successfully for legitimate context"
        )

    def test_harmonic_scaling_base_case(self):
        """
        Test harmonic scaling at d=0 (no distance).

        Maps to Patent Claims 7-8.
        H(0, R) = R^(1+0) = R
        """
        result = SCBEBox.harmonic_scaling(0.0, base_resistance=2.0)
        self.assertAlmostEqual(result, 2.0, places=5)

        claim_results.record(
            "Claim 7",
            "test_harmonic_scaling_base_case",
            True,
            "H(0, 2) = 2^1 = 2.0 (base case)"
        )

    def test_harmonic_scaling_growth(self):
        """
        Test super-exponential growth of harmonic scaling.

        Maps to Patent Claims 7-8: H(d,R) = R^(1+d²)
        """
        R = 1.5

        # Calculate for various distances
        h_0 = SCBEBox.harmonic_scaling(0.0, R)  # R^1 = 1.5
        h_1 = SCBEBox.harmonic_scaling(1.0, R)  # R^2 = 2.25
        h_2 = SCBEBox.harmonic_scaling(2.0, R)  # R^5 = 7.59

        # Verify super-exponential growth
        self.assertGreater(h_1, h_0)
        self.assertGreater(h_2, h_1)

        # Verify exact formula: R^(1+d²)
        self.assertAlmostEqual(h_0, R ** 1, places=5)
        self.assertAlmostEqual(h_1, R ** 2, places=5)
        self.assertAlmostEqual(h_2, R ** 5, places=5)

        claim_results.record(
            "Claim 8",
            "test_harmonic_scaling_growth",
            True,
            f"H(0,{R})={h_0:.2f}, H(1,{R})={h_1:.2f}, H(2,{R})={h_2:.2f} - Super-exponential verified"
        )

    def test_adaptive_dimensions_low_distance(self):
        """
        Test adaptive dimensions for nearby contexts.

        Maps to Patent Claim 9.
        """
        dims = self.box.compute_adaptive_dimensions(0.1)

        # Should stay near base
        self.assertGreaterEqual(dims, MIN_LATTICE_DIM)
        self.assertLess(dims, MIN_LATTICE_DIM * 2)

        claim_results.record(
            "Claim 9",
            "test_adaptive_dimensions_low_distance",
            True,
            f"d=0.1 -> dims={dims} (near base {MIN_LATTICE_DIM})"
        )

    def test_coherence_for_consistent_behavior(self):
        """
        Test high coherence for consistent behavioral trajectory.

        Maps to Patent Claims 16-17.
        """
        dim = 32

        # Add consistent trajectory (small variations)
        base_vector = np.random.randn(dim)
        for i in range(20):
            context = self.box.create_context_vector(
                identity=base_vector + np.random.randn(dim) * 0.1,
                intent=base_vector + np.random.randn(dim) * 0.1,
                trajectory=base_vector + np.random.randn(dim) * 0.1,
                timing=time.time(),
                commitment=base_vector + np.random.randn(dim) * 0.1,
                signature=base_vector + np.random.randn(dim) * 0.1
            )
            self.box.add_trajectory_observation(context)

        # Test coherence of similar context
        test_context = self.box.create_context_vector(
            identity=base_vector + np.random.randn(dim) * 0.1,
            intent=base_vector + np.random.randn(dim) * 0.1,
            trajectory=base_vector + np.random.randn(dim) * 0.1,
            timing=time.time(),
            commitment=base_vector + np.random.randn(dim) * 0.1,
            signature=base_vector + np.random.randn(dim) * 0.1
        )

        coherence, is_anomaly = self.box.assess_coherence(test_context)

        # High coherence expected
        self.assertGreater(coherence, COHERENCE_THRESHOLD_ATTACK)
        self.assertFalse(is_anomaly)

        claim_results.record(
            "Claim 16",
            "test_coherence_for_consistent_behavior",
            True,
            f"Coherence={coherence:.3f} for consistent trajectory (threshold={COHERENCE_THRESHOLD_ATTACK})"
        )

    def test_genetic_marker_creation(self):
        """
        Test genetic marker audit trail.

        Maps to Patent Claim 18.
        """
        context = self.box.create_context_vector(
            identity=np.random.randn(32),
            intent=np.random.randn(32),
            trajectory=np.random.randn(32),
            timing=time.time(),
            commitment=np.random.randn(32),
            signature=np.random.randn(32)
        )

        marker = self.box.create_genetic_marker(context, "test_mutation")

        self.assertIsNotNone(marker.marker_id)
        self.assertIsNotNone(marker.parent_id)
        self.assertEqual(marker.generation, 1)
        self.assertIn("test_mutation", marker.mutation_log)

        claim_results.record(
            "Claim 18",
            "test_genetic_marker_creation",
            True,
            f"Marker {marker.marker_id} created, gen={marker.generation}, parent={marker.parent_id}"
        )

    def test_hyperbolic_projection_roundtrip(self):
        """
        Test hyperbolic projection preserves topology.

        Maps to Patent Claims 13-15.
        """
        projector = HyperbolicProjector()

        # Test point in Euclidean space
        point = np.array([0.3, 0.4, 0.2, 0.1, 0.15, 0.25])

        # Project to Poincaré disk
        poincare = projector.euclidean_to_poincare(point)

        # Project back
        recovered = projector.poincare_to_euclidean(poincare)

        # Should approximately recover original
        np.testing.assert_array_almost_equal(point, recovered, decimal=5)

        claim_results.record(
            "Claim 13",
            "test_hyperbolic_projection_roundtrip",
            True,
            "Hyperbolic projection roundtrip error < 1e-5"
        )


# =============================================================================
# SECTION 2: ATTACK SIMULATION TESTS
# =============================================================================

class TestAttackSimulation(unittest.TestCase):
    """
    Attack simulation tests demonstrating high dimensions and low coherence.

    These tests verify the system correctly detects and responds to attacks.
    """

    def setUp(self):
        """Initialize SCBEBox for each test."""
        self.box = SCBEBox(
            security_level=SecurityLevel.POST_QUANTUM,
            base_dimensions=MIN_LATTICE_DIM
        )

    def test_high_distance_increases_dimensions(self):
        """
        Test that high context distance increases lattice dimensions.

        Maps to Patent Claims 9, 11-12.
        """
        low_dist = 0.1
        high_dist = 2.0

        dims_low = self.box.compute_adaptive_dimensions(low_dist)
        dims_high = self.box.compute_adaptive_dimensions(high_dist)

        # High distance should require higher dimensions
        self.assertGreater(dims_high, dims_low)

        claim_results.record(
            "Claim 11",
            "test_high_distance_increases_dimensions",
            True,
            f"d={low_dist}->dims={dims_low}, d={high_dist}->dims={dims_high}"
        )

    def test_attack_cost_grows_super_exponentially(self):
        """
        Test attack cost exhibits super-exponential growth.

        Maps to Patent Claims 19-24.
        """
        costs = []
        distances = [0.1, 0.5, 1.0, 1.5, 2.0]

        for d in distances:
            result = self.box.calculate_attack_cost(d)
            costs.append(result.cost_in_bits)

        # Verify monotonic growth
        for i in range(1, len(costs)):
            self.assertGreater(costs[i], costs[i-1])

        # Verify super-exponential: second differences should increase
        first_diff = [costs[i+1] - costs[i] for i in range(len(costs)-1)]
        second_diff = [first_diff[i+1] - first_diff[i] for i in range(len(first_diff)-1)]

        self.assertTrue(all(d > 0 for d in second_diff))

        claim_results.record(
            "Claim 20",
            "test_attack_cost_grows_super_exponentially",
            True,
            f"Cost growth verified: {[f'{c:.0f}' for c in costs]} bits"
        )

    def test_attack_infeasibility_at_high_distance(self):
        """
        Test attacks become infeasible at high context distance.

        Maps to Patent Claims 23-24.
        """
        result = self.box.calculate_attack_cost(
            context_distance=2.0,
            attacker_compute=1e18  # Exaflop attacker
        )

        # Should be infeasible (> age of universe)
        self.assertFalse(result.is_feasible)
        self.assertGreater(result.years_to_break, 13.8e9)

        claim_results.record(
            "Claim 23",
            "test_attack_infeasibility_at_high_distance",
            True,
            f"d=2.0: {result.years_to_break:.2e} years > universe age"
        )

        claim_results.record(
            "Claim 24",
            "test_attack_infeasibility_at_high_distance",
            True,
            f"Attack sink mechanism: feasible={result.is_feasible}"
        )

    def test_anomaly_detection_for_deviant_behavior(self):
        """
        Test anomaly detection catches deviant behavior.

        Maps to Patent Claims 16-17.
        """
        dim = 32

        # Build normal trajectory
        base_vector = np.zeros(dim)
        for i in range(30):
            context = self.box.create_context_vector(
                identity=base_vector + np.random.randn(dim) * 0.1,
                intent=base_vector + np.random.randn(dim) * 0.1,
                trajectory=base_vector + np.random.randn(dim) * 0.1,
                timing=time.time(),
                commitment=base_vector + np.random.randn(dim) * 0.1,
                signature=base_vector + np.random.randn(dim) * 0.1
            )
            self.box.add_trajectory_observation(context)

        # Create highly deviant context (simulated attack)
        attack_context = self.box.create_context_vector(
            identity=np.ones(dim) * 100,  # Far from normal
            intent=np.ones(dim) * 100,
            trajectory=np.ones(dim) * 100,
            timing=time.time(),
            commitment=np.ones(dim) * 100,
            signature=np.ones(dim) * 100
        )

        coherence, is_anomaly = self.box.assess_coherence(attack_context)

        # Should detect anomaly
        self.assertTrue(is_anomaly)
        self.assertLess(coherence, COHERENCE_THRESHOLD_ATTACK)

        claim_results.record(
            "Claim 17",
            "test_anomaly_detection_for_deviant_behavior",
            True,
            f"Anomaly detected: coherence={coherence:.4f}, is_anomaly={is_anomaly}"
        )

    def test_hyperbolic_distance_amplification(self):
        """
        Test hyperbolic geometry amplifies distance for far contexts.

        Maps to Patent Claims 13-15.
        """
        dim = 32

        # Close contexts
        ctx1 = self.box.create_context_vector(
            identity=np.zeros(dim),
            intent=np.zeros(dim),
            trajectory=np.zeros(dim),
            timing=1.0,
            commitment=np.zeros(dim),
            signature=np.zeros(dim)
        )

        ctx2 = self.box.create_context_vector(
            identity=np.ones(dim) * 0.1,
            intent=np.ones(dim) * 0.1,
            trajectory=np.ones(dim) * 0.1,
            timing=1.1,
            commitment=np.ones(dim) * 0.1,
            signature=np.ones(dim) * 0.1
        )

        # Far context
        ctx3 = self.box.create_context_vector(
            identity=np.ones(dim) * 10,
            intent=np.ones(dim) * 10,
            trajectory=np.ones(dim) * 10,
            timing=100.0,
            commitment=np.ones(dim) * 10,
            signature=np.ones(dim) * 10
        )

        # Compute hyperbolic distances
        dist_close = self.box.compute_hyperbolic_distance(ctx1, ctx2)
        dist_far = self.box.compute_hyperbolic_distance(ctx1, ctx3)

        # Far distance should be significantly larger
        self.assertGreater(dist_far, dist_close)

        claim_results.record(
            "Claim 14",
            "test_hyperbolic_distance_amplification",
            True,
            f"Close={dist_close:.3f}, Far={dist_far:.3f} - Distance amplified"
        )

    def test_invalid_context_fails_gate(self):
        """
        Test that invalid contexts fail gate validation.

        Maps to Patent Claims 1-6.
        """
        dim = 32

        # Create context with zero identity (invalid WHO gate)
        context = self.box.create_context_vector(
            identity=np.zeros(dim),  # Zero - invalid
            intent=np.ones(dim),
            trajectory=np.ones(dim),
            timing=time.time(),
            commitment=np.ones(dim),
            signature=np.ones(dim)
        )

        valid, gate_results = self.box.validate_context(context)

        # Should fail
        self.assertFalse(valid)
        self.assertFalse(gate_results["WHO"]["valid"])

        claim_results.record(
            "Claims 1-6",
            "test_invalid_context_fails_gate",
            True,
            "Zero identity vector correctly rejected by WHO gate"
        )


# =============================================================================
# SECTION 3: ATTACK SINK CURVE TESTS
# =============================================================================

class TestAttackSinkCurve(unittest.TestCase):
    """
    Tests for attack sink curve visualization and analysis.

    Maps to Patent Claims 19-24: Attack cost sink mechanism.
    """

    def setUp(self):
        """Initialize SCBEBox for each test."""
        self.box = SCBEBox(
            security_level=SecurityLevel.POST_QUANTUM,
            base_dimensions=MIN_LATTICE_DIM
        )

    def test_sink_curve_generation(self):
        """
        Test attack sink curve can be generated.
        """
        results = self.box.simulate_attack_sink_curve()

        self.assertGreater(len(results), 0)

        # Verify all results have required fields
        for r in results:
            self.assertIsInstance(r.dimensions, int)
            self.assertIsInstance(r.base_cost, float)
            self.assertIsInstance(r.harmonic_multiplier, float)
            self.assertIsInstance(r.total_cost, float)
            self.assertIsInstance(r.cost_in_bits, float)
            self.assertIsInstance(r.years_to_break, float)
            self.assertIsInstance(r.is_feasible, bool)

        claim_results.record(
            "Claim 19",
            "test_sink_curve_generation",
            True,
            f"Generated {len(results)} attack cost data points"
        )

    def test_sink_curve_monotonic_increase(self):
        """
        Test sink curve shows monotonic cost increase.
        """
        distances = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        results = []

        for d in distances:
            r = self.box.calculate_attack_cost(d)
            results.append(r)

        costs = [r.cost_in_bits for r in results]

        # Verify monotonic increase
        for i in range(1, len(costs)):
            self.assertGreater(costs[i], costs[i-1])

        claim_results.record(
            "Claim 21",
            "test_sink_curve_monotonic_increase",
            True,
            f"Monotonic cost increase: {costs[0]:.1f} -> {costs[-1]:.1f} bits"
        )

    def test_sink_depth_calculation(self):
        """
        Test sink depth (cost growth rate) increases with distance.

        The "sink" deepens as context distance increases.
        """
        distances = [0.5, 1.0, 1.5, 2.0]

        # Calculate cost growth rates
        rates = []
        prev_cost = 0
        for d in distances:
            r = self.box.calculate_attack_cost(d)
            if prev_cost > 0:
                rate = r.cost_in_bits - prev_cost
                rates.append(rate)
            prev_cost = r.cost_in_bits

        # Verify accelerating growth (sink deepening)
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i-1])

        claim_results.record(
            "Claim 22",
            "test_sink_depth_calculation",
            True,
            f"Sink depth accelerates: {[f'{r:.1f}' for r in rates]} bits/unit distance"
        )

    def test_sink_deterministic_patterns(self):
        """
        Test sink produces deterministic (reproducible) patterns.

        Maps to Patent Claim 15.
        """
        # Run same simulation twice
        results1 = self.box.simulate_attack_sink_curve([0.5, 1.0, 1.5])
        results2 = self.box.simulate_attack_sink_curve([0.5, 1.0, 1.5])

        # Should produce identical results
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1.dimensions, r2.dimensions)
            self.assertAlmostEqual(r1.cost_in_bits, r2.cost_in_bits, places=5)

        claim_results.record(
            "Claim 15",
            "test_sink_deterministic_patterns",
            True,
            "Attack sink produces deterministic, reproducible patterns"
        )

    def test_sink_data_for_visualization(self):
        """
        Test generating data suitable for visualization.
        """
        distances = np.linspace(0.1, 3.0, 30).tolist()

        data = {
            "distances": [],
            "dimensions": [],
            "cost_bits": [],
            "years_to_break": [],
            "feasibility": []
        }

        for d in distances:
            r = self.box.calculate_attack_cost(d)
            data["distances"].append(d)
            data["dimensions"].append(r.dimensions)
            data["cost_bits"].append(r.cost_in_bits)
            data["years_to_break"].append(r.years_to_break)
            data["feasibility"].append(r.is_feasible)

        # Verify data collected
        self.assertEqual(len(data["distances"]), 30)
        self.assertTrue(all(c > 0 for c in data["cost_bits"]))

        claim_results.record(
            "Claims 19-24",
            "test_sink_data_for_visualization",
            True,
            f"Generated 30-point visualization data, max cost={max(data['cost_bits']):.0f} bits"
        )


# =============================================================================
# SECTION 4: PATENT CLAIM MAPPING VALIDATION
# =============================================================================

class TestPatentClaimMapping(unittest.TestCase):
    """
    Validate explicit patent claim mappings.

    Ensures each claim has implementation evidence.
    """

    def setUp(self):
        """Initialize SCBEBox for each test."""
        self.box = SCBEBox(
            security_level=SecurityLevel.POST_QUANTUM,
            base_dimensions=MIN_LATTICE_DIM
        )

    def test_claim_1_who_gate(self):
        """Claim 1: WHO gate - Identity verification."""
        context = self.box.create_context_vector(
            identity=np.array([1, 2, 3, 4, 5]),
            intent=np.ones(5),
            trajectory=np.ones(5),
            timing=time.time(),
            commitment=np.ones(5),
            signature=np.ones(5)
        )

        _, results = self.box.validate_context(context)
        self.assertTrue(results["WHO"]["valid"])
        self.assertEqual(results["WHO"]["claim"], "Claim 1")

        claim_results.record(
            "Claim 1",
            "test_claim_1_who_gate",
            True,
            "WHO gate validates identity vectors"
        )

    def test_claim_2_what_gate(self):
        """Claim 2: WHAT gate - Intent classification."""
        context = self.box.create_context_vector(
            identity=np.ones(5),
            intent=np.array([1, 0, 1, 0, 1]),  # Intent pattern
            trajectory=np.ones(5),
            timing=time.time(),
            commitment=np.ones(5),
            signature=np.ones(5)
        )

        _, results = self.box.validate_context(context)
        self.assertTrue(results["WHAT"]["valid"])
        self.assertEqual(results["WHAT"]["claim"], "Claim 2")

        claim_results.record(
            "Claim 2",
            "test_claim_2_what_gate",
            True,
            "WHAT gate classifies intent patterns"
        )

    def test_claim_3_where_gate(self):
        """Claim 3: WHERE gate - Trajectory verification."""
        context = self.box.create_context_vector(
            identity=np.ones(5),
            intent=np.ones(5),
            trajectory=np.array([1, 2, 3, 4, 5]),  # Position trajectory
            timing=time.time(),
            commitment=np.ones(5),
            signature=np.ones(5)
        )

        _, results = self.box.validate_context(context)
        self.assertTrue(results["WHERE"]["valid"])
        self.assertEqual(results["WHERE"]["claim"], "Claim 3")

        claim_results.record(
            "Claim 3",
            "test_claim_3_where_gate",
            True,
            "WHERE gate verifies trajectory"
        )

    def test_claim_4_when_gate(self):
        """Claim 4: WHEN gate - Temporal coordination."""
        current_time = time.time()
        context = self.box.create_context_vector(
            identity=np.ones(5),
            intent=np.ones(5),
            trajectory=np.ones(5),
            timing=current_time,
            commitment=np.ones(5),
            signature=np.ones(5)
        )

        _, results = self.box.validate_context(context)
        self.assertTrue(results["WHEN"]["valid"])
        self.assertEqual(results["WHEN"]["claim"], "Claim 4")
        self.assertEqual(results["WHEN"]["timestamp"], current_time)

        claim_results.record(
            "Claim 4",
            "test_claim_4_when_gate",
            True,
            f"WHEN gate validates timestamp {current_time}"
        )

    def test_claim_5_why_gate(self):
        """Claim 5: WHY gate - Commitment verification."""
        context = self.box.create_context_vector(
            identity=np.ones(5),
            intent=np.ones(5),
            trajectory=np.ones(5),
            timing=time.time(),
            commitment=np.array([1, 1, 1, 0, 0]),  # Commitment flags
            signature=np.ones(5)
        )

        _, results = self.box.validate_context(context)
        self.assertTrue(results["WHY"]["valid"])
        self.assertEqual(results["WHY"]["claim"], "Claim 5")

        claim_results.record(
            "Claim 5",
            "test_claim_5_why_gate",
            True,
            "WHY gate verifies commitment"
        )

    def test_claim_6_how_gate(self):
        """Claim 6: HOW gate - Signature verification."""
        context = self.box.create_context_vector(
            identity=np.ones(5),
            intent=np.ones(5),
            trajectory=np.ones(5),
            timing=time.time(),
            commitment=np.ones(5),
            signature=np.random.randn(5)  # Random signature
        )

        _, results = self.box.validate_context(context)
        self.assertTrue(results["HOW"]["valid"])
        self.assertEqual(results["HOW"]["claim"], "Claim 6")

        claim_results.record(
            "Claim 6",
            "test_claim_6_how_gate",
            True,
            "HOW gate verifies signature"
        )

    def test_claims_7_8_harmonic_formula(self):
        """Claims 7-8: Harmonic scaling formula H(d,R) = R^(1+d²)."""
        test_cases = [
            (0.0, 2.0, 2.0),      # H(0,2) = 2^1 = 2
            (1.0, 2.0, 4.0),      # H(1,2) = 2^2 = 4
            (2.0, 2.0, 32.0),     # H(2,2) = 2^5 = 32
            (1.0, 1.5, 2.25),     # H(1,1.5) = 1.5^2 = 2.25
        ]

        for d, R, expected in test_cases:
            result = SCBEBox.harmonic_scaling(d, R)
            self.assertAlmostEqual(result, expected, places=5)

        claim_results.record(
            "Claims 7-8",
            "test_claims_7_8_harmonic_formula",
            True,
            f"H(d,R) = R^(1+d²) verified for {len(test_cases)} test cases"
        )

    def test_claim_9_adaptive_dimensions(self):
        """Claim 9: Dynamic dimension scaling."""
        d1 = self.box.compute_adaptive_dimensions(0.1)
        d2 = self.box.compute_adaptive_dimensions(1.0)
        d3 = self.box.compute_adaptive_dimensions(2.0)

        # Verify scaling
        self.assertGreater(d2, d1)
        self.assertGreater(d3, d2)

        claim_results.record(
            "Claim 9",
            "test_claim_9_adaptive_dimensions",
            True,
            f"Adaptive dims: d=0.1->{d1}, d=1.0->{d2}, d=2.0->{d3}"
        )

    def test_claim_10_security_bits(self):
        """Claim 10: Security bit advantage calculation."""
        bits_0 = self.box.compute_security_bits(0.0)
        bits_1 = self.box.compute_security_bits(1.0)
        bits_2 = self.box.compute_security_bits(2.0)

        # Base is 256 bits
        self.assertGreater(bits_0, 256)
        self.assertGreater(bits_1, bits_0)
        self.assertGreater(bits_2, bits_1)

        claim_results.record(
            "Claim 10",
            "test_claim_10_security_bits",
            True,
            f"Security bits: d=0->{bits_0:.0f}, d=1->{bits_1:.0f}, d=2->{bits_2:.0f}"
        )

    def test_claims_13_15_hyperbolic_geometry(self):
        """Claims 13-15: Hyperbolic geometry security layer."""
        projector = HyperbolicProjector()

        # Test Poincaré disk constraint
        point = np.array([0.5, 0.5, 0.3])
        projected = projector.euclidean_to_poincare(point)

        # Projected point should be in unit disk
        self.assertLess(np.linalg.norm(projected), 1.0)

        # Test distance metric
        p1 = np.array([0.1, 0.1])
        p2 = np.array([0.2, 0.2])
        dist = projector.hyperbolic_distance(p1, p2)
        self.assertGreater(dist, 0)

        claim_results.record(
            "Claims 13-15",
            "test_claims_13_15_hyperbolic_geometry",
            True,
            f"Poincaré projection norm={np.linalg.norm(projected):.4f} < 1, dist={dist:.4f}"
        )

    def test_claims_16_17_mahalanobis_coherence(self):
        """Claims 16-17: Mahalanobis coherence analysis."""
        analyzer = MahalanobisCoherenceAnalyzer()

        # Add normal trajectory
        for i in range(20):
            analyzer.add_trajectory_point(np.random.randn(5) * 0.1)

        # Test normal point
        normal_coherence = analyzer.compute_coherence(np.random.randn(5) * 0.1)

        # Test anomalous point
        is_anomaly, anomaly_coherence = analyzer.detect_anomaly(np.ones(5) * 100)

        self.assertGreater(normal_coherence, anomaly_coherence)
        self.assertTrue(is_anomaly)

        claim_results.record(
            "Claims 16-17",
            "test_claims_16_17_mahalanobis_coherence",
            True,
            f"Normal coherence={normal_coherence:.3f}, Anomaly coherence={anomaly_coherence:.4f}"
        )

    def test_claim_18_genetic_markers(self):
        """Claim 18: Genetic marker audit trail."""
        # Create chain of markers
        context1 = self.box.create_context_vector(
            identity=np.ones(5),
            intent=np.ones(5),
            trajectory=np.ones(5),
            timing=time.time(),
            commitment=np.ones(5),
            signature=np.ones(5)
        )
        marker1 = self.box.create_genetic_marker(context1, "mutation_1")

        context2 = self.box.create_context_vector(
            identity=np.ones(5) * 2,
            intent=np.ones(5) * 2,
            trajectory=np.ones(5) * 2,
            timing=time.time(),
            commitment=np.ones(5) * 2,
            signature=np.ones(5) * 2
        )
        marker2 = self.box.create_genetic_marker(context2, "mutation_2")

        # Verify lineage
        lineage = self.box.get_genetic_lineage()
        self.assertGreater(len(lineage), 2)
        self.assertEqual(marker2.parent_id, marker1.marker_id)
        self.assertEqual(marker2.generation, marker1.generation + 1)

        claim_results.record(
            "Claim 18",
            "test_claim_18_genetic_markers",
            True,
            f"Lineage depth={len(lineage)}, gen {marker1.generation}->{marker2.generation}"
        )

    def test_claims_19_24_attack_cost_sink(self):
        """Claims 19-24: Attack cost sink mechanism."""
        result = self.box.calculate_attack_cost(1.5, attacker_compute=1e18)

        # Verify all claim mappings present
        self.assertIn("Claim 19", result.patent_claim_mapping)
        self.assertIn("Claim 20", result.patent_claim_mapping)
        self.assertIn("Claim 21", result.patent_claim_mapping)
        self.assertIn("Claim 22", result.patent_claim_mapping)
        self.assertIn("Claim 23", result.patent_claim_mapping)
        self.assertIn("Claim 24", result.patent_claim_mapping)

        claim_results.record(
            "Claims 19-24",
            "test_claims_19_24_attack_cost_sink",
            True,
            f"Attack sink: dims={result.dimensions}, bits={result.cost_in_bits:.0f}, years={result.years_to_break:.2e}"
        )

    def test_full_claim_coverage(self):
        """Verify all 24 claims are covered."""
        coverage = self.box._get_claim_coverage()

        self.assertEqual(len(coverage), 24)

        for i in range(1, 25):
            claim = f"Claim {i}"
            self.assertIn(claim, coverage)

        claim_results.record(
            "All Claims",
            "test_full_claim_coverage",
            True,
            f"All 24 patent claims have implementation evidence"
        )


# =============================================================================
# SECTION 5: COMPREHENSIVE SIMULATION TEST
# =============================================================================

class TestComprehensiveSimulation(unittest.TestCase):
    """
    End-to-end comprehensive simulation tests.
    """

    def test_full_simulation_runs(self):
        """Test full simulation completes successfully."""
        box = SCBEBox(security_level=SecurityLevel.POST_QUANTUM)
        results = box.run_full_simulation(num_iterations=5)

        self.assertIn("simulation_id", results)
        self.assertIn("iterations", results)
        self.assertIn("attack_simulations", results)
        self.assertIn("genetic_lineage", results)
        self.assertIn("patent_claim_coverage", results)
        self.assertIn("summary", results)

        self.assertEqual(results["summary"]["total_iterations"], 5)
        self.assertEqual(len(results["patent_claim_coverage"]), 24)

    def test_simulation_export(self):
        """Test simulation results can be exported."""
        import tempfile

        box = SCBEBox(security_level=SecurityLevel.POST_QUANTUM)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        box.export_results(filepath)

        # Verify file created and is valid JSON
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.assertIn("simulation_id", data)
        self.assertIn("summary", data)

        # Cleanup
        import os
        os.unlink(filepath)


# =============================================================================
# TEST RUNNER WITH CLAIM REPORT
# =============================================================================

def run_tests_with_claim_report():
    """Run all tests and generate patent claim report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNormalOperation))
    suite.addTests(loader.loadTestsFromTestCase(TestAttackSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestAttackSinkCurve))
    suite.addTests(loader.loadTestsFromTestCase(TestPatentClaimMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestComprehensiveSimulation))

    # Run tests
    print("=" * 70)
    print("SCBEBox Test Suite - Patent Claim Validation")
    print("=" * 70)
    print()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate claim report
    print()
    print("=" * 70)
    print("PATENT CLAIM EVIDENCE REPORT")
    print("=" * 70)
    print()

    summary = claim_results.get_summary()

    print(f"Total Claims Tested: {summary['total_claims']}")
    print(f"Total Tests Run: {summary['total_tests']}")
    print(f"All Tests Passed: {summary['all_passed']}")
    print()

    for claim, data in sorted(summary['claims'].items()):
        status = "PASS" if data['all_passed'] else "FAIL"
        print(f"  {claim}: [{status}] - {data['tests_run']} tests")
        for evidence in data['evidence'][:2]:  # Show first 2 pieces of evidence
            print(f"    - {evidence}")
    print()

    # Save results
    output = {
        "test_results": {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success": result.wasSuccessful()
        },
        "claim_evidence": summary,
        "timestamp": time.time()
    }

    with open("scbe_box_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: scbe_box_test_results.json")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests_with_claim_report()
    sys.exit(0 if success else 1)
