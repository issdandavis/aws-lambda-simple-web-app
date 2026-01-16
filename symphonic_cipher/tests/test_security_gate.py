"""
Tests for the Security Gate Module
(Integrated from scbe-security-gate repository)
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scbe_aethermoore.security_gate import (
    # Computational Immune System
    ScienceDomain,
    MultidimensionalExpertMind,
    ContextVector,
    ErrorAccumulator,
    DriftSimulationEngine,

    # DNA Multi-Layer Encoding
    MORSE_TO_DNA,
    DNA_TO_MORSE,
    encode_morse_dna,
    TemporalVector,
    EmotionalIntentVector,
    SpatialVector,
    DNAMultiLayerMessage,

    # Entropic Dual-Quantum System
    N0_BITS,
    N0,
    K_DEFAULT,
    ForwardSecureRatchet,
    ReplayError,
    MarsReceiver,
    AdaptiveKController,
)


class TestComputationalImmuneSystem:
    """Tests for the Computational Immune System module."""

    def test_science_domains_exist(self):
        """All science domains should be defined."""
        assert ScienceDomain.PHYSICS is not None
        assert ScienceDomain.CHEMISTRY is not None
        assert ScienceDomain.BIOLOGY is not None
        assert ScienceDomain.MATHEMATICS is not None
        assert ScienceDomain.INFORMATION is not None

    def test_multidimensional_expert_mind_init(self):
        """Expert mind should initialize with all domains."""
        mind = MultidimensionalExpertMind()
        assert len(mind.domains) == 5
        assert mind.linguistic_drift_rate == 0.026

    def test_simulate_drift(self):
        """Drift simulation should return valid statistics."""
        mind = MultidimensionalExpertMind()
        stats = mind.simulate_drift(iterations=100)

        assert 'mean_total_drift' in stats
        assert 'std_total_drift' in stats
        assert 'max_drift' in stats
        assert 'min_drift' in stats
        assert stats['mean_total_drift'] >= 0

    def test_context_vector_creation(self):
        """Context vectors should be creatable with 5W1H."""
        ctx = ContextVector(
            who="RESEARCHER",
            what="test_action",
            when=time.time(),
            where=(47.6, -122.3, 0.0),
            why="security_test",
            how="api_call"
        )

        assert ctx.who == "RESEARCHER"
        assert ctx.why == "security_test"

    def test_context_vector_to_vector(self):
        """Context should convert to numerical vector."""
        ctx = ContextVector(
            who="RESEARCHER",
            what="test_action",
            when=time.time(),
            where=(47.6, -122.3, 0.0),
            why="security_test",
            how="api_call"
        )

        vec = ctx.to_vector()
        assert len(vec) == 8
        assert isinstance(vec, np.ndarray)

    def test_context_distance(self):
        """Context distance should be computable."""
        ctx1 = ContextVector("RESEARCHER", "action1", 1000, (0, 0, 0), "test", "api")
        ctx2 = ContextVector("WRITER", "action2", 2000, (1, 1, 1), "prod", "queue")

        distance = ctx1.context_distance(ctx2)
        assert distance >= 0

    def test_error_accumulator(self):
        """Error accumulator should track cumulative error."""
        acc = ErrorAccumulator()

        assert acc.cumulative_error == 0.0

        acc.add_operation_error('hash', 1.0)
        assert acc.cumulative_error > 0

        acc.add_operation_error('encrypt', 2.0)
        assert acc.cumulative_error > 0
        assert len(acc.error_history) == 2


class TestDNAMultiLayerEncoding:
    """Tests for the DNA Multi-Layer Encoding module."""

    def test_morse_dna_mapping(self):
        """Morse to DNA mapping should be consistent."""
        assert len(MORSE_TO_DNA) == 4
        assert len(DNA_TO_MORSE) == 4

        # Round-trip
        for morse, dna in MORSE_TO_DNA.items():
            assert DNA_TO_MORSE[dna] == morse

    def test_temporal_vector(self):
        """Temporal vectors should compute distance."""
        tv1 = TemporalVector(
            past=-0.5,
            present=0.5,
            future=1.5,
            instruction_time=1000,
            receipt_time=2000
        )

        tv2 = TemporalVector(
            past=-0.2,
            present=0.8,
            future=1.2,
            instruction_time=1500,
            receipt_time=2500
        )

        distance = tv1.compute_temporal_distance(tv2)
        assert distance >= 0

    def test_emotional_intent_vector(self):
        """Emotional intent vectors should compute gravitational force."""
        ev1 = EmotionalIntentVector(
            anchor_strength=0.8,
            bridge_strength=0.6,
            cut_strength=0.1,
            paradox_strength=0.2,
            joy_strength=0.4,
            harmony_strength=0.5,
            sender_intent="system_admin",
            purpose="security_test",
            method="api_call"
        )

        ev2 = EmotionalIntentVector(
            anchor_strength=0.5,
            bridge_strength=0.5,
            cut_strength=0.5,
            paradox_strength=0.5,
            joy_strength=0.5,
            harmony_strength=0.5,
            sender_intent="user",
            purpose="query",
            method="web"
        )

        distance = ev1.compute_emotional_distance(ev2)
        assert distance >= 0

        force = ev1.compute_gravitational_force(ev2, distance=1.0, G=1e-5)
        assert force >= 0

    def test_spatial_vector(self):
        """Spatial vectors should compute 3D distance."""
        sv1 = SpatialVector(x=0, y=0, z=0, node_id=1, hop_count=1)
        sv2 = SpatialVector(x=3, y=4, z=0, node_id=2, hop_count=2)

        distance = sv1.distance_to(sv2)
        assert distance == 5.0  # 3-4-5 triangle

    def test_dna_multi_layer_message_complexity(self):
        """DNA multi-layer message should compute total complexity."""
        msg = DNAMultiLayerMessage(
            plaintext="TEST",
            morse_encoded="- . ... -",
            dna_encoded="TATT",
            temporal=TemporalVector(-0.5, 0.7, 1.5, 1000, 2000),  # Non-zero present
            emotional=EmotionalIntentVector(0.8, 0.6, 0.1, 0.2, 0.4, 0.5, "admin", "security", "api"),
            spatial=SpatialVector(47.6, -122.3, 0, 1, 5),  # Non-zero hop_count
            cipher_pattern="[Vel][root]['ar][object][medial-link][temporal]",
            language="Anchor"
        )

        complexity = msg.compute_total_complexity()
        assert complexity > 0


class TestEntropicDualQuantumSystem:
    """Tests for the Entropic Dual-Quantum System module."""

    def test_constants(self):
        """Constants should have expected values."""
        assert N0_BITS == 256
        assert N0 == 2.0 ** 256
        assert K_DEFAULT == 0.069

    def test_forward_secure_ratchet_state_deletion(self):
        """Forward secure ratchet should delete old state."""
        ratchet = ForwardSecureRatchet(b"test_seed_123")

        key0 = ratchet.derive_key(0)
        state_after_0 = ratchet.state

        key1 = ratchet.derive_key(1)
        state_after_1 = ratchet.state

        # States should be different (ratcheted)
        assert state_after_0 != state_after_1
        assert key0 != key1

    def test_forward_secure_ratchet_determinism(self):
        """Same seed should produce same keys."""
        r1 = ForwardSecureRatchet(b"same_seed")
        r2 = ForwardSecureRatchet(b"same_seed")

        assert r1.derive_key(0) == r2.derive_key(0)

    def test_mars_receiver_replay_detection_timestamp(self):
        """Mars receiver should detect timestamp replay."""
        receiver = MarsReceiver(b"seed", K_DEFAULT)

        # First message at t=100
        receiver.fast_forward_decode(b"msg1", 100, b"nonce1")

        # Try to replay with old timestamp
        with pytest.raises(ReplayError):
            receiver.fast_forward_decode(b"msg2", 99, b"nonce2")

    def test_mars_receiver_replay_detection_nonce(self):
        """Mars receiver should detect nonce replay."""
        receiver = MarsReceiver(b"seed", K_DEFAULT)

        # First message
        receiver.fast_forward_decode(b"msg1", 100, b"nonce1")

        # Try to replay with same nonce
        with pytest.raises(ReplayError):
            receiver.fast_forward_decode(b"msg2", 101, b"nonce1")

    def test_adaptive_k_controller_init(self):
        """Adaptive k controller should initialize with bounds."""
        controller = AdaptiveKController()

        assert controller.k_min == 0.01
        assert controller.k_max == 100
        assert controller.k_current == K_DEFAULT

    def test_adaptive_k_controller_update(self):
        """Adaptive k controller should adjust k based on threat."""
        controller = AdaptiveKController()

        # Simulate quantum breakthrough
        threat_data = {'quantum_ops_per_sec': 1e18}
        k_new = controller.update_k(threat_data)

        assert k_new >= controller.k_min
        assert k_new <= controller.k_max

    def test_adaptive_k_controller_rate_limiting(self):
        """Adaptive k controller should rate-limit changes."""
        controller = AdaptiveKController()

        # Extreme threat
        threat_data = {'quantum_ops_per_sec': 1e30}
        k_new = controller.update_k(threat_data)

        # Should be limited to 50% increase
        expected_max = K_DEFAULT * 1.5
        assert k_new <= expected_max * 1.01  # Small tolerance


class TestModuleIntegration:
    """Integration tests for security gate module."""

    def test_import_from_main_module(self):
        """Components should be importable from main scbe_aethermoore."""
        from scbe_aethermoore import (
            ForwardSecureRatchet,
            MarsReceiver,
            ContextVector,
            DriftSimulationEngine,
        )

        assert ForwardSecureRatchet is not None
        assert MarsReceiver is not None
        assert ContextVector is not None
        assert DriftSimulationEngine is not None

    def test_security_gate_submodule_import(self):
        """security_gate should be importable as submodule."""
        from scbe_aethermoore import security_gate

        assert security_gate.__version__ == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
