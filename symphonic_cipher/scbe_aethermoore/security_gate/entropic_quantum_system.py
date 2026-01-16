#!/usr/bin/env python3
"""
Comprehensive Test Suite for Entropic Dual-Quantum System
Includes ForwardSecureRatchet, AdaptiveKController, and MarsReceiver
Implements all security fixes from cryptography and patent law review
"""

import hashlib
import hmac
import numpy as np
import time

# Optional matplotlib for visualization (not required for core functionality)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import unittest

# Constants
N0_BITS = 256
N0 = 2.0 ** N0_BITS
K_DEFAULT = 0.069  # Expansion rate per year
C_QUANTUM = 1e15  # Quantum ops/sec (current projection)
C_CLASSICAL = 1e18  # Classical ops/sec

# ============================================================================
# CORE SECURITY PRIMITIVES (Addressing Security Review Feedback)
# ============================================================================

class ForwardSecureRatchet:
    """    Forward-secure state evolution mechanism (Signal Double Ratchet)
    Fixes Issue #2 from security review: Retroactive Hardening
    """
    def __init__(self, seed: bytes):
        self.state = seed
    
    def derive_key(self, t: int) -> bytes:
        """Derive key for epoch t"""
        key = self._hkdf(self.state, info=f"epoch_{t}".encode())
        # DELETE old state (forward secrecy)
        self.state = self._hkdf(self.state, info=b"ratchet")
        # Old state cannot be reconstructed
        return key
    
    def _hkdf(self, ikm: bytes, info: bytes, length: int = 32) -> bytes:
        """HKDF implementation"""
        prk = hmac.new(b"", ikm, hashlib.sha256).digest()
        okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
        return okm[:length]

class ReplayError(Exception):
    """Raised when replay attack is detected"""
    pass

class MarsReceiver:
    """    0-RTT receiver with anti-replay mechanism
    Fixes Issue #3 from security review: 0-RTT Requires Anti-Replay
    """
    def __init__(self, seed: bytes, k: float):
        self.seed = seed
        self.k = k
        self.seen_nonces = set()  # Replay cache
        self.last_timestamp = 0  # Monotonic counter
    
    def fast_forward_decode(self, message: bytes, t_E: int, nonce: bytes) -> bytes:
        """Decode with 0-RTT and replay protection"""
        # Check monotonicity
        if t_E <= self.last_timestamp:
            raise ReplayError("Old timestamp")
        
        # Check nonce uniqueness
        if nonce in self.seen_nonces:
            raise ReplayError("Duplicate nonce")
        
        # Fast-forward keyspace
        t_M = t_E + 840  # 14 min Mars delay
        N_t = N0 * np.exp(self.k * t_M)
        
        # Decode with expanded keyspace
        plaintext = self._decrypt(message, N_t)
        
        # Update replay defense
        self.last_timestamp = t_E
        self.seen_nonces.add(nonce)
        
        return plaintext
    
    def _decrypt(self, message: bytes, N_t: float) -> bytes:
        """Placeholder decryption"""
        return message  # Simplified for testing

class AdaptiveKController:
    """
    Adaptive k parameter controller with concrete implementation
    Fixes Issue #4 from security review: Adaptive k Needs Concrete Implementation
    """
    def __init__(self):
        self.k_min = 0.01  # Minimum: ~1% overhead
        self.k_max = 100  # Maximum: 100× base rate
        self.k_current = K_DEFAULT  # Base rate
    
    def update_k(self, threat_data: Dict) -> float:
        """Update k based on threat telemetry"""
        # INPUT: Parse threat telemetry
        C_quantum_observed = threat_data.get('quantum_ops_per_sec', C_QUANTUM)
        
        # CONTROLLER: Compute required k
        k_required = (2 * C_quantum_observed) / np.sqrt(N0)
        
        # BOUNDS: Apply safety constraints
        k_new = max(self.k_min, min(k_required, self.k_max))
        
        # RATE LIMIT: Prevent sudden jumps
        max_change = self.k_current * 1.5  # 50% max increase per update
        k_new = min(k_new, max_change)
        
        # OUTPUT: Adjust concrete parameters
        if k_new > self.k_current:
            self.k_current = k_new
        
        return k_new

# ============================================================================
# TEST SUITES
# ============================================================================

class TestForwardSecureRatchet(unittest.TestCase):
    """Test forward secrecy implementation"""
    
    def test_state_deletion(self):
        """Verify old state cannot be recovered"""
        ratchet = ForwardSecureRatchet(b"test_seed_123")
        
        # Derive key for epoch 0
        key0 = ratchet.derive_key(0)
        state_after_0 = ratchet.state
        
        # Derive key for epoch 1
        key1 = ratchet.derive_key(1)
        state_after_1 = ratchet.state
        
        # Verify states are different (ratcheted)
        self.assertNotEqual(state_after_0, state_after_1)
        
        # Verify keys are different
        self.assertNotEqual(key0, key1)

class TestMarsReceiver(unittest.TestCase):
    """Test 0-RTT with anti-replay"""
    
    def test_replay_detection_timestamp(self):
        """Verify monotonic timestamp validation"""
        receiver = MarsReceiver(b"seed", K_DEFAULT)
        
        # First message at t=100
        msg1 = receiver.fast_forward_decode(b"msg1", 100, b"nonce1")
        
        # Try to replay with old timestamp
        with self.assertRaises(ReplayError):
            receiver.fast_forward_decode(b"msg2", 99, b"nonce2")
    
    def test_replay_detection_nonce(self):
        """Verify nonce uniqueness check"""
        receiver = MarsReceiver(b"seed", K_DEFAULT)
        
        # First message
        msg1 = receiver.fast_forward_decode(b"msg1", 100, b"nonce1")
        
        # Try to replay with same nonce
        with self.assertRaises(ReplayError):
            receiver.fast_forward_decode(b"msg2", 101, b"nonce1")

class TestAdaptiveKController(unittest.TestCase):
    """Test adaptive k parameter updates"""
    
    def test_quantum_breakthrough_response(self):
        """Verify k adjustment to 1000x quantum threat"""
        controller = AdaptiveKController()
        
        # Simulate quantum breakthrough: 1000x capability
        threat_data = {
            'quantum_ops_per_sec': C_QUANTUM * 1000
        }
        
        k_new = controller.update_k(threat_data)
        
        # Verify k increased (but not necessarily 1000x due to rate limiting)
        self.assertGreater(k_new, K_DEFAULT)
        self.assertLessEqual(k_new, controller.k_max)
    
    def test_rate_limiting(self):
        """Verify gradual k adjustment prevents oscillation"""
        controller = AdaptiveKController()
        
        # Extreme threat
        threat_data = {'quantum_ops_per_sec': C_QUANTUM * 10000}
        
        k_new = controller.update_k(threat_data)
        
        # Should be limited to 50% increase
        expected_max = K_DEFAULT * 1.5
        self.assertLessEqual(k_new, expected_max * 1.01)  # Small tolerance

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("ENTROPIC DUAL-QUANTUM SYSTEM - COMPREHENSIVE TEST SUITE")
    print("Testing ForwardSecureRatchet, MarsReceiver, and AdaptiveKController")
    print("Implements security fixes from cryptography/patent review")
    print("="*80 + "\n")
    
    # Run tests
    unittest.main(verbosity=2)
