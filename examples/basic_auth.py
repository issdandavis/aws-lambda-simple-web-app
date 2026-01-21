#!/usr/bin/env python3
"""
Example 1: Basic Context-Bound Authentication

This shows the simplest use case: verifying that a request
comes from the expected context.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scbe_aethermoore.hypercube_brain import (
    hypercube_brain_classify,
    SignatureMode
)


def authenticate_request(
    timestamp: float,
    device_id: int,
    threat_level: float,
    user_velocity: float
) -> tuple[bool, str]:
    """
    Authenticate a request based on its context.

    Returns:
        (authorized, reason)
    """
    # Build context vector
    context = np.array([
        timestamp,
        float(device_id),
        threat_level,
        0.5,  # entropy (could be from session)
        10.0, # server load (could be real metric)
        user_velocity  # How fast is user making requests?
    ])

    # Classify geometrically
    state = hypercube_brain_classify(context)

    # Check for trapdoor (too fast = attack)
    if state.signature_mode == SignatureMode.TRAPDOOR_FROZEN:
        return False, f"BLOCKED: Velocity too high (γ={state.gamma:.2f})"

    # Check risk level
    if state.risk_factor > 0.8:
        return False, f"BLOCKED: Risk too high ({state.risk_factor:.2f})"

    # Authorized
    mode = "internal" if state.signature_mode == SignatureMode.KYBER_INTERNAL else "external"
    return True, f"AUTHORIZED: {mode} operation (γ={state.gamma:.2f})"


# Demo
if __name__ == "__main__":
    print("BASIC AUTHENTICATION DEMO")
    print("=" * 50)

    # Normal user
    ok, reason = authenticate_request(
        timestamp=1704700000.0,
        device_id=12345,
        threat_level=2.0,
        user_velocity=0.3
    )
    print(f"\nNormal user: {reason}")

    # User making requests too fast (possible attack)
    ok, reason = authenticate_request(
        timestamp=1704700000.0,
        device_id=12345,
        threat_level=2.0,
        user_velocity=0.95  # Very fast!
    )
    print(f"Fast user:   {reason}")

    # High threat context
    ok, reason = authenticate_request(
        timestamp=1704700000.0,
        device_id=12345,
        threat_level=9.5,  # High threat
        user_velocity=0.3
    )
    print(f"High threat: {reason}")
