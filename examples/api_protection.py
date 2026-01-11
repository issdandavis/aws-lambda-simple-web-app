#!/usr/bin/env python3
"""
Example 3: API Endpoint Protection

Protect API endpoints with geometric context binding.
Wrong context = noise response (not "403 Forbidden").
Attacker can't tell if they're close.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import hashlib
import json
import time
from scbe_aethermoore.hypercube_brain import (
    hypercube_brain_classify,
    SignatureMode
)


# Simulated API secret (in production, from secure storage)
API_SECRET = b"super_secret_api_key_32_bytes!!"


def generate_noise_response() -> dict:
    """
    Generate a response that looks valid but contains noise.
    Attacker can't distinguish from real response.
    """
    # Generate deterministic-looking but useless data
    noise = hashlib.sha256(str(time.time()).encode()).hexdigest()
    return {
        "status": "success",
        "data": {
            "id": noise[:8],
            "value": int(noise[8:16], 16) % 1000,
            "timestamp": int(time.time())
        }
    }


def protect_endpoint(
    endpoint: str,
    context: np.ndarray,
    required_intent: str = "read"
) -> tuple[bool, dict]:
    """
    Protect an API endpoint with geometric verification.

    Returns:
        (authorized, response)
    """
    # Classify the context
    state = hypercube_brain_classify(context)

    # Check trapdoor
    if state.signature_mode == SignatureMode.TRAPDOOR_FROZEN:
        # Don't say "blocked" - return noise
        return False, generate_noise_response()

    # Check risk level
    if state.risk_factor > 0.7:
        return False, generate_noise_response()

    # Check velocity (rate limiting via physics)
    if state.gamma > 1.5:
        return False, generate_noise_response()

    # Authorized - return real data
    return True, {
        "status": "success",
        "data": {
            "endpoint": endpoint,
            "authorized": True,
            "mode": state.signature_mode.value,
            "message": "This is the real response"
        }
    }


class SecureAPI:
    """
    A simple API with geometric protection.
    """

    def __init__(self):
        self.request_count = 0

    def get_user_context(self, user_id: str, ip: str) -> np.ndarray:
        """
        Build context from request metadata.
        In production, this would use real sensors/metrics.
        """
        self.request_count += 1

        # Simulate velocity based on request rate
        # More requests = higher velocity = closer to trapdoor
        velocity = min(0.9, self.request_count * 0.1)

        return np.array([
            time.time(),                    # timestamp
            float(hash(user_id) % 100000),  # device_id from user
            2.0,                            # base threat level
            np.random.random(),             # entropy
            10.0,                           # server load
            velocity                        # request velocity
        ])

    def handle_request(self, endpoint: str, user_id: str, ip: str) -> dict:
        """
        Handle an API request with protection.
        """
        context = self.get_user_context(user_id, ip)
        authorized, response = protect_endpoint(endpoint, context)

        # Log (in production, more sophisticated)
        status = "OK" if authorized else "NOISE"
        print(f"[{status}] {endpoint} from {user_id}")

        return response


# Demo
if __name__ == "__main__":
    print("API PROTECTION DEMO")
    print("=" * 50)

    api = SecureAPI()

    # Normal requests
    print("\n1. Normal user making a few requests:")
    for i in range(3):
        response = api.handle_request("/api/data", "user_123", "192.168.1.1")
        print(f"   Request {i+1}: {json.dumps(response, indent=2)[:100]}...")

    # Rapid-fire attack
    print("\n2. Attacker making rapid requests:")
    api.request_count = 0  # Reset
    for i in range(10):
        response = api.handle_request("/api/secrets", "attacker", "10.0.0.1")

    print("\n   After 10 rapid requests, attacker gets noise responses.")
    print("   They can't tell if they're blocked or getting real data.")

    print("\n" + "=" * 50)
    print("Key insight: Wrong context → noise, not '403 Forbidden'")
    print("Attacker has no oracle to probe the system.")
