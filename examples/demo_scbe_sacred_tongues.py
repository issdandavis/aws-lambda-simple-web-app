#!/usr/bin/env python3
"""
SCBE-AETHERMOORE + Sacred Tongues Demo
======================================
Integrates hyperbolic geometry (SCBE), geometric trust (GeoSeal),
and cryptolinguistic encoding (Spiralverse Sacred Tongues).

Demonstrates:
- Encoding a nonce to spell-text
- SS1 blob formatting
- SCBE verification with geometric trust
- Roundtable consensus simulation

Requirements: numpy (from requirements.txt)

Usage:
    python examples/demo_scbe_sacred_tongues.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Import Sacred Tongue Tokenizer
from symphonic_cipher.spiralverse.tongues.tokenizer import (
    SacredTongueTokenizer,
    encode_to_spelltext,
    format_ss1_blob,
)

# Golden Ratio
PHI = (1 + np.sqrt(5)) / 2


def hyperbolic_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute hyperbolic distance in the Poincaré ball model.

    d_H(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
    """
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    # Clamp to interior of ball
    if u_norm >= 1.0 - eps:
        u = u * ((1.0 - eps) / u_norm)
    if v_norm >= 1.0 - eps:
        v = v * ((1.0 - eps) / v_norm)

    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    diff_norm_sq = np.linalg.norm(u - v) ** 2
    denom = (1 - u_norm**2) * (1 - v_norm**2)

    return float(np.arccosh(max(1.0, 1 + 2 * diff_norm_sq / denom)))


def harmonic_scaling(d_star: float, alpha: float = 10.0, beta: float = 0.5) -> float:
    """
    Bounded harmonic scaling: H(d*) = 1 + α·tanh(β·d*)

    This provides exponential-like growth that saturates,
    preventing overflow while maintaining the security property.
    """
    return float(1 + alpha * np.tanh(beta * d_star))


def geoseal_distance(context: np.ndarray, policy: np.ndarray) -> Tuple[float, str]:
    """
    GeoSeal Dual-Space Manifold distance.

    Projects context to sphere S^n and policy to hypercube [0,1]^m,
    then computes alignment distance.

    Returns:
        (distance, path) where path is 'interior' or 'exterior'
    """
    # Sphere projection (normalize to unit sphere)
    norm = np.linalg.norm(context) or 1.0
    sphere = context / norm

    # Hypercube projection (clamp to [0,1])
    hypercube = np.clip(policy, 0, 1)

    # Map sphere [-1,1] to [0,1] for comparison
    sphere_normalized = (sphere + 1) / 2

    # Euclidean distance in normalized space
    distance = float(np.linalg.norm(sphere_normalized[:len(hypercube)] - hypercube))

    # Interior threshold (default 0.5)
    path = 'interior' if distance < 0.5 else 'exterior'

    return distance, path


@dataclass
class AuthorizationResult:
    """Result of SCBE authorization verification."""
    decision: str               # ALLOW, QUARANTINE, DENY, SNAP
    risk_score: float          # Normalized risk [0, 1]
    hyperbolic_distance: float # d_H from trusted center
    harmonic_factor: float     # H(d_H)
    geoseal_path: str          # interior or exterior
    time_dilation: float       # τ = exp(-γ·r)
    consensus_signatures: int  # Roundtable signatures obtained
    spelltext_nonce: str       # Nonce encoded in spell-text


class SCBEWithTongues:
    """
    SCBE Verifier with Sacred Tongues integration.

    Combines:
    - Hyperbolic geometry verification
    - GeoSeal dual-space trust
    - Cryptolinguistic encoding (Sacred Tongues)
    - Roundtable consensus simulation
    """

    def __init__(self, key: bytes = b"demo_key"):
        self.key = key
        self.trusted_center = np.array([0.0, 0.0])
        self.risk_thresholds = {
            'allow': 0.2,
            'quarantine': 0.4,
            'deny': 0.8,
        }

        # Initialize tokenizers for each tongue
        self.tokenizers = {
            'ko': SacredTongueTokenizer('ko'),  # Kor'aelin - nonce/flow
            'ru': SacredTongueTokenizer('ru'),  # Runethic - binding
            'um': SacredTongueTokenizer('um'),  # Umbroth - redaction
        }

    def encode_nonce_to_spelltext(self, nonce: bytes) -> str:
        """Encode nonce to Kor'aelin spell-text."""
        return self.tokenizers['ko'].encode_to_string(nonce, separator=' ')

    def simulate_roundtable_consensus(
        self,
        risk: float,
        tongues: List[str],
    ) -> Tuple[int, List[str]]:
        """
        Simulate Roundtable multi-tongue consensus.

        High-risk actions require signatures from multiple tongues.
        Low-risk actions need fewer signatures.

        Args:
            risk: Calculated risk score
            tongues: List of tongue codes to query

        Returns:
            (signatures_obtained, signing_tongues)
        """
        # Required signatures based on risk
        if risk >= 0.7:
            required = 3  # High risk: need all tongues
        elif risk >= 0.4:
            required = 2  # Medium risk: need majority
        else:
            required = 1  # Low risk: single tongue sufficient

        # Simulate tongue responses (higher trust = more likely to sign)
        trust_factor = 1 - risk
        signers = []

        for tongue in tongues:
            # Each tongue has different signing threshold
            tongue_thresholds = {'ko': 0.3, 'ru': 0.5, 'um': 0.7}
            threshold = tongue_thresholds.get(tongue, 0.5)

            if trust_factor > threshold:
                signers.append(tongue)

        return len(signers), signers

    def verify(
        self,
        identity: str,
        intent: str,
        context: dict,
        nonce: bytes,
    ) -> AuthorizationResult:
        """
        Verify authorization using SCBE + Sacred Tongues.

        Pipeline:
        1. Encode nonce to spell-text (cryptolinguistic layer)
        2. Format SS1 blob (simulated PQC envelope)
        3. Compute hyperbolic distance to trusted center
        4. Apply harmonic scaling
        5. GeoSeal dual-space verification
        6. Roundtable consensus
        7. Final decision

        Args:
            identity: Agent/user identifier
            intent: Action intent (read, write, admin, etc.)
            context: Additional context dict
            nonce: Random nonce bytes

        Returns:
            AuthorizationResult with all metrics
        """
        # 1. Encode nonce to spell-text
        spelltext_nonce = self.encode_nonce_to_spelltext(nonce)
        print(f"\n📜 Nonce → KO Spell-Text: {spelltext_nonce}")

        # 2. Format SS1 blob (simulated)
        ss1_blob = format_ss1_blob(
            kid="scbe_v3",
            aad=f"mission:{context.get('mission', 'demo')}",
            salt=hashlib.sha256(identity.encode()).digest()[:16],
            nonce=nonce,
            ciphertext=b'\x00' * 32,  # Placeholder
            tag=b'\x00' * 16,          # Placeholder
        )
        print(f"🔐 SS1 Blob: {ss1_blob[:80]}...")

        # 3. Compute user point from identity hash
        id_hash = hashlib.sha256(identity.encode()).digest()
        user_point = np.array([
            (id_hash[0] / 255.0) * 0.8 - 0.4,
            (id_hash[1] / 255.0) * 0.8 - 0.4
        ])

        # 4. Hyperbolic distance
        d_H = hyperbolic_distance(user_point, self.trusted_center)

        # 5. Harmonic scaling
        H = harmonic_scaling(d_H)

        # 6. Intent risk
        intent_risks = {
            'read': 0.1,
            'view': 0.1,
            'write': 0.3,
            'update': 0.3,
            'admin': 0.6,
            'delete': 0.8,
            'purge': 0.9,
        }
        intent_risk = intent_risks.get(intent.lower(), 0.5)

        # 7. GeoSeal verification
        context_vector = np.array([
            context.get('trust_level', 0.5),
            1 - intent_risk,
            d_H / 5,  # Normalized distance
        ])
        policy_vector = np.array([0.8, 0.7, 0.3])

        geo_distance, geo_path = geoseal_distance(context_vector, policy_vector)
        time_dilation = float(np.exp(-2 * geo_distance))

        # 8. Calculate raw risk
        raw_risk = intent_risk * H
        if geo_path == 'exterior':
            raw_risk *= 1.5  # Penalty for exterior path

        # Normalize risk to [0, 1]
        normalized_risk = min(1.0, raw_risk / 5)

        # 9. Roundtable consensus
        tongues = ['ko', 'ru', 'um']
        signatures, signers = self.simulate_roundtable_consensus(normalized_risk, tongues)

        # 10. Final decision
        if normalized_risk >= self.risk_thresholds['deny']:
            if signatures == 0:
                decision = 'SNAP'  # Fail-to-noise
            else:
                decision = 'DENY'
        elif normalized_risk >= self.risk_thresholds['quarantine']:
            decision = 'QUARANTINE'
        elif normalized_risk >= self.risk_thresholds['allow']:
            if signatures >= 1:
                decision = 'ALLOW'
            else:
                decision = 'QUARANTINE'
        else:
            decision = 'ALLOW'

        return AuthorizationResult(
            decision=decision,
            risk_score=normalized_risk,
            hyperbolic_distance=d_H,
            harmonic_factor=H,
            geoseal_path=geo_path,
            time_dilation=time_dilation,
            consensus_signatures=signatures,
            spelltext_nonce=spelltext_nonce,
        )


def run_demo():
    """Run the integrated SCBE + Sacred Tongues demo."""
    print("=" * 60)
    print("    SCBE-AETHERMOORE + Sacred Tongues Integrated Demo")
    print("=" * 60)

    verifier = SCBEWithTongues()

    # Test cases
    test_cases = [
        {
            'name': 'Trusted Agent - Read',
            'identity': 'agent-alpha-001',
            'intent': 'read',
            'context': {'trust_level': 0.9, 'mission': 'data_retrieval'},
            'nonce': b'\x00\x2a\xff',
        },
        {
            'name': 'Unknown Agent - Admin',
            'identity': 'unknown-user',
            'intent': 'admin',
            'context': {'trust_level': 0.3, 'mission': 'system_config'},
            'nonce': b'\x99\xaa\xbb',
        },
        {
            'name': 'Compromised Agent - Delete',
            'identity': 'agent-compromised',
            'intent': 'delete',
            'context': {'trust_level': 0.1, 'mission': 'data_purge'},
            'nonce': b'\xde\xad\xbe\xef',
        },
        {
            'name': 'AI Model - Write',
            'identity': 'ai-model-v4',
            'intent': 'write',
            'context': {'trust_level': 0.6, 'mission': 'training_data'},
            'nonce': b'\x42\x42\x42',
        },
    ]

    for case in test_cases:
        print(f"\n{'─' * 60}")
        print(f"🎯 Scenario: {case['name']}")
        print(f"   Identity: {case['identity']}")
        print(f"   Intent: {case['intent']}")
        print(f"   Trust Level: {case['context']['trust_level']}")

        result = verifier.verify(
            identity=case['identity'],
            intent=case['intent'],
            context=case['context'],
            nonce=case['nonce'],
        )

        # Color-coded decision
        decision_colors = {
            'ALLOW': '\033[92m',      # Green
            'QUARANTINE': '\033[93m', # Yellow
            'DENY': '\033[91m',       # Red
            'SNAP': '\033[95m',       # Magenta
        }
        color = decision_colors.get(result.decision, '')
        reset = '\033[0m'

        print(f"\n📊 Results:")
        print(f"   Decision: {color}{result.decision}{reset}")
        print(f"   Risk Score: {result.risk_score:.4f}")
        print(f"   Hyperbolic Distance: {result.hyperbolic_distance:.4f}")
        print(f"   Harmonic Factor: {result.harmonic_factor:.4f}")
        print(f"   GeoSeal Path: {result.geoseal_path}")
        print(f"   Time Dilation τ: {result.time_dilation:.4f}")
        print(f"   Consensus Signatures: {result.consensus_signatures}/3")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    run_demo()
