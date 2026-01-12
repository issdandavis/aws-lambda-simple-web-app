"""
Conlang Harmonic System for SCBE-AETHERMOORE
=============================================

Six Sacred Tongues integration with the security envelope.
Phase-modulated semantic encoding at 60-degree intervals.

Patent Claims: 4, 5, 12, 13, 15 (phase modulation)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# Golden ratio for harmonic scaling
PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# SACRED TONGUES DEFINITIONS
# =============================================================================

SACRED_TONGUES = {
    'KO': {
        'phase': 0.0,
        'degrees': 0,
        'domain': 'origin/identity',
        'gate': 1,
        'description': 'Root of trust, identity binding'
    },
    'AV': {
        'phase': np.pi / 3,
        'degrees': 60,
        'domain': 'affirmation/yes',
        'gate': 2,
        'description': 'Positive intent, constructive harmonic'
    },
    'RU': {
        'phase': 2 * np.pi / 3,
        'degrees': 120,
        'domain': 'query/reflection',
        'gate': 3,
        'description': 'Coherence check, trajectory analysis'
    },
    'CA': {
        'phase': np.pi,
        'degrees': 180,
        'domain': 'negation/opposition',
        'gate': None,  # Counter-wave, not a gate
        'description': 'Destructive interference generation'
    },
    'UM': {
        'phase': 4 * np.pi / 3,
        'degrees': 240,
        'domain': 'uncertainty/doubt',
        'gate': 4,
        'description': 'Explorer tagging, fractional trust'
    },
    'DR': {
        'phase': 5 * np.pi / 3,
        'degrees': 300,
        'domain': 'completion/closure',
        'gate': 6,
        'description': 'Master signature, transaction seal'
    }
}


def tongue_to_vector(tongue: str) -> complex:
    """
    Convert Sacred Tongue to complex unit vector.

    Args:
        tongue: One of KO, AV, RU, CA, UM, DR

    Returns:
        Complex unit vector e^(iθ)
    """
    if tongue not in SACRED_TONGUES:
        raise ValueError(f"Unknown tongue: {tongue}. Valid: {list(SACRED_TONGUES.keys())}")

    phase = SACRED_TONGUES[tongue]['phase']
    return np.exp(1j * phase)


def detect_semantic_phase(intent_vector: complex) -> Tuple[str, float]:
    """
    Classify intent vector by nearest Sacred Tongue.

    Args:
        intent_vector: Complex intent value

    Returns:
        (tongue_name, distance_to_tongue)
    """
    phase = np.angle(intent_vector)
    if phase < 0:
        phase += 2 * np.pi

    # Find nearest tongue
    min_dist = float('inf')
    nearest_tongue = 'KO'

    for tongue, data in SACRED_TONGUES.items():
        tongue_phase = data['phase']
        # Angular distance (handles wraparound)
        dist = min(abs(phase - tongue_phase), 2 * np.pi - abs(phase - tongue_phase))
        if dist < min_dist:
            min_dist = dist
            nearest_tongue = tongue

    return nearest_tongue, min_dist


# =============================================================================
# HARMONIC RESONANCE PATTERNS
# =============================================================================

def hexagonal_sum() -> complex:
    """
    Sum of all six Sacred Tongue vectors.
    Should equal zero (perfect balance).
    """
    total = sum(tongue_to_vector(t) for t in SACRED_TONGUES.keys())
    return total


def constructive_triangle() -> complex:
    """
    KO + AV + DR triangle (0°, 60°, 300°).
    Returns constructive (positive real) sum.
    """
    return tongue_to_vector('KO') + tongue_to_vector('AV') + tongue_to_vector('DR')


def destructive_pair(tongue1: str, tongue2: str) -> complex:
    """
    Check if two tongues form a destructive (canceling) pair.
    """
    return tongue_to_vector(tongue1) + tongue_to_vector(tongue2)


def phase_interference(tongues: List[str]) -> Dict[str, float]:
    """
    Calculate interference pattern from multiple tongues.

    Returns:
        - magnitude: resultant amplitude
        - phase: resultant phase
        - coherence: how constructive (1.0 = perfect, 0.0 = cancellation)
    """
    total = sum(tongue_to_vector(t) for t in tongues)
    n = len(tongues)

    return {
        'magnitude': np.abs(total),
        'phase': np.angle(total),
        'coherence': np.abs(total) / n if n > 0 else 0.0
    }


# =============================================================================
# TRAJECTORY CLASSIFICATION
# =============================================================================

def classify_tongue_trajectory(phases: List[float]) -> Dict:
    """
    Classify a trajectory of phases by Sacred Tongue patterns.

    Args:
        phases: List of phase angles over time

    Returns:
        Classification dict with pattern analysis
    """
    # Detect tongue at each step
    tongues = []
    for phase in phases:
        dummy_vector = np.exp(1j * phase)
        tongue, _ = detect_semantic_phase(dummy_vector)
        tongues.append(tongue)

    # Count tongue occurrences
    counts = {t: 0 for t in SACRED_TONGUES.keys()}
    for t in tongues:
        counts[t] += 1

    # Detect patterns
    dominant = max(counts, key=counts.get)

    # Check for concerning patterns
    ca_ratio = counts['CA'] / len(tongues) if tongues else 0
    um_ratio = counts['UM'] / len(tongues) if tongues else 0

    # Pattern classification
    if ca_ratio > 0.5:
        pattern = "OPPOSITIONAL"
        status = "THREAT"
    elif um_ratio > 0.4:
        pattern = "OSCILLATORY"
        status = "PROBE"
    elif counts['KO'] > 0 and counts['AV'] > 0 and counts['DR'] > 0:
        pattern = "CONSTRUCTIVE"
        status = "HARMONIC"
    else:
        pattern = "MIXED"
        status = "NORMAL"

    return {
        'tongues': tongues,
        'counts': counts,
        'dominant': dominant,
        'pattern': pattern,
        'status': status,
        'ca_ratio': ca_ratio,
        'um_ratio': um_ratio
    }


# =============================================================================
# KLEIN BOTTLE ORIENTATION FLIP
# =============================================================================

def klein_tongue_flip(tongue: str) -> str:
    """
    Apply Klein bottle orientation reversal to a tongue.
    Adds 180° and maps to nearest tongue.

    This represents the automatic negative detection when
    intent traverses the Klein surface.
    """
    original_phase = SACRED_TONGUES[tongue]['phase']
    flipped_phase = (original_phase + np.pi) % (2 * np.pi)

    # Find nearest tongue to flipped phase
    flipped_vector = np.exp(1j * flipped_phase)
    flipped_tongue, _ = detect_semantic_phase(flipped_vector)

    return flipped_tongue


def detect_intent_inversion(original: str, observed: str) -> bool:
    """
    Check if observed tongue is the Klein-flipped version of original.
    Indicates potential attack masquerading as legitimate intent.
    """
    expected_flip = klein_tongue_flip(original)
    return observed == expected_flip


# =============================================================================
# EMOTIONAL SPIN WAVE INTEGRATION
# =============================================================================

def emotional_spin_wave(t: float, tongue: str, amplitude: float = 1.0,
                        frequency: float = 0.1) -> complex:
    """
    Generate emotional spin wave modulated by Sacred Tongue phase.

    v(t) = A · e^(i(ωt + φ_tongue))

    Args:
        t: Time value
        tongue: Sacred Tongue for phase offset
        amplitude: Wave amplitude [0.2, 1.5]
        frequency: Angular frequency (0.1 stable, 1.0 reactive)

    Returns:
        Complex spin wave value
    """
    phi_tongue = SACRED_TONGUES[tongue]['phase']
    return amplitude * np.exp(1j * (frequency * t + phi_tongue))


def multi_agent_interference(waves: List[complex]) -> Dict:
    """
    Calculate interference pattern from multiple agent spin waves.

    Returns coherence metrics.
    """
    total = sum(waves)
    n = len(waves)

    # Coherence ratio
    max_possible = n  # If all perfectly aligned
    coherence = np.abs(total) / max_possible if n > 0 else 0

    return {
        'resultant': total,
        'magnitude': np.abs(total),
        'phase': np.angle(total),
        'coherence': coherence,
        'is_harmonic': coherence > 0.7,
        'is_conflict': coherence < 0.3
    }


# =============================================================================
# COUNTER-WAVE GENERATION (CA PHASE)
# =============================================================================

def generate_counter_wave(threat_spectrum: np.ndarray,
                          amplification: float = 2.0) -> np.ndarray:
    """
    Generate CA-phase counter-wave for destructive interference.

    Uses the CA (180°) phase to create inverted signal.

    Args:
        threat_spectrum: FFT spectrum of threat trajectory
        amplification: Counter-wave strength multiplier

    Returns:
        Counter-wave spectrum for cancellation
    """
    ca_phase = tongue_to_vector('CA')  # = -1
    return threat_spectrum * ca_phase * amplification


def apply_destructive_interference(signal: np.ndarray,
                                   counter: np.ndarray) -> np.ndarray:
    """
    Apply counter-wave to neutralize threat signal.
    """
    return signal + counter


# =============================================================================
# TESTS
# =============================================================================

def test_sacred_tongues():
    """Test Sacred Tongues harmonic system."""
    print("=" * 60)
    print("SACRED TONGUES HARMONIC TESTS")
    print("=" * 60)

    # Test 1: All tongue vectors are unit magnitude
    print("\n[TEST 1] Unit vector verification:")
    for tongue in SACRED_TONGUES.keys():
        z = tongue_to_vector(tongue)
        mag = np.abs(z)
        assert abs(mag - 1.0) < 1e-10, f"{tongue} not unit: {mag}"
        print(f"  {tongue}: {z:.4f} (|z| = {mag:.6f})")
    print("  [PASS] All tongues are unit vectors")

    # Test 2: Hexagonal sum equals zero
    print("\n[TEST 2] Hexagonal balance:")
    hex_sum = hexagonal_sum()
    assert np.abs(hex_sum) < 1e-10, f"Hexagonal sum not zero: {hex_sum}"
    print(f"  Sum of all tongues: {hex_sum:.10f}")
    print("  [PASS] Perfect hexagonal balance (sum = 0)")

    # Test 3: Constructive triangle
    print("\n[TEST 3] Constructive triangle (KO + AV + DR):")
    tri = constructive_triangle()
    print(f"  Result: {tri:.4f} (magnitude = {np.abs(tri):.4f})")
    assert np.abs(tri) > 1.5, "Triangle should be constructive"
    print("  [PASS] Constructive interference verified")

    # Test 4: Destructive pair (KO + CA)
    print("\n[TEST 4] Destructive pair (KO + CA):")
    cancel = destructive_pair('KO', 'CA')
    print(f"  KO + CA = {cancel:.10f}")
    assert np.abs(cancel) < 1e-10, "Should cancel"
    print("  [PASS] Perfect cancellation")

    # Test 5: Klein flip mapping
    print("\n[TEST 5] Klein bottle orientation flips:")
    flip_pairs = [
        ('KO', 'CA'),  # 0° → 180°
        ('AV', 'UM'),  # 60° → 240°
        ('RU', 'DR'),  # 120° → 300°
    ]
    for orig, expected in flip_pairs:
        flipped = klein_tongue_flip(orig)
        print(f"  {orig} ({SACRED_TONGUES[orig]['degrees']}°) → " +
              f"{flipped} ({SACRED_TONGUES[flipped]['degrees']}°)")
        assert flipped == expected, f"Expected {expected}, got {flipped}"
    print("  [PASS] Klein orientation mapping correct")

    # Test 6: Semantic phase detection
    print("\n[TEST 6] Semantic phase detection:")
    test_vectors = [
        (1.0 + 0j, 'KO'),
        (0.5 + 0.866j, 'AV'),
        (-1.0 + 0j, 'CA'),
    ]
    for vec, expected in test_vectors:
        detected, dist = detect_semantic_phase(vec)
        print(f"  {vec} → {detected} (distance: {dist:.4f})")
        assert detected == expected, f"Expected {expected}, got {detected}"
    print("  [PASS] Phase detection accurate")

    # Test 7: Trajectory classification
    print("\n[TEST 7] Trajectory classification:")

    # Legitimate: KO → AV → RU → DR
    legit_phases = [0, np.pi/3, 2*np.pi/3, 5*np.pi/3]
    legit_result = classify_tongue_trajectory(legit_phases)
    print(f"  Legit trajectory: {legit_result['tongues']}")
    print(f"    Pattern: {legit_result['pattern']}, Status: {legit_result['status']}")
    assert legit_result['status'] in ['HARMONIC', 'NORMAL'], "Legit should be harmonic/normal"

    # Attack: CA → CA → CA → CA
    attack_phases = [np.pi, np.pi, np.pi, np.pi]
    attack_result = classify_tongue_trajectory(attack_phases)
    print(f"  Attack trajectory: {attack_result['tongues']}")
    print(f"    Pattern: {attack_result['pattern']}, Status: {attack_result['status']}")
    assert attack_result['status'] == 'THREAT', "Attack should be threat"

    print("  [PASS] Trajectory classification correct")

    # Test 8: Emotional spin waves
    print("\n[TEST 8] Emotional spin wave interference:")
    t = 1.0

    # Harmonic agents (similar phases)
    harmonic_waves = [
        emotional_spin_wave(t, 'KO', 1.0, 0.1),
        emotional_spin_wave(t, 'AV', 1.0, 0.1),
    ]
    harmonic_result = multi_agent_interference(harmonic_waves)
    print(f"  Harmonic waves coherence: {harmonic_result['coherence']:.4f}")

    # Conflicting agents (opposite phases)
    conflict_waves = [
        emotional_spin_wave(t, 'KO', 1.0, 0.1),
        emotional_spin_wave(t, 'CA', 1.0, 0.1),
    ]
    conflict_result = multi_agent_interference(conflict_waves)
    print(f"  Conflicting waves coherence: {conflict_result['coherence']:.4f}")

    assert harmonic_result['coherence'] > conflict_result['coherence'], \
        "Harmonic should have higher coherence"
    print("  [PASS] Coherence differentiation works")

    # Test 9: Counter-wave generation
    print("\n[TEST 9] Counter-wave (CA phase) generation:")
    threat = np.array([1.0, 0.5, 0.3, 0.2])
    counter = generate_counter_wave(threat, amplification=1.0)
    neutralized = apply_destructive_interference(threat, counter)

    print(f"  Threat: {threat}")
    print(f"  Counter: {counter}")
    print(f"  Neutralized: {neutralized}")

    assert np.allclose(neutralized, 0), "Should neutralize"
    print("  [PASS] Destructive interference successful")

    print("\n" + "=" * 60)
    print("ALL SACRED TONGUES TESTS PASSED")
    print("=" * 60)

    return True


# =============================================================================
# INTEGRATION WITH SCBE CORE
# =============================================================================

def modulate_context_vector(context_v2: complex, semantic_tongue: str) -> complex:
    """
    Modulate v₂ (intent) component with Sacred Tongue phase.

    This creates the semantic-cryptographic binding where
    linguistic intent directly affects the geometric manifold.
    """
    tongue_phase = tongue_to_vector(semantic_tongue)
    return context_v2 * tongue_phase


def extract_tongue_from_context(context_v2: complex) -> str:
    """
    Extract dominant Sacred Tongue from context vector v₂.
    """
    tongue, _ = detect_semantic_phase(context_v2)
    return tongue


if __name__ == "__main__":
    test_sacred_tongues()
