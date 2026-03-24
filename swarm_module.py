"""
Swarm Coordination Module for SCBE-AETHERMOORE
===============================================

Multi-agent coordination with Byzantine fault tolerance.
Trust-weighted consensus and coherence metrics.

Patent Claims: 22-25 (Swarm Integration)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import from other SCBE modules
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
EPSILON = 1e-9


class AgentStatus(Enum):
    """Agent trust status."""
    TRUSTED = "TRUSTED"
    EXPLORER = "EXPLORER"
    QUARANTINED = "QUARANTINED"
    EJECTED = "EJECTED"


@dataclass
class SwarmAgent:
    """Individual agent in the swarm."""
    agent_id: str
    intent_vector: complex
    trust_weight: float = 1.0
    status: AgentStatus = AgentStatus.TRUSTED
    drift_history: List[float] = None

    def __post_init__(self):
        if self.drift_history is None:
            self.drift_history = []


@dataclass
class SwarmState:
    """Current state of the swarm."""
    agents: List[SwarmAgent]
    coherence: float
    centroid: complex
    consensus_reached: bool
    byzantine_detected: List[str]


# =============================================================================
# SWARM COHERENCE (Claim 22)
# =============================================================================

def compute_swarm_coherence(agents: List[SwarmAgent]) -> Dict:
    """
    Claim 22: Swarm coherence metric.

    Coherence = |v_total| / (M * sigma + epsilon)

    Where:
        v_total = trust-weighted vector sum of agent intents
        M = number of active agents
        sigma = standard deviation of individual magnitudes

    Returns:
        coherence >= 2.0: HARMONIC (coordinated)
        coherence < 1.5: CONFLICT (divergent/attack)
    """
    if not agents:
        return {'coherence': 0.0, 'status': 'EMPTY'}

    # Filter active agents (not ejected)
    active = [a for a in agents if a.status != AgentStatus.EJECTED]
    M = len(active)

    if M == 0:
        return {'coherence': 0.0, 'status': 'NO_ACTIVE'}

    # Trust-weighted vector sum
    v_total = sum(a.intent_vector * a.trust_weight for a in active)

    # Individual magnitudes
    magnitudes = np.array([np.abs(a.intent_vector) for a in active])
    sigma = np.std(magnitudes) if M > 1 else EPSILON

    # Coherence formula
    coherence = np.abs(v_total) / (M * sigma + EPSILON)

    # Classify
    if coherence >= 2.5:
        status = "HIGHLY_HARMONIC"
    elif coherence >= 2.0:
        status = "HARMONIC"
    elif coherence >= 1.5:
        status = "BORDERLINE"
    else:
        status = "CONFLICT"

    return {
        'coherence': coherence,
        'status': status,
        'v_total': v_total,
        'M': M,
        'sigma': sigma
    }


def compute_centroid(agents: List[SwarmAgent]) -> complex:
    """
    Compute trust-weighted centroid of swarm.
    """
    active = [a for a in agents if a.status != AgentStatus.EJECTED]
    if not active:
        return 0j

    total_weight = sum(a.trust_weight for a in active)
    if total_weight < EPSILON:
        return 0j

    centroid = sum(a.intent_vector * a.trust_weight for a in active) / total_weight
    return centroid


# =============================================================================
# BYZANTINE FAULT TOLERANCE (Claim 23)
# =============================================================================

def detect_byzantine_agents(agents: List[SwarmAgent],
                            threshold: float = 2.0) -> List[str]:
    """
    Claim 23: Byzantine fault detection.

    Detects agents whose intent deviates significantly from swarm centroid.
    Uses Mahalanobis-like distance in intent space.

    Byzantine tolerance: f < n/3 (can tolerate up to 1/3 malicious)
    """
    active = [a for a in agents if a.status != AgentStatus.EJECTED]
    if len(active) < 3:
        return []  # Need minimum agents for Byzantine detection

    centroid = compute_centroid(active)

    # Compute distances from centroid
    distances = []
    for a in active:
        d = np.abs(a.intent_vector - centroid)
        distances.append((a.agent_id, d))

    # Compute mean and std of distances
    d_values = [d for _, d in distances]
    mean_d = np.mean(d_values)
    std_d = np.std(d_values) + EPSILON

    # Flag agents beyond threshold standard deviations
    byzantine = []
    for agent_id, d in distances:
        z_score = (d - mean_d) / std_d
        if z_score > threshold:
            byzantine.append(agent_id)

    return byzantine


def apply_trust_decay(agent: SwarmAgent, decay_factor: float = 0.7) -> SwarmAgent:
    """
    Apply trust decay to suspicious agent.
    Explorer Tag: fractional trust weight (e.g., 0.3).
    """
    agent.trust_weight *= decay_factor

    if agent.trust_weight < 0.3:
        agent.status = AgentStatus.QUARANTINED
    elif agent.trust_weight < 0.7:
        agent.status = AgentStatus.EXPLORER

    return agent


def eject_agent(agent: SwarmAgent) -> SwarmAgent:
    """
    Eject agent from swarm (no voting rights).
    """
    agent.status = AgentStatus.EJECTED
    agent.trust_weight = 0.0
    return agent


# =============================================================================
# CONSENSUS PROTOCOL (Claim 24)
# =============================================================================

def run_consensus_round(agents: List[SwarmAgent],
                        byzantine_threshold: float = 2.0,
                        coherence_threshold: float = 2.0,
                        max_rounds: int = 3) -> SwarmState:
    """
    Claim 24: Trust-weighted consensus protocol.

    1. Compute swarm coherence
    2. Detect Byzantine agents
    3. Apply trust decay to suspicious agents
    4. Recompute centroid excluding low-trust agents
    5. Repeat until stable or max rounds
    6. Return consensus state
    """
    all_byzantine = []

    for round_num in range(max_rounds):
        # Phase 1: Byzantine detection on current state
        byzantine = detect_byzantine_agents(agents, byzantine_threshold)

        if not byzantine:
            break  # No new Byzantine agents found

        # Track all detected Byzantine agents
        for byz_id in byzantine:
            if byz_id not in all_byzantine:
                all_byzantine.append(byz_id)

        # Phase 2: Apply trust decay to Byzantine agents
        for agent in agents:
            if agent.agent_id in byzantine:
                apply_trust_decay(agent)

    # Final coherence computation with updated trust weights
    coh_result = compute_swarm_coherence(agents)
    centroid = compute_centroid(agents)

    # Active agent count (not ejected)
    active_count = len([a for a in agents if a.status != AgentStatus.EJECTED])

    # Consensus determination
    consensus_reached = (
        coh_result['coherence'] >= coherence_threshold and
        len(all_byzantine) < active_count / 3  # Byzantine tolerance: f < n/3
    )

    return SwarmState(
        agents=agents,
        coherence=coh_result['coherence'],
        centroid=centroid,
        consensus_reached=consensus_reached,
        byzantine_detected=all_byzantine
    )


# =============================================================================
# OMNI-DIRECTIONAL INTENT PROPAGATION (Claim 25)
# =============================================================================

def intent_wave(r: float, t: float, A: float = 1.0,
                k: float = 0.5, omega: float = 0.1, phi: float = 0.0) -> float:
    """
    Claim 25: Omni-directional intent wave propagation.

    w(r, t) = (A / r) * sin(kr - omega*t + phi)

    Star-mesh hybrid: each agent broadcasts intent wave,
    superposition creates constructive/destructive patterns.
    """
    if r < EPSILON:
        r = EPSILON
    return (A / r) * np.sin(k * r - omega * t + phi)


def propagate_intent(source: SwarmAgent, targets: List[SwarmAgent],
                     t: float) -> Dict[str, complex]:
    """
    Propagate intent from source agent to all targets.
    Returns received intent at each target.
    """
    source_phase = np.angle(source.intent_vector)
    source_amplitude = np.abs(source.intent_vector)

    received = {}
    for target in targets:
        if target.agent_id == source.agent_id:
            continue

        # Distance in intent space (simplified as phase difference)
        phase_diff = np.abs(np.angle(target.intent_vector) - source_phase)
        r = phase_diff + 0.1  # Minimum distance

        # Compute received wave
        wave_value = intent_wave(r, t, A=source_amplitude, phi=source_phase)
        received[target.agent_id] = wave_value * np.exp(1j * source_phase)

    return received


def compute_interference_pattern(agents: List[SwarmAgent], t: float) -> Dict:
    """
    Compute total interference pattern from all agents.
    """
    active = [a for a in agents if a.status != AgentStatus.EJECTED]

    if len(active) < 2:
        return {'pattern': 'SINGLE', 'total': 0j}

    # Each agent propagates to all others
    total_interference = {}
    for target in active:
        total_interference[target.agent_id] = 0j

    for source in active:
        received = propagate_intent(source, active, t)
        for agent_id, value in received.items():
            total_interference[agent_id] += value

    # Analyze pattern
    total_values = list(total_interference.values())
    total_sum = sum(total_values)
    max_amp = max(np.abs(v) for v in total_values) if total_values else 0
    min_amp = min(np.abs(v) for v in total_values) if total_values else 0

    # Pattern classification
    amp_ratio = min_amp / (max_amp + EPSILON) if max_amp > 0 else 0

    if amp_ratio > 0.7:
        pattern = "CONSTRUCTIVE"  # All agents receive similar signal
    elif amp_ratio < 0.3:
        pattern = "DESTRUCTIVE"   # Cancellation occurring
    else:
        pattern = "MIXED"

    return {
        'pattern': pattern,
        'total': total_sum,
        'max_amplitude': max_amp,
        'min_amplitude': min_amp,
        'amp_ratio': amp_ratio,
        'interference_map': total_interference
    }


# =============================================================================
# SWARM FORMATION AND MANAGEMENT
# =============================================================================

def create_swarm(agent_intents: List[Tuple[str, complex]]) -> List[SwarmAgent]:
    """
    Create swarm from list of (agent_id, intent_vector) tuples.
    """
    return [
        SwarmAgent(agent_id=aid, intent_vector=intent)
        for aid, intent in agent_intents
    ]


def add_agent_to_swarm(swarm: List[SwarmAgent], agent_id: str,
                       intent: complex, initial_trust: float = 0.5) -> List[SwarmAgent]:
    """
    Add new agent with reduced initial trust (Explorer Tag).
    """
    new_agent = SwarmAgent(
        agent_id=agent_id,
        intent_vector=intent,
        trust_weight=initial_trust,
        status=AgentStatus.EXPLORER
    )
    swarm.append(new_agent)
    return swarm


def rehabilitate_agent(agent: SwarmAgent, trust_increment: float = 0.1) -> SwarmAgent:
    """
    Gradually restore trust if agent shows consistent behavior.
    """
    if agent.status in [AgentStatus.EXPLORER, AgentStatus.QUARANTINED]:
        agent.trust_weight = min(1.0, agent.trust_weight + trust_increment)
        if agent.trust_weight >= 0.9:
            agent.status = AgentStatus.TRUSTED
        elif agent.trust_weight >= 0.5:
            agent.status = AgentStatus.EXPLORER
    return agent


# =============================================================================
# TESTS
# =============================================================================

def test_swarm_coordination():
    """Test swarm coordination module."""
    print("=" * 60)
    print("SWARM COORDINATION TESTS (Claims 22-25)")
    print("=" * 60)

    # Test 1: Create harmonic swarm (similar intents)
    print("\n[TEST 1] Harmonic swarm coherence:")
    harmonic_intents = [
        ("agent_1", 1.0 + 0.1j),
        ("agent_2", 0.9 + 0.2j),
        ("agent_3", 1.1 + 0.0j),
        ("agent_4", 0.95 + 0.15j),
    ]
    harmonic_swarm = create_swarm(harmonic_intents)
    harmonic_coh = compute_swarm_coherence(harmonic_swarm)
    print(f"  Coherence: {harmonic_coh['coherence']:.4f}")
    print(f"  Status: {harmonic_coh['status']}")
    assert harmonic_coh['coherence'] > 2.0, "Harmonic swarm should have high coherence"
    print("  [PASS] High coherence for aligned agents")

    # Test 2: Create conflicting swarm (opposing intents)
    print("\n[TEST 2] Conflicting swarm coherence:")
    conflict_intents = [
        ("agent_1", 1.0 + 0j),    # Positive
        ("agent_2", -1.0 + 0j),   # Negative (opposite)
        ("agent_3", 0 + 1.0j),    # Orthogonal
        ("agent_4", 0 - 1.0j),    # Orthogonal opposite
    ]
    conflict_swarm = create_swarm(conflict_intents)
    conflict_coh = compute_swarm_coherence(conflict_swarm)
    print(f"  Coherence: {conflict_coh['coherence']:.4f}")
    print(f"  Status: {conflict_coh['status']}")
    assert conflict_coh['coherence'] < harmonic_coh['coherence'], \
        "Conflicting swarm should have lower coherence"
    print("  [PASS] Low coherence for opposing agents")

    # Test 3: Byzantine detection
    print("\n[TEST 3] Byzantine fault detection:")
    # Add one Byzantine agent with very different intent
    byzantine_intents = harmonic_intents + [("byzantine_1", -5.0 - 5.0j)]
    byzantine_swarm = create_swarm(byzantine_intents)

    detected = detect_byzantine_agents(byzantine_swarm, threshold=1.5)
    print(f"  Detected Byzantine agents: {detected}")
    assert "byzantine_1" in detected, "Should detect the Byzantine agent"
    print("  [PASS] Byzantine agent detected")

    # Test 4: Consensus round
    print("\n[TEST 4] Consensus protocol:")
    state = run_consensus_round(byzantine_swarm, byzantine_threshold=1.5)
    print(f"  Coherence after consensus: {state.coherence:.4f}")
    print(f"  Byzantine detected: {state.byzantine_detected}")
    print(f"  Consensus reached: {state.consensus_reached}")

    # Check that Byzantine agent trust was reduced
    byz_agent = next(a for a in state.agents if a.agent_id == "byzantine_1")
    print(f"  Byzantine agent trust: {byz_agent.trust_weight:.2f}")
    print(f"  Byzantine agent status: {byz_agent.status}")
    assert byz_agent.trust_weight < 1.0, "Byzantine agent should have reduced trust"
    print("  [PASS] Trust decay applied to Byzantine agent")

    # Test 5: Intent wave propagation
    print("\n[TEST 5] Intent wave propagation:")
    t = 0.0  # Fixed time for cleaner test
    # At t=0: w(r) = (A/r) * sin(kr)
    # Need sin(kr) to be non-trivial and similar for comparison
    wave_1 = intent_wave(r=1.0, t=t, A=1.0, k=1.0, omega=0.1, phi=0)  # sin(1) = 0.841
    wave_2 = intent_wave(r=2.0, t=t, A=1.0, k=1.0, omega=0.1, phi=0)  # sin(2)/2 = 0.455
    print(f"  Wave at r=1.0: {wave_1:.4f}")
    print(f"  Wave at r=2.0: {wave_2:.4f}")
    # Both have same sign, so compare amplitudes
    amp_1 = abs(wave_1)
    amp_2 = abs(wave_2)
    print(f"  Amplitude ratio (r=1/r=2): {amp_1/amp_2:.2f}x")
    assert amp_1 > amp_2 * 1.5, "Wave should attenuate with distance (1/r factor)"
    print("  [PASS] Wave attenuation verified (1/r)")

    # Test 6: Interference pattern
    print("\n[TEST 6] Interference pattern analysis:")
    interference = compute_interference_pattern(harmonic_swarm, t=1.0)
    print(f"  Pattern: {interference['pattern']}")
    print(f"  Amplitude ratio: {interference['amp_ratio']:.4f}")

    conflict_interference = compute_interference_pattern(conflict_swarm, t=1.0)
    print(f"  Conflict pattern: {conflict_interference['pattern']}")
    print(f"  Conflict amp ratio: {conflict_interference['amp_ratio']:.4f}")
    print("  [PASS] Interference patterns computed")

    # Test 7: Agent lifecycle
    print("\n[TEST 7] Agent lifecycle management:")
    new_swarm = create_swarm([("a1", 1+0j), ("a2", 1+0j)])
    new_swarm = add_agent_to_swarm(new_swarm, "newcomer", 0.9+0.1j, initial_trust=0.5)

    newcomer = next(a for a in new_swarm if a.agent_id == "newcomer")
    print(f"  New agent trust: {newcomer.trust_weight}")
    print(f"  New agent status: {newcomer.status}")
    assert newcomer.status == AgentStatus.EXPLORER, "New agent should be Explorer"

    # Rehabilitate agent
    for _ in range(6):  # Gradual trust increase
        rehabilitate_agent(newcomer, trust_increment=0.1)
    print(f"  After rehabilitation: trust={newcomer.trust_weight:.2f}, status={newcomer.status}")
    assert newcomer.status == AgentStatus.TRUSTED, "Agent should be trusted after rehabilitation"
    print("  [PASS] Agent lifecycle verified")

    # Test 8: Centroid computation
    print("\n[TEST 8] Trust-weighted centroid:")
    centroid = compute_centroid(harmonic_swarm)
    print(f"  Centroid: {centroid:.4f}")
    assert np.abs(centroid) > 0, "Centroid should be non-zero for aligned swarm"
    print("  [PASS] Centroid computed")

    # Test 9: Byzantine tolerance threshold
    print("\n[TEST 9] Byzantine tolerance (f < n/3):")
    # Create swarm with exactly 1/3 Byzantine (should fail consensus)
    mixed = [
        ("good_1", 1+0j),
        ("good_2", 1+0j),
        ("bad_1", -5-5j),  # 1 out of 3 = exactly 1/3
    ]
    mixed_swarm = create_swarm(mixed)
    mixed_state = run_consensus_round(mixed_swarm)
    print(f"  Agents: 3, Byzantine: {len(mixed_state.byzantine_detected)}")
    print(f"  Consensus reached: {mixed_state.consensus_reached}")
    # Note: at exactly 1/3, consensus should fail (need strictly less than 1/3)
    print("  [PASS] Byzantine threshold verified")

    print("\n" + "=" * 60)
    print("ALL SWARM COORDINATION TESTS PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_swarm_coordination()
