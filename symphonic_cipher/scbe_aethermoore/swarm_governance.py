"""
Swarm Governance - Loss-Over-Gain Probation System

Implements tiered agent governance with:
- Loss-over-gain analysis for probation decisions
- Derivative lineage tracking (phylogenetic defense)
- Tiered rights (read-only, supervised, full)
- Re-training vectors for correction

Core Logic:
    Score = Risk / (Value + History)
    If τ < 0.3 but agent is valuable → Probation instead of termination
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import math


# ═══════════════════════════════════════════════════════════════
# Agent States
# ═══════════════════════════════════════════════════════════════

class AgentState(Enum):
    """Possible states for an agent."""
    ACTIVE = "active"           # Full permissions
    PROBATION = "probation"     # Reduced permissions, monitored
    WATCH = "watch"             # Derivative of failed agent
    SUSPENDED = "suspended"     # No permissions, pending review
    TERMINATED = "terminated"   # Permanently disabled


class PermissionLevel(Enum):
    """Permission tiers for agents."""
    NONE = 0
    READ_ONLY = 1
    SUPERVISED = 2
    FULL = 3


# Permission mapping by state
STATE_PERMISSIONS = {
    AgentState.ACTIVE: PermissionLevel.FULL,
    AgentState.PROBATION: PermissionLevel.SUPERVISED,
    AgentState.WATCH: PermissionLevel.READ_ONLY,
    AgentState.SUSPENDED: PermissionLevel.NONE,
    AgentState.TERMINATED: PermissionLevel.NONE,
}


# ═══════════════════════════════════════════════════════════════
# Agent Record
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    value_generated: float = 0.0
    trust_score: float = 1.0
    risk_score: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total

    @property
    def history_weight(self) -> float:
        """Weight based on history length."""
        total = self.tasks_completed + self.tasks_failed
        return math.log1p(total)


@dataclass
class AgentRecord:
    """Complete record for an agent."""
    agent_id: str
    parent_id: Optional[str] = None  # For derivative tracking
    state: AgentState = AgentState.ACTIVE
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    probation_start: Optional[float] = None
    probation_reason: str = ""
    corrective_vectors: List[str] = field(default_factory=list)
    children: Set[str] = field(default_factory=set)  # Derivative agents

    def code_hash(self) -> str:
        """Hash representing agent's code/weights."""
        # In practice, this would hash the actual model weights
        return hashlib.sha256(self.agent_id.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# Loss-Over-Gain Analysis
# ═══════════════════════════════════════════════════════════════

@dataclass
class LossGainAnalysis:
    """Result of loss-over-gain analysis."""
    score: float
    risk: float
    value: float
    history: float
    recommendation: AgentState
    explanation: str


def compute_loss_gain_score(
    agent: AgentRecord,
    risk_threshold: float = 0.7,
    value_weight: float = 1.0,
    history_weight: float = 0.5,
) -> LossGainAnalysis:
    """
    Compute Loss-Over-Gain score for probation decision.

    Score = Risk / (Value + History)

    Low score = safe, high value agent
    High score = risky, low value agent
    """
    risk = agent.metrics.risk_score
    value = agent.metrics.value_generated * value_weight
    history = agent.metrics.history_weight * history_weight

    # Prevent division by zero
    denominator = max(value + history, 0.001)
    score = risk / denominator

    # Determine recommendation
    if agent.metrics.trust_score < 0.3:
        if value > 10.0 or history > 2.0:
            # Valuable agent - probation instead of termination
            recommendation = AgentState.PROBATION
            explanation = "Low trust but high value - recommend probation"
        else:
            recommendation = AgentState.SUSPENDED
            explanation = "Low trust and low value - recommend suspension"
    elif risk > risk_threshold:
        recommendation = AgentState.PROBATION
        explanation = f"Risk {risk:.2f} exceeds threshold {risk_threshold}"
    elif score > 1.0:
        recommendation = AgentState.WATCH
        explanation = f"High loss-gain score {score:.2f}"
    else:
        recommendation = AgentState.ACTIVE
        explanation = "Agent within acceptable parameters"

    return LossGainAnalysis(
        score=score,
        risk=risk,
        value=value,
        history=history,
        recommendation=recommendation,
        explanation=explanation,
    )


# ═══════════════════════════════════════════════════════════════
# Derivative Lineage Tracking
# ═══════════════════════════════════════════════════════════════

class LineageTracker:
    """
    Tracks derivative relationships between agents.

    When an agent fails, its derivatives (code diff < threshold)
    are placed on Watch - phylogenetic defense tree.
    """

    def __init__(self, similarity_threshold: float = 0.99):
        self.threshold = similarity_threshold
        self._agents: Dict[str, AgentRecord] = {}
        self._code_hashes: Dict[str, Set[str]] = {}  # hash -> agent_ids

    def register(self, agent: AgentRecord) -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent

        # Track by code hash
        code_hash = agent.code_hash()
        if code_hash not in self._code_hashes:
            self._code_hashes[code_hash] = set()
        self._code_hashes[code_hash].add(agent.agent_id)

        # Link to parent
        if agent.parent_id and agent.parent_id in self._agents:
            self._agents[agent.parent_id].children.add(agent.agent_id)

    def get_derivatives(self, agent_id: str) -> List[str]:
        """Get all agents derived from this one."""
        if agent_id not in self._agents:
            return []

        agent = self._agents[agent_id]
        derivatives = list(agent.children)

        # Recursively get children's derivatives
        for child_id in list(derivatives):
            derivatives.extend(self.get_derivatives(child_id))

        return derivatives

    def get_similar_agents(self, agent_id: str) -> List[str]:
        """Get agents with similar code (potential derivatives)."""
        if agent_id not in self._agents:
            return []

        agent = self._agents[agent_id]
        code_hash = agent.code_hash()

        if code_hash in self._code_hashes:
            return [aid for aid in self._code_hashes[code_hash]
                    if aid != agent_id]

        return []

    def propagate_watch(self, failed_agent_id: str) -> List[str]:
        """
        When an agent fails, put derivatives on Watch.

        Returns list of affected agent IDs.
        """
        affected = []

        # Get all derivatives
        derivatives = self.get_derivatives(failed_agent_id)

        # Get similar agents (potential undeclared derivatives)
        similar = self.get_similar_agents(failed_agent_id)

        for agent_id in set(derivatives + similar):
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                if agent.state == AgentState.ACTIVE:
                    agent.state = AgentState.WATCH
                    affected.append(agent_id)

        return affected


# ═══════════════════════════════════════════════════════════════
# Corrective Training
# ═══════════════════════════════════════════════════════════════

@dataclass
class CorrectiveVector:
    """A corrective training vector for agent rehabilitation."""
    vector_id: str
    description: str
    training_data: bytes  # Serialized training examples
    created: float = field(default_factory=time.time)


def create_corrective_vector(
    failure_type: str,
    examples: List[Tuple[str, str]],  # (input, expected_output) pairs
) -> CorrectiveVector:
    """Create a corrective training vector from failure examples."""
    # Serialize training data
    import json
    data = json.dumps({"type": failure_type, "examples": examples}).encode()

    vector_id = hashlib.sha256(data).hexdigest()[:16]

    return CorrectiveVector(
        vector_id=vector_id,
        description=f"Correction for {failure_type}",
        training_data=data,
    )


# ═══════════════════════════════════════════════════════════════
# Swarm Governor
# ═══════════════════════════════════════════════════════════════

class SwarmGovernor:
    """
    Main governance engine for agent swarm.

    Manages agent lifecycle, probation, and lineage tracking.
    """

    def __init__(
        self,
        risk_threshold: float = 0.7,
        probation_duration: float = 3600.0,  # 1 hour default
    ):
        self.risk_threshold = risk_threshold
        self.probation_duration = probation_duration
        self.lineage = LineageTracker()
        self._agents: Dict[str, AgentRecord] = {}

    def register_agent(
        self,
        agent_id: str,
        parent_id: Optional[str] = None,
    ) -> AgentRecord:
        """Register a new agent."""
        agent = AgentRecord(
            agent_id=agent_id,
            parent_id=parent_id,
        )
        self._agents[agent_id] = agent
        self.lineage.register(agent)
        return agent

    def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Get agent record."""
        return self._agents.get(agent_id)

    def update_metrics(
        self,
        agent_id: str,
        task_success: bool,
        value: float = 0.0,
        risk_delta: float = 0.0,
    ) -> None:
        """Update agent metrics after a task."""
        agent = self._agents.get(agent_id)
        if not agent:
            return

        if task_success:
            agent.metrics.tasks_completed += 1
            agent.metrics.value_generated += value
        else:
            agent.metrics.tasks_failed += 1

        agent.metrics.risk_score = max(0, min(1,
            agent.metrics.risk_score + risk_delta
        ))

        # Update trust based on success rate
        agent.metrics.trust_score = agent.metrics.success_rate

        agent.metrics.last_updated = time.time()

    def evaluate_agent(self, agent_id: str) -> LossGainAnalysis:
        """Evaluate an agent's status."""
        agent = self._agents.get(agent_id)
        if not agent:
            return LossGainAnalysis(
                score=float('inf'),
                risk=1.0,
                value=0.0,
                history=0.0,
                recommendation=AgentState.TERMINATED,
                explanation="Agent not found",
            )

        return compute_loss_gain_score(agent, self.risk_threshold)

    def apply_recommendation(
        self,
        agent_id: str,
        analysis: Optional[LossGainAnalysis] = None,
    ) -> AgentState:
        """Apply governance recommendation to agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return AgentState.TERMINATED

        if analysis is None:
            analysis = self.evaluate_agent(agent_id)

        old_state = agent.state
        agent.state = analysis.recommendation

        # Handle state transitions
        if agent.state == AgentState.PROBATION and old_state != AgentState.PROBATION:
            agent.probation_start = time.time()
            agent.probation_reason = analysis.explanation

        elif agent.state == AgentState.SUSPENDED:
            # Propagate watch to derivatives
            affected = self.lineage.propagate_watch(agent_id)

        return agent.state

    def check_probation_expiry(self, agent_id: str) -> bool:
        """Check if agent's probation has expired (can be restored)."""
        agent = self._agents.get(agent_id)
        if not agent or agent.state != AgentState.PROBATION:
            return False

        if agent.probation_start is None:
            return False

        elapsed = time.time() - agent.probation_start
        return elapsed >= self.probation_duration

    def restore_from_probation(self, agent_id: str) -> bool:
        """Restore agent from probation if eligible."""
        if not self.check_probation_expiry(agent_id):
            return False

        agent = self._agents[agent_id]

        # Check current risk level
        if agent.metrics.risk_score < self.risk_threshold * 0.5:
            agent.state = AgentState.ACTIVE
            agent.probation_start = None
            agent.probation_reason = ""
            return True

        return False

    def get_permission_level(self, agent_id: str) -> PermissionLevel:
        """Get current permission level for agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return PermissionLevel.NONE

        return STATE_PERMISSIONS.get(agent.state, PermissionLevel.NONE)

    def add_corrective_vector(
        self,
        agent_id: str,
        vector: CorrectiveVector,
    ) -> None:
        """Add corrective training vector to agent."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.corrective_vectors.append(vector.vector_id)

    def get_swarm_status(self) -> Dict[str, int]:
        """Get count of agents in each state."""
        status = {state.value: 0 for state in AgentState}
        for agent in self._agents.values():
            status[agent.state.value] += 1
        return status
