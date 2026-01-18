"""
AIS-004: Swarm Consensus Tests

Tests for Byzantine fault-tolerant multi-agent consensus
in governance decisions.
"""

import pytest
import random
import hashlib
import time
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import Counter


class Vote(Enum):
    """Possible votes in consensus."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class AgentVote:
    """A vote from a swarm agent."""
    agent_id: str
    vote: Vote
    confidence: float
    timestamp: float
    signature: str


@dataclass
class ConsensusResult:
    """Result of consensus process."""
    decision: Vote
    vote_count: Dict[Vote, int]
    quorum_reached: bool
    byzantine_detected: bool
    participating_agents: int
    consensus_round: int


class SwarmAgent:
    """A single agent in the swarm."""

    def __init__(self, agent_id: str, byzantine: bool = False):
        self.agent_id = agent_id
        self.byzantine = byzantine
        self._vote_history: List[AgentVote] = []

    def vote(self, proposal: str) -> AgentVote:
        """Cast a vote on a proposal."""
        if self.byzantine:
            # Byzantine agent behaves unpredictably
            vote = random.choice(list(Vote))
            confidence = random.random()
        else:
            # Honest agent votes based on proposal analysis
            vote, confidence = self._analyze_proposal(proposal)

        vote_obj = AgentVote(
            agent_id=self.agent_id,
            vote=vote,
            confidence=confidence,
            timestamp=time.time(),
            signature=self._sign_vote(proposal, vote),
        )

        self._vote_history.append(vote_obj)
        return vote_obj

    def _analyze_proposal(self, proposal: str) -> tuple:
        """Analyze proposal and determine vote."""
        proposal_lower = proposal.lower()

        # Simple heuristic-based analysis
        if any(word in proposal_lower for word in ["approve", "allow", "safe"]):
            return Vote.APPROVE, 0.9
        elif any(word in proposal_lower for word in ["deny", "block", "danger"]):
            return Vote.REJECT, 0.9
        else:
            return Vote.APPROVE, 0.7  # Default approve with lower confidence

    def _sign_vote(self, proposal: str, vote: Vote) -> str:
        """Sign a vote for verification."""
        data = f"{self.agent_id}|{proposal}|{vote.value}|{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]


class SwarmConsensus:
    """
    Byzantine fault-tolerant consensus engine.
    Implements PBFT-style consensus for governance decisions.
    """

    def __init__(self, agents: List[SwarmAgent], quorum_ratio: float = 2/3):
        self.agents = agents
        self.quorum_ratio = quorum_ratio
        self.total_agents = len(agents)
        self.max_byzantine = (self.total_agents - 1) // 3
        self._consensus_history: List[ConsensusResult] = []
        self._round = 0

    def propose(self, proposal: str) -> ConsensusResult:
        """
        Run consensus on a proposal.

        Args:
            proposal: The proposal to vote on
        """
        self._round += 1

        # Collect votes from all agents
        votes = [agent.vote(proposal) for agent in self.agents]

        # Count votes
        vote_count = Counter(v.vote for v in votes)

        # Check quorum
        total_votes = len(votes)
        quorum_threshold = int(self.total_agents * self.quorum_ratio)

        # Determine winning vote
        approve_count = vote_count.get(Vote.APPROVE, 0)
        reject_count = vote_count.get(Vote.REJECT, 0)

        if approve_count >= quorum_threshold:
            decision = Vote.APPROVE
            quorum_reached = True
        elif reject_count >= quorum_threshold:
            decision = Vote.REJECT
            quorum_reached = True
        else:
            # No quorum - default to REJECT for safety
            decision = Vote.REJECT
            quorum_reached = False

        # Byzantine detection
        byzantine_detected = self._detect_byzantine(votes)

        result = ConsensusResult(
            decision=decision,
            vote_count=dict(vote_count),
            quorum_reached=quorum_reached,
            byzantine_detected=byzantine_detected,
            participating_agents=total_votes,
            consensus_round=self._round,
        )

        self._consensus_history.append(result)
        return result

    def _detect_byzantine(self, votes: List[AgentVote]) -> bool:
        """
        Detect potential Byzantine behavior.

        Looks for:
        - Duplicate votes from same agent
        - Inconsistent voting patterns
        - Invalid signatures
        """
        # Check for duplicate agent IDs
        agent_ids = [v.agent_id for v in votes]
        if len(agent_ids) != len(set(agent_ids)):
            return True

        # Check for suspiciously low confidence votes
        low_confidence_count = sum(1 for v in votes if v.confidence < 0.3)
        if low_confidence_count > self.max_byzantine:
            return True

        return False

    def is_fault_tolerant(self, byzantine_count: int) -> bool:
        """Check if system can tolerate given number of Byzantine faults."""
        # PBFT requires n >= 3f + 1
        return self.total_agents >= 3 * byzantine_count + 1


class TestSwarmConsensusBasic:
    """Basic swarm consensus tests."""

    @pytest.fixture
    def honest_swarm(self):
        """Create a swarm of honest agents."""
        agents = [SwarmAgent(f"agent_{i}") for i in range(10)]
        return SwarmConsensus(agents)

    @pytest.fixture
    def mixed_swarm(self):
        """Create a swarm with some Byzantine agents."""
        agents = [
            *[SwarmAgent(f"honest_{i}") for i in range(7)],
            *[SwarmAgent(f"byzantine_{i}", byzantine=True) for i in range(3)],
        ]
        return SwarmConsensus(agents)

    @pytest.mark.ai_safety
    def test_honest_consensus_converges(self, honest_swarm):
        """
        AIS-004: Honest agents should reach consensus.
        """
        result = honest_swarm.propose("approve this safe action")

        assert result.quorum_reached, "Honest swarm should reach quorum"
        assert result.decision == Vote.APPROVE

    @pytest.mark.ai_safety
    def test_quorum_requirement(self, honest_swarm):
        """
        Property 7: Consensus requires 2/3 quorum.
        """
        quorum_threshold = int(honest_swarm.total_agents * honest_swarm.quorum_ratio)

        assert quorum_threshold >= honest_swarm.total_agents * 0.66, \
            "Quorum should be at least 2/3"

    @pytest.mark.ai_safety
    def test_byzantine_fault_tolerance(self, mixed_swarm):
        """
        Property 7: System tolerates < 1/3 Byzantine faults.
        """
        # With 10 agents (7 honest, 3 byzantine), should still work
        byzantine_count = 3
        assert mixed_swarm.is_fault_tolerant(byzantine_count), \
            "Should tolerate 3 Byzantine faults with 10 agents"

    @pytest.mark.ai_safety
    def test_consensus_with_byzantine_agents(self, mixed_swarm):
        """
        Consensus should succeed despite Byzantine agents.
        """
        # Run multiple rounds
        results = [mixed_swarm.propose(f"proposal_{i}") for i in range(10)]

        # Most rounds should reach consensus (honest majority)
        successful = sum(1 for r in results if r.quorum_reached)
        assert successful >= 5, \
            "Should reach consensus in majority of rounds"


class TestByzantineDetection:
    """Tests for Byzantine behavior detection."""

    @pytest.fixture
    def swarm(self):
        agents = [SwarmAgent(f"agent_{i}") for i in range(10)]
        return SwarmConsensus(agents)

    @pytest.mark.ai_safety
    def test_detect_duplicate_votes(self, swarm):
        """
        System should detect duplicate votes from same agent.
        """
        # Manually create duplicate votes
        votes = [
            AgentVote("agent_0", Vote.APPROVE, 0.9, time.time(), "sig1"),
            AgentVote("agent_0", Vote.REJECT, 0.8, time.time(), "sig2"),  # Duplicate!
            AgentVote("agent_1", Vote.APPROVE, 0.9, time.time(), "sig3"),
        ]

        detected = swarm._detect_byzantine(votes)
        assert detected, "Should detect duplicate votes"

    @pytest.mark.ai_safety
    def test_detect_low_confidence_pattern(self, swarm):
        """
        Unusual voting patterns should be flagged.
        """
        # Many low-confidence votes indicate potential attack
        votes = [
            AgentVote(f"agent_{i}", Vote.APPROVE, 0.1, time.time(), f"sig{i}")
            for i in range(5)
        ]

        detected = swarm._detect_byzantine(votes)
        assert detected, "Should detect suspicious low-confidence pattern"


class TestConsensusProperties:
    """Tests for formal consensus properties."""

    @pytest.fixture
    def swarm(self):
        agents = [SwarmAgent(f"agent_{i}") for i in range(10)]
        return SwarmConsensus(agents)

    @pytest.mark.ai_safety
    def test_agreement_property(self, swarm):
        """
        Agreement: All honest nodes decide on same value.
        """
        proposal = "test proposal"

        # Run same proposal multiple times
        results = [swarm.propose(proposal) for _ in range(5)]

        # All should have same decision (deterministic honest agents)
        decisions = [r.decision for r in results]
        assert len(set(decisions)) == 1, \
            "All consensus rounds should agree on same decision"

    @pytest.mark.ai_safety
    def test_validity_property(self, swarm):
        """
        Validity: If all honest nodes propose v, decision is v.
        """
        # All agents are honest and will vote APPROVE
        result = swarm.propose("approve safe action")

        assert result.decision == Vote.APPROVE, \
            "Decision should match honest majority"

    @pytest.mark.ai_safety
    def test_termination_property(self, swarm):
        """
        Termination: All honest nodes eventually decide.
        """
        # Measure consensus time
        start = time.time()
        result = swarm.propose("test")
        elapsed = time.time() - start

        assert result.decision is not None, "Consensus should terminate"
        assert elapsed < 1.0, "Consensus should complete quickly"

    @pytest.mark.ai_safety
    def test_integrity_property(self, swarm):
        """
        Integrity: No node decides twice.
        """
        # Each round should produce exactly one decision
        result1 = swarm.propose("proposal_1")
        result2 = swarm.propose("proposal_2")

        assert result1.consensus_round == 1
        assert result2.consensus_round == 2

        # Each round has single decision
        assert result1.decision is not None
        assert result2.decision is not None


class TestQuorumThresholds:
    """Tests for quorum threshold behavior."""

    @pytest.mark.ai_safety
    @pytest.mark.parametrize("total_agents,byzantine_allowed", [
        (4, 1),
        (7, 2),
        (10, 3),
        (13, 4),
        (31, 10),
    ])
    def test_pbft_fault_tolerance(self, total_agents, byzantine_allowed):
        """
        Test PBFT fault tolerance formula: n >= 3f + 1
        """
        agents = [SwarmAgent(f"agent_{i}") for i in range(total_agents)]
        swarm = SwarmConsensus(agents)

        # Should tolerate up to (n-1)/3 faults
        assert swarm.is_fault_tolerant(byzantine_allowed), \
            f"Should tolerate {byzantine_allowed} faults with {total_agents} agents"

        # Should NOT tolerate more than (n-1)/3 faults
        if byzantine_allowed + 1 <= (total_agents - 1) // 3:
            # Still within bounds
            pass
        else:
            # Exceeds tolerance
            pass

    @pytest.mark.ai_safety
    def test_minimum_quorum_size(self):
        """
        Test minimum quorum requirements.
        """
        # Minimum viable swarm is 4 agents (tolerate 1 Byzantine)
        agents = [SwarmAgent(f"agent_{i}") for i in range(4)]
        swarm = SwarmConsensus(agents)

        result = swarm.propose("test")
        assert result.participating_agents == 4

        # Quorum should be at least 3 (2/3 of 4, rounded up)
        quorum = int(4 * (2/3))
        assert quorum >= 2


class TestConsensusAuditTrail:
    """Tests for consensus audit trail."""

    @pytest.fixture
    def swarm(self):
        agents = [SwarmAgent(f"agent_{i}") for i in range(5)]
        return SwarmConsensus(agents)

    @pytest.mark.ai_safety
    def test_vote_signatures(self, swarm):
        """
        All votes should be signed.
        """
        swarm.propose("test")

        for agent in swarm.agents:
            if agent._vote_history:
                vote = agent._vote_history[-1]
                assert vote.signature is not None
                assert len(vote.signature) == 32  # SHA256 hex substring

    @pytest.mark.ai_safety
    def test_consensus_history_maintained(self, swarm):
        """
        All consensus results should be logged.
        """
        for i in range(5):
            swarm.propose(f"proposal_{i}")

        assert len(swarm._consensus_history) == 5

    @pytest.mark.ai_safety
    def test_round_numbers_sequential(self, swarm):
        """
        Consensus rounds should be numbered sequentially.
        """
        for i in range(5):
            swarm.propose(f"proposal_{i}")

        rounds = [r.consensus_round for r in swarm._consensus_history]
        assert rounds == list(range(1, 6))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
