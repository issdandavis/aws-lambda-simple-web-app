"""
Swarm Consensus Module (AXIS 5)

Implements distributed trust with harmonic trust weighting.
Nodes that deviate from the swarm's harmonic center lose trust super-exponentially.

Reference: Section 5.5 of SCBE-AETHER-UNIFIED-2026-001
Claims: 34, 36, 40, 43, 51
"""

import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from .constants import PERFECT_FIFTH, PARAMETER_RANGES
from .harmonic import harmonic_scaling


class NodeStatus(Enum):
    """Node status in the swarm."""
    ACTIVE = "active"
    PROBATION = "probation"
    EXCLUDED = "excluded"
    CANDIDATE = "candidate"


@dataclass
class SwarmNode:
    """
    A node in the distributed swarm.
    """
    node_id: str
    trust: float = 1.0
    harmonic_dimension: int = 6
    status: NodeStatus = NodeStatus.ACTIVE
    validation_count: int = 0
    rejection_count: int = 0

    # Participation threshold (Claim 34(d))
    tau_participate: float = field(
        default=PARAMETER_RANGES["tau_participate"]["default"]
    )

    def can_participate(self) -> bool:
        """Check if node can participate in consensus."""
        return (
            self.trust >= self.tau_participate and
            self.status == NodeStatus.ACTIVE
        )

    def trust_ratio(self) -> float:
        """Return validation/rejection ratio as trust indicator."""
        total = self.validation_count + self.rejection_count
        if total == 0:
            return 0.5
        return self.validation_count / total


def trust_update(
    current_trust: float,
    validity_factor: float,
    alpha: float = 0.9
) -> float:
    """
    Standard trust update rule.

    τ_new = α × τ_old + (1-α) × validity_factor

    Args:
        current_trust: Current trust level
        validity_factor: New validity observation (0-1)
        alpha: Smoothing factor (default: 0.9)

    Returns:
        Updated trust level

    Claim: 36
    """
    return alpha * current_trust + (1 - alpha) * validity_factor


def harmonic_trust_decay(
    current_trust: float,
    validity_factor: float,
    d_deviation: int,
    alpha: float = 0.9,
    R: float = PERFECT_FIFTH
) -> float:
    """
    Trust update with harmonic decay for deviant nodes.

    τ_new = α × τ_old + (1-α) × validity_factor × (1 / H(d_deviation, R₅))

    Nodes that deviate from the swarm's harmonic center lose trust super-exponentially.
    A deviation of d=3 multiplies trust loss by 38×.

    Args:
        current_trust: Current trust level
        validity_factor: New validity observation
        d_deviation: Harmonic deviation from swarm center
        alpha: Smoothing factor
        R: Harmonic ratio

    Returns:
        Updated trust level

    Reference: Section 5.5
    Claim: 51
    """
    if d_deviation <= 0:
        # No deviation - standard update
        return trust_update(current_trust, validity_factor, alpha)

    # Super-exponential penalty
    H = harmonic_scaling(d_deviation, R)
    penalized_validity = validity_factor / H

    return alpha * current_trust + (1 - alpha) * penalized_validity


def harmonic_trust_table() -> List[dict]:
    """
    Generate trust decay table for different deviation levels.

    Reference: Test 13
    """
    results = []
    for d in range(0, 5):
        if d == 0:
            H = 1.0
        else:
            H = harmonic_scaling(d)

        decay_factor = 1 / H

        results.append({
            "d_deviation": d,
            "H": H,
            "decay_factor": decay_factor,
            "trust_after_one_update": decay_factor  # Assuming validity=1, alpha=0
        })

    return results


def swarm_consensus(
    votes: Dict[str, bool],
    node_trusts: Dict[str, float],
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Compute swarm consensus with trust weighting.

    Args:
        votes: Dict of node_id -> vote (True/False)
        node_trusts: Dict of node_id -> trust level
        threshold: Consensus threshold

    Returns:
        Tuple of (consensus_result, confidence)

    Claim: 34
    """
    if not votes:
        return (False, 0.0)

    weighted_yes = 0.0
    weighted_no = 0.0
    total_weight = 0.0

    for node_id, vote in votes.items():
        trust = node_trusts.get(node_id, 0.5)
        total_weight += trust

        if vote:
            weighted_yes += trust
        else:
            weighted_no += trust

    if total_weight == 0:
        return (False, 0.0)

    yes_ratio = weighted_yes / total_weight
    consensus = yes_ratio >= threshold
    confidence = abs(yes_ratio - 0.5) * 2  # 0 at 50/50, 1 at unanimous

    return (consensus, confidence)


def compute_harmonic_centroid(nodes: List[SwarmNode]) -> float:
    """
    Compute the trust-weighted harmonic centroid of the swarm.

    The centroid is the average harmonic dimension weighted by trust.

    Args:
        nodes: List of swarm nodes

    Returns:
        Harmonic centroid dimension
    """
    if not nodes:
        return 6.0  # Default

    total_weight = sum(n.trust for n in nodes if n.status == NodeStatus.ACTIVE)
    if total_weight == 0:
        return 6.0

    weighted_sum = sum(
        n.harmonic_dimension * n.trust
        for n in nodes
        if n.status == NodeStatus.ACTIVE
    )

    return weighted_sum / total_weight


def self_exclusion_check(
    node: SwarmNode,
    exclusion_threshold: float = 0.1
) -> bool:
    """
    Check if a node should self-exclude from the swarm.

    Nodes with trust below threshold should voluntarily exclude.

    Args:
        node: Node to check
        exclusion_threshold: Trust level below which to exclude

    Returns:
        True if node should self-exclude

    Claim: 40, 43
    """
    if node.trust < exclusion_threshold:
        return True

    # Also check validation ratio
    if node.validation_count + node.rejection_count > 10:
        if node.trust_ratio() < 0.2:
            return True

    return False


class Swarm:
    """
    Complete swarm consensus system with harmonic trust.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        tau_participate: float = 0.3,
        exclusion_threshold: float = 0.1,
        consensus_threshold: float = 0.5
    ):
        """
        Initialize swarm.

        Args:
            alpha: Trust smoothing factor
            tau_participate: Minimum trust to participate
            exclusion_threshold: Trust level for exclusion
            consensus_threshold: Voting threshold
        """
        self.alpha = alpha
        self.tau_participate = tau_participate
        self.exclusion_threshold = exclusion_threshold
        self.consensus_threshold = consensus_threshold

        self.nodes: Dict[str, SwarmNode] = {}

    def add_node(
        self,
        node_id: str,
        initial_trust: float = 0.5,
        harmonic_dimension: int = 6
    ) -> SwarmNode:
        """Add a new node to the swarm."""
        node = SwarmNode(
            node_id=node_id,
            trust=initial_trust,
            harmonic_dimension=harmonic_dimension,
            status=NodeStatus.CANDIDATE,
            tau_participate=self.tau_participate
        )
        self.nodes[node_id] = node
        return node

    def promote_node(self, node_id: str) -> bool:
        """Promote a candidate to active status."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if node.status == NodeStatus.CANDIDATE and node.trust >= self.tau_participate:
                node.status = NodeStatus.ACTIVE
                return True
        return False

    def get_centroid(self) -> float:
        """Get current harmonic centroid."""
        return compute_harmonic_centroid(list(self.nodes.values()))

    def update_trust(
        self,
        node_id: str,
        validity_factor: float,
        use_harmonic_decay: bool = True
    ) -> float:
        """
        Update a node's trust.

        Args:
            node_id: Node to update
            validity_factor: Validity observation (0-1)
            use_harmonic_decay: Apply harmonic penalty for deviation

        Returns:
            New trust level
        """
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")

        node = self.nodes[node_id]

        if use_harmonic_decay:
            centroid = self.get_centroid()
            d_deviation = abs(node.harmonic_dimension - round(centroid))
            new_trust = harmonic_trust_decay(
                node.trust,
                validity_factor,
                int(d_deviation),
                self.alpha
            )
        else:
            new_trust = trust_update(node.trust, validity_factor, self.alpha)

        node.trust = new_trust

        # Update counters
        if validity_factor > 0.5:
            node.validation_count += 1
        else:
            node.rejection_count += 1

        # Check for exclusion
        if self_exclusion_check(node, self.exclusion_threshold):
            node.status = NodeStatus.EXCLUDED

        return new_trust

    def vote(
        self,
        votes: Dict[str, bool]
    ) -> Tuple[bool, float, dict]:
        """
        Run a consensus vote.

        Args:
            votes: Dict of node_id -> vote

        Returns:
            Tuple of (result, confidence, details)
        """
        # Filter to participating nodes only
        valid_votes = {}
        trusts = {}

        for node_id, vote in votes.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.can_participate():
                    valid_votes[node_id] = vote
                    trusts[node_id] = node.trust

        result, confidence = swarm_consensus(
            valid_votes,
            trusts,
            self.consensus_threshold
        )

        return (result, confidence, {
            "valid_voters": len(valid_votes),
            "total_nodes": len(self.nodes),
            "excluded_votes": len(votes) - len(valid_votes),
            "centroid": self.get_centroid()
        })

    def get_active_nodes(self) -> List[SwarmNode]:
        """Get all active nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]

    def get_excluded_nodes(self) -> List[SwarmNode]:
        """Get all excluded nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.EXCLUDED]

    def swarm_health(self) -> dict:
        """Get swarm health metrics."""
        nodes = list(self.nodes.values())
        active = [n for n in nodes if n.status == NodeStatus.ACTIVE]
        excluded = [n for n in nodes if n.status == NodeStatus.EXCLUDED]

        avg_trust = sum(n.trust for n in active) / len(active) if active else 0

        return {
            "total_nodes": len(nodes),
            "active_nodes": len(active),
            "excluded_nodes": len(excluded),
            "candidate_nodes": len([n for n in nodes if n.status == NodeStatus.CANDIDATE]),
            "average_trust": avg_trust,
            "harmonic_centroid": self.get_centroid(),
            "participating_capacity": sum(1 for n in active if n.can_participate())
        }

    def prune_excluded(self) -> int:
        """Remove excluded nodes. Returns count removed."""
        excluded_ids = [
            nid for nid, n in self.nodes.items()
            if n.status == NodeStatus.EXCLUDED
        ]
        for nid in excluded_ids:
            del self.nodes[nid]
        return len(excluded_ids)
