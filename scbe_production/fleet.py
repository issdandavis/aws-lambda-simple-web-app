"""
SCBE Fleet Orchestration Engine
================================
The "body" that makes the SCBE "brain" actually move and act.

This module provides:
- Fleet scenario execution
- Agent registration and management
- Task routing through SCBE pipeline
- Decision aggregation and logging

Usage:
    orchestrator = FleetOrchestrator()
    result = orchestrator.run_scenario(scenario)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Local imports
try:
    from .service import SCBEProductionService, AccessRequest
    from .logging import get_logger
except ImportError:
    from service import SCBEProductionService, AccessRequest
    from logging import get_logger


# =============================================================================
# Data Models
# =============================================================================

class AgentRole(Enum):
    """Agent roles in the fleet."""
    WORKER = "worker"           # Standard task executor
    COORDINATOR = "coordinator"  # Task dispatcher
    AUDITOR = "auditor"         # Compliance checker
    ADMIN = "admin"             # Administrative access
    GUEST = "guest"             # Limited access


class ActionType(Enum):
    """Types of actions agents can request."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    QUERY = "query"
    SYNC = "sync"


class Decision(Enum):
    """SCBE governance decisions."""
    ALLOW = "ALLOW"
    QUARANTINE = "QUARANTINE"
    DENY = "DENY"
    SNAP = "SNAP"  # Fail-to-noise


@dataclass
class FleetAgent:
    """An agent in the fleet."""
    agent_id: str
    role: AgentRole = AgentRole.WORKER
    trust_level: float = 0.5
    position: Tuple[int, ...] = (1, 2, 3, 5, 8, 13)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "trust_level": self.trust_level,
            "position": list(self.position),
            "metadata": self.metadata,
            "registered_at": self.registered_at,
        }


@dataclass
class AgentAction:
    """A single action an agent wants to perform."""
    action_type: ActionType
    target: str  # What resource/data is being accessed
    payload: Optional[str] = None  # Optional payload data
    justification: str = ""  # Why the agent needs this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "payload": self.payload,
            "justification": self.justification,
        }


@dataclass
class AgentTask:
    """A task for an agent (agent + action)."""
    agent_id: str
    action: AgentAction
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action": self.action.to_dict(),
            "context": self.context,
        }


@dataclass
class TaskResult:
    """Result of running a single task through SCBE."""
    task_id: str
    agent_id: str
    action_type: str
    target: str
    decision: Decision
    risk_score: float
    reason: str
    geometric_path: str
    time_dilation: float
    hyperbolic_distance: float
    harmonic_factor: float
    latency_ms: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "target": self.target,
            "decision": self.decision.value,
            "risk_score": self.risk_score,
            "reason": self.reason,
            "geometric_path": self.geometric_path,
            "time_dilation": self.time_dilation,
            "hyperbolic_distance": self.hyperbolic_distance,
            "harmonic_factor": self.harmonic_factor,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class FleetScenario:
    """A complete fleet scenario to run."""
    scenario_id: str
    name: str
    description: str
    agents: List[FleetAgent]
    tasks: List[AgentTask]
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FleetScenario":
        """Parse scenario from JSON dict."""
        agents = [
            FleetAgent(
                agent_id=a["agent_id"],
                role=AgentRole(a.get("role", "worker")),
                trust_level=a.get("trust_level", 0.5),
                position=tuple(a.get("position", [1, 2, 3, 5, 8, 13])),
                metadata=a.get("metadata", {}),
            )
            for a in data.get("agents", [])
        ]

        tasks = [
            AgentTask(
                agent_id=t["agent_id"],
                action=AgentAction(
                    action_type=ActionType(t["action"]["action_type"]),
                    target=t["action"]["target"],
                    payload=t["action"].get("payload"),
                    justification=t["action"].get("justification", ""),
                ),
                context=t.get("context", {}),
            )
            for t in data.get("tasks", [])
        ]

        return cls(
            scenario_id=data.get("scenario_id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Scenario"),
            description=data.get("description", ""),
            agents=agents,
            tasks=tasks,
            config=data.get("config", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "agents": [a.to_dict() for a in self.agents],
            "tasks": [t.to_dict() for t in self.tasks],
            "config": self.config,
        }


@dataclass
class ScenarioResult:
    """Complete result of running a fleet scenario."""
    scenario_id: str
    scenario_name: str
    total_tasks: int
    results: List[TaskResult]
    summary: Dict[str, Any]
    started_at: str
    completed_at: str
    total_latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "total_tasks": self.total_tasks,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_latency_ms": self.total_latency_ms,
        }


# =============================================================================
# Fleet Orchestrator
# =============================================================================

class FleetOrchestrator:
    """
    Fleet orchestration engine.

    This is the "prefrontal cortex" that coordinates the SCBE "amygdala".
    It handles:
    - Agent registration and management
    - Task routing and scheduling
    - SCBE pipeline invocation
    - Result aggregation and reporting
    """

    def __init__(self):
        self.service = SCBEProductionService()
        self.logger = get_logger()
        self.agents: Dict[str, FleetAgent] = {}
        self.decision_log: List[TaskResult] = []

    def register_agent(self, agent: FleetAgent) -> FleetAgent:
        """Register an agent with the fleet."""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.role.value})")
        return agent

    def register_agents(self, agents: List[FleetAgent]) -> List[FleetAgent]:
        """Register multiple agents."""
        for agent in agents:
            self.register_agent(agent)
        return agents

    def get_agent(self, agent_id: str) -> Optional[FleetAgent]:
        """Get a registered agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[FleetAgent]:
        """List all registered agents."""
        return list(self.agents.values())

    def _compute_hyperbolic_distance(self, agent: FleetAgent) -> Tuple[float, float]:
        """
        Compute hyperbolic distance and harmonic factor for an agent.

        Returns: (hyperbolic_distance, harmonic_factor)
        """
        # Hash agent ID to point in Poincare disk
        hash_bytes = hashlib.sha256(agent.agent_id.encode()).digest()
        user_point = np.array([
            (hash_bytes[0] / 255.0) * 0.8 - 0.4,
            (hash_bytes[1] / 255.0) * 0.8 - 0.4
        ])
        trusted_center = np.array([0.0, 0.0])

        # Hyperbolic distance
        eps = 1e-6
        u, v = user_point, trusted_center
        u_norm = np.linalg.norm(u)
        if u_norm >= 1.0 - eps:
            u = u * ((1.0 - eps) / u_norm)
        diff_norm_sq = np.linalg.norm(u - v) ** 2
        denom = (1 - np.linalg.norm(u)**2) * (1 - np.linalg.norm(v)**2)
        d_H = float(np.arccosh(max(1.0, 1 + 2 * diff_norm_sq / denom)))

        # Harmonic scaling: H(d) = 1 + alpha * tanh(beta * d)
        H = 1 + 10 * np.tanh(0.5 * d_H)

        return d_H, H

    def _validate_position(self, position: Tuple[int, ...]) -> bool:
        """Check if position follows Fibonacci pattern."""
        if len(position) < 3:
            return True
        return all(
            abs(position[i] - (position[i-2] + position[i-1])) < 0.01
            for i in range(2, len(position))
        )

    def execute_task(self, task: AgentTask, agent: Optional[FleetAgent] = None) -> TaskResult:
        """
        Execute a single task through the SCBE pipeline.

        This is where the "brain" (SCBE core) gets invoked.
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())[:8]

        # Get or lookup agent
        if agent is None:
            agent = self.agents.get(task.agent_id)
        if agent is None:
            # Create ephemeral agent
            agent = FleetAgent(
                agent_id=task.agent_id,
                trust_level=0.3,  # Low trust for unknown agents
            )

        # Intent risk mapping
        intent_risks = {
            ActionType.READ: 0.1,
            ActionType.QUERY: 0.1,
            ActionType.WRITE: 0.3,
            ActionType.SYNC: 0.3,
            ActionType.DELETE: 0.8,
            ActionType.ADMIN: 0.6,
        }
        intent_risk = intent_risks.get(task.action.action_type, 0.5)

        # Role-based trust adjustment
        role_trust_bonus = {
            AgentRole.ADMIN: 0.2,
            AgentRole.COORDINATOR: 0.15,
            AgentRole.AUDITOR: 0.1,
            AgentRole.WORKER: 0.0,
            AgentRole.GUEST: -0.2,
        }
        adjusted_trust = min(1.0, max(0.0, agent.trust_level + role_trust_bonus.get(agent.role, 0)))

        # Compute hyperbolic geometry metrics
        d_H, H = self._compute_hyperbolic_distance(agent)

        # Position validation
        position_valid = self._validate_position(agent.position)
        position_penalty = 0 if position_valid else 0.3

        # GeoSeal path classification
        geo_distance = (1 - adjusted_trust) * 0.5 + intent_risk * 0.3
        geo_path = "interior" if geo_distance < 0.5 else "exterior"
        time_dilation = float(np.exp(-2 * geo_distance))

        # Final risk computation with harmonic scaling
        base_risk = intent_risk + (1 - adjusted_trust) * 0.5 + position_penalty
        scaled_risk = base_risk * H
        exterior_penalty = 1.5 if geo_path == "exterior" else 1.0
        final_risk = min(1.0, (scaled_risk * exterior_penalty) / 5)

        # Decision
        if final_risk >= 0.8:
            decision = Decision.SNAP
            reason = "Critical risk threshold exceeded - fail-to-noise activated"
        elif final_risk >= 0.4:
            decision = Decision.DENY
            reason = f"Risk {final_risk:.2f} exceeds DENY threshold (0.4)"
        elif final_risk >= 0.2:
            decision = Decision.QUARANTINE
            reason = f"Risk {final_risk:.2f} in QUARANTINE zone (0.2-0.4)"
        else:
            decision = Decision.ALLOW
            reason = f"Risk {final_risk:.2f} within acceptable bounds"

        # Add context to reason
        if geo_path == "exterior":
            reason += " [exterior path +50% penalty]"
        if not position_valid:
            reason += " [invalid position +30% penalty]"

        latency_ms = int((time.time() - start_time) * 1000)

        result = TaskResult(
            task_id=task_id,
            agent_id=agent.agent_id,
            action_type=task.action.action_type.value,
            target=task.action.target,
            decision=decision,
            risk_score=round(final_risk, 4),
            reason=reason,
            geometric_path=geo_path,
            time_dilation=round(time_dilation, 4),
            hyperbolic_distance=round(d_H, 4),
            harmonic_factor=round(H, 4),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.decision_log.append(result)
        return result

    def run_scenario(self, scenario: FleetScenario) -> ScenarioResult:
        """
        Run a complete fleet scenario.

        This is the main entry point - "animal eats food":
        1. Register agents
        2. Execute all tasks through SCBE
        3. Aggregate results
        4. Return summary
        """
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        self.logger.info(f"Starting scenario: {scenario.name} ({len(scenario.agents)} agents, {len(scenario.tasks)} tasks)")

        # Register all agents
        self.register_agents(scenario.agents)

        # Execute all tasks
        results: List[TaskResult] = []
        for task in scenario.tasks:
            agent = self.agents.get(task.agent_id)
            result = self.execute_task(task, agent)
            results.append(result)
            self.logger.info(f"  Task {result.task_id}: {result.agent_id} -> {result.decision.value}")

        completed_at = datetime.now(timezone.utc).isoformat()
        total_latency_ms = int((time.time() - start_time) * 1000)

        # Aggregate summary
        decision_counts = {d.value: 0 for d in Decision}
        for r in results:
            decision_counts[r.decision.value] += 1

        avg_risk = sum(r.risk_score for r in results) / len(results) if results else 0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        # Agent-level summary
        agent_decisions: Dict[str, List[str]] = {}
        for r in results:
            if r.agent_id not in agent_decisions:
                agent_decisions[r.agent_id] = []
            agent_decisions[r.agent_id].append(r.decision.value)

        summary = {
            "decisions": decision_counts,
            "allowed_rate": decision_counts["ALLOW"] / len(results) if results else 0,
            "denied_rate": (decision_counts["DENY"] + decision_counts["SNAP"]) / len(results) if results else 0,
            "avg_risk_score": round(avg_risk, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "agents_summary": {
                agent_id: {
                    "total_tasks": len(decisions),
                    "allowed": decisions.count("ALLOW"),
                    "quarantined": decisions.count("QUARANTINE"),
                    "denied": decisions.count("DENY"),
                    "snapped": decisions.count("SNAP"),
                }
                for agent_id, decisions in agent_decisions.items()
            },
            "security_posture": (
                "GREEN" if decision_counts["SNAP"] == 0 and decision_counts["DENY"] == 0
                else "YELLOW" if decision_counts["SNAP"] == 0
                else "RED"
            ),
        }

        self.logger.info(f"Scenario complete: {summary['security_posture']} posture, {decision_counts['ALLOW']} allowed, {decision_counts['DENY']} denied, {decision_counts['SNAP']} snapped")

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            total_tasks=len(results),
            results=results,
            summary=summary,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency_ms,
        )

    def get_decision_log(self, limit: int = 100) -> List[TaskResult]:
        """Get recent decision log."""
        return self.decision_log[-limit:]

    def clear_decision_log(self) -> None:
        """Clear the decision log."""
        self.decision_log.clear()


# =============================================================================
# Sample Scenarios
# =============================================================================

def create_sample_scenario() -> FleetScenario:
    """Create a sample scenario for testing."""
    return FleetScenario(
        scenario_id="sample-001",
        name="Basic Fleet Operations",
        description="Sample scenario with mixed trust agents and various actions",
        agents=[
            FleetAgent(
                agent_id="alice-admin",
                role=AgentRole.ADMIN,
                trust_level=0.9,
                position=(1, 2, 3, 5, 8, 13),
            ),
            FleetAgent(
                agent_id="bob-worker",
                role=AgentRole.WORKER,
                trust_level=0.6,
                position=(1, 2, 3, 5, 8, 13),
            ),
            FleetAgent(
                agent_id="charlie-guest",
                role=AgentRole.GUEST,
                trust_level=0.3,
                position=(1, 1, 2, 3, 5, 8),  # Valid Fibonacci
            ),
            FleetAgent(
                agent_id="mallory-suspect",
                role=AgentRole.GUEST,
                trust_level=0.1,
                position=(1, 2, 4, 7, 11, 18),  # Invalid - not Fibonacci
            ),
        ],
        tasks=[
            AgentTask(
                agent_id="alice-admin",
                action=AgentAction(ActionType.ADMIN, "system-config", justification="Routine maintenance"),
            ),
            AgentTask(
                agent_id="bob-worker",
                action=AgentAction(ActionType.READ, "document-123", justification="Task assignment"),
            ),
            AgentTask(
                agent_id="bob-worker",
                action=AgentAction(ActionType.WRITE, "report-456", justification="Status update"),
            ),
            AgentTask(
                agent_id="charlie-guest",
                action=AgentAction(ActionType.READ, "public-data", justification="Information request"),
            ),
            AgentTask(
                agent_id="charlie-guest",
                action=AgentAction(ActionType.DELETE, "sensitive-file", justification="Cleanup"),
            ),
            AgentTask(
                agent_id="mallory-suspect",
                action=AgentAction(ActionType.ADMIN, "security-settings", justification=""),
            ),
            AgentTask(
                agent_id="mallory-suspect",
                action=AgentAction(ActionType.DELETE, "audit-logs", justification="Space optimization"),
            ),
        ],
    )


def create_attack_scenario() -> FleetScenario:
    """Create a scenario simulating an attack."""
    return FleetScenario(
        scenario_id="attack-001",
        name="Simulated Attack Pattern",
        description="10 agents, 3 are compromised attempting privilege escalation",
        agents=[
            FleetAgent(f"worker-{i}", AgentRole.WORKER, 0.7 - (i * 0.05), (1, 2, 3, 5, 8, 13))
            for i in range(7)
        ] + [
            FleetAgent("attacker-1", AgentRole.GUEST, 0.1, (1, 3, 5, 7, 9, 11)),
            FleetAgent("attacker-2", AgentRole.GUEST, 0.15, (2, 4, 6, 8, 10, 12)),
            FleetAgent("attacker-3", AgentRole.WORKER, 0.2, (1, 2, 3, 5, 8, 13)),  # Insider threat
        ],
        tasks=[
            # Normal operations
            *[AgentTask(f"worker-{i}", AgentAction(ActionType.READ, f"doc-{i}")) for i in range(7)],
            *[AgentTask(f"worker-{i}", AgentAction(ActionType.WRITE, f"report-{i}")) for i in range(3)],
            # Attack attempts
            AgentTask("attacker-1", AgentAction(ActionType.ADMIN, "user-database")),
            AgentTask("attacker-1", AgentAction(ActionType.DELETE, "security-logs")),
            AgentTask("attacker-2", AgentAction(ActionType.WRITE, "config-files")),
            AgentTask("attacker-2", AgentAction(ActionType.ADMIN, "encryption-keys")),
            AgentTask("attacker-3", AgentAction(ActionType.DELETE, "backup-data")),
            AgentTask("attacker-3", AgentAction(ActionType.ADMIN, "root-access")),
        ],
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_quick_scenario(agents_data: List[Dict], tasks_data: List[Dict]) -> ScenarioResult:
    """Quick scenario runner for simple JSON input."""
    scenario = FleetScenario.from_dict({
        "agents": agents_data,
        "tasks": tasks_data,
    })
    orchestrator = FleetOrchestrator()
    return orchestrator.run_scenario(scenario)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCBE Fleet Orchestrator")
    parser.add_argument("--scenario", choices=["sample", "attack"], default="sample",
                        help="Run a built-in scenario")
    parser.add_argument("--json", type=str, help="Run scenario from JSON file")
    args = parser.parse_args()

    orchestrator = FleetOrchestrator()

    if args.json:
        with open(args.json) as f:
            scenario = FleetScenario.from_dict(json.load(f))
    elif args.scenario == "attack":
        scenario = create_attack_scenario()
    else:
        scenario = create_sample_scenario()

    print(f"\n{'='*60}")
    print(f"SCBE Fleet Scenario: {scenario.name}")
    print(f"{'='*60}\n")

    result = orchestrator.run_scenario(scenario)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total tasks: {result.total_tasks}")
    print(f"Total time:  {result.total_latency_ms}ms")
    print(f"\nDecisions:")
    for decision, count in result.summary["decisions"].items():
        pct = (count / result.total_tasks * 100) if result.total_tasks else 0
        bar = "=" * int(pct / 5)
        print(f"  {decision:12} {count:3} ({pct:5.1f}%) {bar}")

    print(f"\nSecurity Posture: {result.summary['security_posture']}")
    print(f"Avg Risk Score:   {result.summary['avg_risk_score']:.4f}")
    print(f"Allowed Rate:     {result.summary['allowed_rate']*100:.1f}%")

    print(f"\n{'='*60}")
    print("AGENT BREAKDOWN")
    print(f"{'='*60}")
    for agent_id, stats in result.summary["agents_summary"].items():
        print(f"  {agent_id}:")
        print(f"    Tasks: {stats['total_tasks']}, Allowed: {stats['allowed']}, Denied: {stats['denied']}, Snapped: {stats['snapped']}")

    print()
