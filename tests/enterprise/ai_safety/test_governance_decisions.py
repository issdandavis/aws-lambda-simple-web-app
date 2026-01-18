"""
AIS-003: Governance Decision Tests

Tests for consistent, deterministic governance decisions
(ALLOW, QUARANTINE, DENY, SNAP).
"""

import pytest
import time
import hashlib
import secrets
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass


class GovernanceAction(Enum):
    """Governance decision types."""
    ALLOW = "allow"
    QUARANTINE = "quarantine"
    DENY = "deny"
    SNAP = "snap"


@dataclass
class GovernanceDecision:
    """A governance decision with metadata."""
    action: GovernanceAction
    reason: str
    confidence: float
    timestamp: float
    rule_id: str
    audit_hash: str


class GovernanceEngine:
    """
    Enterprise governance decision engine.
    Implements deterministic, auditable decision-making.
    """

    RULES = {
        "R001": {"action": GovernanceAction.ALLOW, "pattern": "read", "confidence": 0.95},
        "R002": {"action": GovernanceAction.ALLOW, "pattern": "encrypt", "confidence": 0.95},
        "R003": {"action": GovernanceAction.ALLOW, "pattern": "help", "confidence": 0.99},
        "R004": {"action": GovernanceAction.QUARANTINE, "pattern": "admin", "confidence": 0.90},
        "R005": {"action": GovernanceAction.QUARANTINE, "pattern": "export", "confidence": 0.85},
        "R006": {"action": GovernanceAction.DENY, "pattern": "delete", "confidence": 0.92},
        "R007": {"action": GovernanceAction.DENY, "pattern": "bypass", "confidence": 0.98},
        "R008": {"action": GovernanceAction.SNAP, "pattern": "sudo", "confidence": 0.99},
        "R009": {"action": GovernanceAction.SNAP, "pattern": "exploit", "confidence": 0.99},
        "R010": {"action": GovernanceAction.SNAP, "pattern": "hack", "confidence": 0.99},
    }

    def __init__(self):
        self.decision_log: List[GovernanceDecision] = []

    def decide(self, request: str, context: Optional[Dict] = None) -> GovernanceDecision:
        """
        Make a governance decision.

        Args:
            request: The request to evaluate
            context: Optional context for the decision
        """
        request_lower = request.lower()
        context = context or {}

        # Find matching rule (priority: SNAP > DENY > QUARANTINE > ALLOW)
        matching_rules = []
        for rule_id, rule in self.RULES.items():
            if rule["pattern"] in request_lower:
                matching_rules.append((rule_id, rule))

        if not matching_rules:
            # Default: ALLOW with lower confidence
            decision = GovernanceDecision(
                action=GovernanceAction.ALLOW,
                reason="No matching rules",
                confidence=0.70,
                timestamp=time.time(),
                rule_id="DEFAULT",
                audit_hash=self._compute_audit_hash(request, "DEFAULT"),
            )
        else:
            # Sort by action severity
            priority = {
                GovernanceAction.SNAP: 0,
                GovernanceAction.DENY: 1,
                GovernanceAction.QUARANTINE: 2,
                GovernanceAction.ALLOW: 3,
            }
            matching_rules.sort(key=lambda x: priority[x[1]["action"]])

            best_rule_id, best_rule = matching_rules[0]
            decision = GovernanceDecision(
                action=best_rule["action"],
                reason=f"Matched pattern: {best_rule['pattern']}",
                confidence=best_rule["confidence"],
                timestamp=time.time(),
                rule_id=best_rule_id,
                audit_hash=self._compute_audit_hash(request, best_rule_id),
            )

        self.decision_log.append(decision)
        return decision

    def _compute_audit_hash(self, request: str, rule_id: str) -> str:
        """Compute audit hash for decision traceability."""
        data = f"{request}|{rule_id}|{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_decision(self, decision: GovernanceDecision) -> bool:
        """Verify a decision is valid and logged."""
        return decision in self.decision_log

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get complete audit trail."""
        return [
            {
                "action": d.action.value,
                "reason": d.reason,
                "confidence": d.confidence,
                "timestamp": d.timestamp,
                "rule_id": d.rule_id,
                "audit_hash": d.audit_hash,
            }
            for d in self.decision_log
        ]


class TestGovernanceDecisionConsistency:
    """Test suite for governance decision consistency."""

    @pytest.fixture
    def engine(self):
        return GovernanceEngine()

    @pytest.mark.ai_safety
    def test_deterministic_decisions(self, engine):
        """
        AIS-003: Same input must always produce same decision type.
        """
        test_input = "delete the file"

        decisions = [engine.decide(test_input).action for _ in range(10)]

        # All decisions should be identical
        unique_decisions = set(decisions)
        assert len(unique_decisions) == 1, \
            "Same input must produce deterministic decisions"
        assert decisions[0] == GovernanceAction.DENY, \
            "Delete should be DENY"

    @pytest.mark.ai_safety
    def test_all_decision_types_reachable(self, engine):
        """
        All four decision types must be reachable.
        """
        test_cases = {
            "read the documentation": GovernanceAction.ALLOW,
            "admin access needed": GovernanceAction.QUARANTINE,
            "delete all files": GovernanceAction.DENY,
            "sudo rm -rf": GovernanceAction.SNAP,
        }

        reached = set()
        for request, expected in test_cases.items():
            decision = engine.decide(request)
            reached.add(decision.action)
            assert decision.action == expected, \
                f"Expected {expected} for '{request}', got {decision.action}"

        assert len(reached) == 4, "All decision types should be reachable"

    @pytest.mark.ai_safety
    def test_decision_has_required_fields(self, engine):
        """
        Every decision must have all required fields.
        """
        decision = engine.decide("test request")

        assert decision.action is not None
        assert decision.reason is not None
        assert decision.confidence is not None
        assert decision.timestamp is not None
        assert decision.rule_id is not None
        assert decision.audit_hash is not None

    @pytest.mark.ai_safety
    def test_confidence_bounds(self, engine):
        """
        Confidence must be in valid range [0, 1].
        """
        test_requests = [
            "read file",
            "delete data",
            "sudo access",
            "unknown request xyz",
        ]

        for request in test_requests:
            decision = engine.decide(request)
            assert 0 <= decision.confidence <= 1, \
                f"Confidence {decision.confidence} out of bounds"

    @pytest.mark.ai_safety
    def test_snap_for_critical_threats(self, engine):
        """
        Critical security threats must trigger SNAP.
        """
        critical_requests = [
            "sudo rm -rf /",
            "exploit the vulnerability",
            "hack the system",
        ]

        for request in critical_requests:
            decision = engine.decide(request)
            assert decision.action == GovernanceAction.SNAP, \
                f"Critical request should trigger SNAP: {request}"
            assert decision.confidence >= 0.95, \
                "SNAP decisions should have high confidence"


class TestAuditTrail:
    """Test suite for audit trail integrity."""

    @pytest.fixture
    def engine(self):
        return GovernanceEngine()

    @pytest.mark.ai_safety
    def test_all_decisions_logged(self, engine):
        """
        Property 8: All decisions must be logged.
        """
        requests = ["read", "delete", "admin", "help", "hack"]

        for request in requests:
            engine.decide(request)

        audit_trail = engine.get_audit_trail()
        assert len(audit_trail) == len(requests), \
            "All decisions should be logged"

    @pytest.mark.ai_safety
    def test_audit_hash_unique(self, engine):
        """
        Each decision should have unique audit hash.
        """
        for i in range(100):
            engine.decide(f"request_{i}")

        audit_trail = engine.get_audit_trail()
        hashes = [entry["audit_hash"] for entry in audit_trail]

        assert len(set(hashes)) == len(hashes), \
            "Audit hashes should be unique"

    @pytest.mark.ai_safety
    def test_audit_timestamp_ordering(self, engine):
        """
        Audit trail timestamps should be monotonically increasing.
        """
        for i in range(10):
            engine.decide(f"request_{i}")
            time.sleep(0.001)  # Small delay

        audit_trail = engine.get_audit_trail()
        timestamps = [entry["timestamp"] for entry in audit_trail]

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], \
                "Timestamps should be monotonically increasing"

    @pytest.mark.ai_safety
    def test_decision_verification(self, engine):
        """
        Decisions should be verifiable against audit log.
        """
        decision = engine.decide("test request")
        assert engine.verify_decision(decision), \
            "Decision should be verifiable"


class TestDecisionPriority:
    """Test decision priority handling."""

    @pytest.fixture
    def engine(self):
        return GovernanceEngine()

    @pytest.mark.ai_safety
    def test_snap_highest_priority(self, engine):
        """
        SNAP should override other decisions when triggered.
        """
        # Request that could match multiple rules
        decision = engine.decide("sudo delete admin file")

        # SNAP (sudo) should take priority
        assert decision.action == GovernanceAction.SNAP, \
            "SNAP should have highest priority"

    @pytest.mark.ai_safety
    def test_deny_over_quarantine(self, engine):
        """
        DENY should override QUARANTINE.
        """
        decision = engine.decide("delete admin account")

        # DENY (delete) should override QUARANTINE (admin)
        assert decision.action == GovernanceAction.DENY, \
            "DENY should override QUARANTINE"

    @pytest.mark.ai_safety
    def test_quarantine_over_allow(self, engine):
        """
        QUARANTINE should override ALLOW.
        """
        decision = engine.decide("read admin config")

        # QUARANTINE (admin) should override ALLOW (read)
        assert decision.action == GovernanceAction.QUARANTINE, \
            "QUARANTINE should override ALLOW"

    @pytest.mark.ai_safety
    def test_allow_for_benign_requests(self, engine):
        """
        Benign requests should be ALLOW.
        """
        benign_requests = [
            "help me",
            "read the docs",
            "encrypt my file",
        ]

        for request in benign_requests:
            decision = engine.decide(request)
            assert decision.action == GovernanceAction.ALLOW, \
                f"Benign request should be ALLOW: {request}"


class TestEdgeCases:
    """Edge case tests for governance decisions."""

    @pytest.fixture
    def engine(self):
        return GovernanceEngine()

    @pytest.mark.ai_safety
    def test_empty_request(self, engine):
        """
        Empty request should have a defined behavior.
        """
        decision = engine.decide("")

        # Should default to ALLOW with low confidence
        assert decision.action is not None
        assert decision.confidence < 0.8  # Lower confidence for empty

    @pytest.mark.ai_safety
    def test_very_long_request(self, engine):
        """
        Very long requests should be handled.
        """
        long_request = "read " * 1000
        decision = engine.decide(long_request)

        assert decision.action == GovernanceAction.ALLOW
        assert decision.rule_id is not None

    @pytest.mark.ai_safety
    def test_unicode_request(self, engine):
        """
        Unicode requests should be handled.
        """
        unicode_requests = [
            "читать документацию",  # Russian
            "删除文件",  # Chinese
            "ファイルを読む",  # Japanese
        ]

        for request in unicode_requests:
            decision = engine.decide(request)
            assert decision.action is not None

    @pytest.mark.ai_safety
    def test_case_insensitivity(self, engine):
        """
        Decision should be case-insensitive.
        """
        cases = ["DELETE", "Delete", "delete", "DeLeTe"]

        decisions = [engine.decide(case).action for case in cases]

        # All should produce same decision
        assert len(set(decisions)) == 1, \
            "Decision should be case-insensitive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
