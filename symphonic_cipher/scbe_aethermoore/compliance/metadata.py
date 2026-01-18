"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLIANCE METADATA SYSTEM                                ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Decorators and classes for adding compliance metadata to tests              ║
║                                                                              ║
║  Features:                                                                   ║
║  - @compliance_test decorator for marking tests                              ║
║  - Automatic layer and standard detection                                    ║
║  - Rich metadata capture (timing, outcomes, artifacts)                       ║
║  - Integration with pytest markers                                           ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import functools
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .layer_mapping import (
    SCBELayer,
    ComplianceStandard,
    LayerMapping,
    get_layer_for_test,
    get_axioms_for_layers,
)


class TestOutcome(Enum):
    """Test execution outcome."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    XFAIL = "xfail"      # Expected failure
    XPASS = "xpass"      # Unexpected pass


class RiskLevel(Enum):
    """Risk level for compliance failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceMetadata:
    """
    Complete compliance metadata for a test.

    This captures all information needed for compliance reporting,
    including SCBE layer mappings, standards coverage, and audit trail.
    """
    # Test identification
    test_id: str
    test_name: str
    test_class: str = ""
    test_module: str = ""

    # Compliance mappings
    category: str = ""
    layers: Set[SCBELayer] = field(default_factory=set)
    standards: Set[ComplianceStandard] = field(default_factory=set)
    axioms: Set[str] = field(default_factory=set)

    # Risk and priority
    risk_level: RiskLevel = RiskLevel.MEDIUM
    priority: int = 5  # 1-10, 1 = highest

    # Requirements traceability
    requirement_ids: List[str] = field(default_factory=list)
    control_ids: List[str] = field(default_factory=list)  # NIST control IDs

    # Documentation
    description: str = ""
    rationale: str = ""
    evidence_required: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'test_class': self.test_class,
            'test_module': self.test_module,
            'category': self.category,
            'layers': [l.name for l in self.layers],
            'layer_numbers': sorted([l.value for l in self.layers]),
            'standards': [s.value for s in self.standards],
            'axioms': list(self.axioms),
            'risk_level': self.risk_level.value,
            'priority': self.priority,
            'requirement_ids': self.requirement_ids,
            'control_ids': self.control_ids,
            'description': self.description,
            'rationale': self.rationale,
            'evidence_required': self.evidence_required,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }

    @classmethod
    def from_mapping(cls, mapping: LayerMapping, **kwargs) -> ComplianceMetadata:
        """Create metadata from a LayerMapping."""
        return cls(
            test_id=mapping.test_id,
            test_name=mapping.test_name,
            category=mapping.category,
            layers=mapping.layers,
            standards=mapping.standards,
            axioms=mapping.axioms,
            description=mapping.description,
            **kwargs,
        )


@dataclass
class TestResult:
    """
    Complete test result with compliance metadata.

    Captures both the test execution details and compliance context.
    """
    # Metadata
    metadata: ComplianceMetadata

    # Execution details
    outcome: TestOutcome
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Error details (if any)
    error_message: str = ""
    error_traceback: str = ""

    # Artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Audit trail
    audit_notes: List[str] = field(default_factory=list)
    reviewer: str = ""
    review_date: Optional[datetime] = None

    @property
    def is_passing(self) -> bool:
        """Check if test passed."""
        return self.outcome in {TestOutcome.PASSED, TestOutcome.XFAIL}

    @property
    def requires_attention(self) -> bool:
        """Check if test requires attention."""
        return self.outcome in {TestOutcome.FAILED, TestOutcome.ERROR}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'metadata': self.metadata.to_dict(),
            'outcome': self.outcome.value,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat(),
            'is_passing': self.is_passing,
            'requires_attention': self.requires_attention,
            'audit_notes': self.audit_notes,
        }
        if self.error_message:
            result['error_message'] = self.error_message
            result['error_traceback'] = self.error_traceback
        if self.artifacts:
            result['artifacts'] = self.artifacts
        if self.reviewer:
            result['reviewer'] = self.reviewer
            result['review_date'] = self.review_date.isoformat() if self.review_date else None
        return result


# Global registry for compliance metadata
_COMPLIANCE_REGISTRY: Dict[str, ComplianceMetadata] = {}
_TEST_RESULTS: List[TestResult] = []


def compliance_test(
    category: str,
    standards: Optional[Set[ComplianceStandard]] = None,
    layers: Optional[Set[SCBELayer]] = None,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    priority: int = 5,
    requirement_ids: Optional[List[str]] = None,
    control_ids: Optional[List[str]] = None,
    description: str = "",
    rationale: str = "",
):
    """
    Decorator for marking a test as compliance-tracked.

    Usage:
        @compliance_test(
            category="medical_phi",
            standards={ComplianceStandard.HIPAA},
            risk_level=RiskLevel.HIGH,
            description="Validates PHI encryption",
        )
        def test_phi_encryption(self):
            ...

    Args:
        category: Test category for layer mapping
        standards: Compliance standards (auto-detected if None)
        layers: SCBE layers (auto-detected if None)
        risk_level: Risk level for failures
        priority: Priority (1-10, 1 = highest)
        requirement_ids: Traceability requirement IDs
        control_ids: NIST control IDs
        description: Test description
        rationale: Why this test is needed
    """
    def decorator(func: Callable) -> Callable:
        # Get test identification
        test_name = func.__name__
        test_module = func.__module__ if hasattr(func, '__module__') else ""
        test_class = ""

        # Build layer mapping
        mapping = get_layer_for_test(
            test_name=test_name,
            test_id=test_name,
            category=category,
            standards=standards,
            description=description,
        )

        # Override layers if explicitly provided
        if layers:
            mapping.layers = layers
            mapping.axioms = get_axioms_for_layers(layers)

        # Create metadata
        metadata = ComplianceMetadata(
            test_id=test_name,
            test_name=test_name,
            test_module=test_module,
            test_class=test_class,
            category=category,
            layers=mapping.layers,
            standards=mapping.standards,
            axioms=mapping.axioms,
            risk_level=risk_level,
            priority=priority,
            requirement_ids=requirement_ids or [],
            control_ids=control_ids or [],
            description=description,
            rationale=rationale,
        )

        # Register metadata
        _COMPLIANCE_REGISTRY[test_name] = metadata

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            outcome = TestOutcome.PASSED
            error_msg = ""
            error_tb = ""

            try:
                result = func(*args, **kwargs)
            except AssertionError as e:
                outcome = TestOutcome.FAILED
                error_msg = str(e)
                error_tb = traceback.format_exc()
                raise
            except Exception as e:
                outcome = TestOutcome.ERROR
                error_msg = str(e)
                error_tb = traceback.format_exc()
                raise
            finally:
                duration = (time.perf_counter() - start_time) * 1000

                # Record result
                test_result = TestResult(
                    metadata=metadata,
                    outcome=outcome,
                    duration_ms=duration,
                    error_message=error_msg,
                    error_traceback=error_tb,
                )
                _TEST_RESULTS.append(test_result)

            return result

        # Attach metadata to function for introspection
        wrapper._compliance_metadata = metadata
        return wrapper

    return decorator


def extract_compliance_metadata(func: Callable) -> Optional[ComplianceMetadata]:
    """Extract compliance metadata from a decorated function."""
    return getattr(func, '_compliance_metadata', None)


def get_registered_metadata() -> Dict[str, ComplianceMetadata]:
    """Get all registered compliance metadata."""
    return _COMPLIANCE_REGISTRY.copy()


def get_test_results() -> List[TestResult]:
    """Get all recorded test results."""
    return _TEST_RESULTS.copy()


def clear_test_results():
    """Clear recorded test results."""
    _TEST_RESULTS.clear()


def get_results_by_standard(standard: ComplianceStandard) -> List[TestResult]:
    """Get test results for a specific compliance standard."""
    return [
        r for r in _TEST_RESULTS
        if standard in r.metadata.standards
    ]


def get_results_by_layer(layer: SCBELayer) -> List[TestResult]:
    """Get test results for a specific SCBE layer."""
    return [
        r for r in _TEST_RESULTS
        if layer in r.metadata.layers
    ]


def get_results_by_outcome(outcome: TestOutcome) -> List[TestResult]:
    """Get test results by outcome."""
    return [r for r in _TEST_RESULTS if r.outcome == outcome]


def get_failing_results() -> List[TestResult]:
    """Get all failing test results."""
    return [r for r in _TEST_RESULTS if r.requires_attention]


def get_compliance_summary() -> Dict[str, Any]:
    """Get a summary of compliance test results."""
    total = len(_TEST_RESULTS)
    if total == 0:
        return {'total': 0, 'message': 'No tests recorded'}

    passed = len([r for r in _TEST_RESULTS if r.outcome == TestOutcome.PASSED])
    failed = len([r for r in _TEST_RESULTS if r.outcome == TestOutcome.FAILED])
    errors = len([r for r in _TEST_RESULTS if r.outcome == TestOutcome.ERROR])

    # Standards coverage
    standards_tested = set()
    standards_passed = set()
    for result in _TEST_RESULTS:
        standards_tested.update(result.metadata.standards)
        if result.is_passing:
            standards_passed.update(result.metadata.standards)

    # Layers covered
    layers_tested = set()
    layers_passed = set()
    for result in _TEST_RESULTS:
        layers_tested.update(result.metadata.layers)
        if result.is_passing:
            layers_passed.update(result.metadata.layers)

    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': passed / total if total > 0 else 0,
        'standards_tested': len(standards_tested),
        'standards_passed': len(standards_passed),
        'layers_tested': len(layers_tested),
        'layers_passed': len(layers_passed),
        'layer_coverage': len(layers_tested) / 14,  # 14 total layers
    }
