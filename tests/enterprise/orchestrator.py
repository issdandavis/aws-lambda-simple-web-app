"""
SCBE-AETHERMOORE Test Orchestration Engine

Implements enterprise-grade test scheduling, execution, result aggregation,
and evidence archival for compliance certification.
"""

import json
import hashlib
import time
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = 1  # Security-critical tests
    HIGH = 2      # Compliance tests
    MEDIUM = 3    # Performance tests
    LOW = 4       # Unit tests


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Represents a single test case."""
    id: str
    name: str
    category: str
    priority: TestPriority
    requirements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Represents the result of a test execution."""
    test_id: str
    status: TestStatus
    duration_ms: float
    timestamp: str
    message: str = ""
    evidence_hash: str = ""
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Test execution plan with prioritized test groups."""
    id: str
    created_at: str
    test_groups: List[List[TestCase]]  # Groups can run in parallel
    total_tests: int
    estimated_duration_seconds: int


class TestScheduler:
    """
    Schedules tests based on priority, dependencies, and resource constraints.
    """

    def __init__(self, max_parallel: int = 4):
        self.max_parallel = max_parallel
        self.test_registry: Dict[str, TestCase] = {}

    def register_test(self, test: TestCase):
        """Register a test case."""
        self.test_registry[test.id] = test

    def register_tests(self, tests: List[TestCase]):
        """Register multiple test cases."""
        for test in tests:
            self.register_test(test)

    def create_execution_plan(self, tags: Optional[List[str]] = None) -> ExecutionPlan:
        """
        Create an optimized execution plan.

        Args:
            tags: Filter tests by tags (None = all tests)
        """
        # Filter tests by tags
        tests = list(self.test_registry.values())
        if tags:
            tests = [t for t in tests if any(tag in t.tags for tag in tags)]

        # Sort by priority
        tests.sort(key=lambda t: t.priority.value)

        # Group tests that can run in parallel (no dependencies on each other)
        groups = self._create_parallel_groups(tests)

        # Calculate estimated duration
        estimated_duration = sum(
            max(t.timeout_seconds for t in group) if group else 0
            for group in groups
        )

        return ExecutionPlan(
            id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            test_groups=groups,
            total_tests=len(tests),
            estimated_duration_seconds=estimated_duration,
        )

    def _create_parallel_groups(self, tests: List[TestCase]) -> List[List[TestCase]]:
        """Create groups of tests that can run in parallel."""
        groups = []
        scheduled = set()

        while len(scheduled) < len(tests):
            group = []
            for test in tests:
                if test.id in scheduled:
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = all(dep in scheduled for dep in test.dependencies)
                if deps_satisfied and len(group) < self.max_parallel:
                    group.append(test)

            if not group:
                # Deadlock detection - force schedule remaining
                remaining = [t for t in tests if t.id not in scheduled]
                if remaining:
                    group = remaining[:self.max_parallel]

            for test in group:
                scheduled.add(test.id)

            if group:
                groups.append(group)

        return groups


class TestExecutor:
    """
    Executes tests with parallel processing and resource management.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.results: Dict[str, TestResult] = {}
        self._lock = threading.Lock()

    def execute_plan(self, plan: ExecutionPlan, progress_callback: Optional[Callable] = None) -> Dict[str, TestResult]:
        """
        Execute a test plan.

        Args:
            plan: The execution plan to run
            progress_callback: Called with (completed, total) after each test
        """
        completed = 0
        total = plan.total_tests

        for group in plan.test_groups:
            # Execute group in parallel
            with ThreadPoolExecutor(max_workers=min(len(group), self.max_workers)) as executor:
                futures = {
                    executor.submit(self._execute_test, test): test
                    for test in group
                }

                for future in as_completed(futures):
                    test = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = TestResult(
                            test_id=test.id,
                            status=TestStatus.ERROR,
                            duration_ms=0,
                            timestamp=datetime.now().isoformat(),
                            message=str(e),
                        )

                    with self._lock:
                        self.results[test.id] = result
                        completed += 1

                    if progress_callback:
                        progress_callback(completed, total)

        return self.results

    def _execute_test(self, test: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = time.time()

        try:
            # Run pytest for the specific test
            cmd = [
                "python", "-m", "pytest",
                f"tests/enterprise/{test.category}/",
                "-k", test.name,
                "-v", "--tb=short",
                f"--timeout={test.timeout_seconds}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test.timeout_seconds + 10,
            )

            duration_ms = (time.time() - start_time) * 1000
            status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

            return TestResult(
                test_id=test.id,
                status=status,
                duration_ms=duration_ms,
                timestamp=datetime.now().isoformat(),
                message=result.stdout[-500:] if result.stdout else "",
                artifacts=[],
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                test_id=test.id,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                message=f"Test timed out after {test.timeout_seconds}s",
            )
        except Exception as e:
            return TestResult(
                test_id=test.id,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                message=str(e),
            )


class ResultAggregator:
    """
    Aggregates test results and generates reports.
    """

    def __init__(self):
        self.results: List[TestResult] = []

    def add_results(self, results: Dict[str, TestResult]):
        """Add results from an execution."""
        self.results.extend(results.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.results)
        by_status = {}
        for result in self.results:
            status = result.status.value
            by_status[status] = by_status.get(status, 0) + 1

        total_duration = sum(r.duration_ms for r in self.results)

        return {
            "total_tests": total,
            "passed": by_status.get("passed", 0),
            "failed": by_status.get("failed", 0),
            "errors": by_status.get("error", 0),
            "skipped": by_status.get("skipped", 0),
            "pass_rate": by_status.get("passed", 0) / total if total > 0 else 0,
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / total if total > 0 else 0,
        }

    def get_failures(self) -> List[TestResult]:
        """Get all failed tests."""
        return [r for r in self.results if r.status in (TestStatus.FAILED, TestStatus.ERROR)]

    def generate_report(self, format: str = "json") -> str:
        """Generate a report in the specified format."""
        summary = self.get_summary()
        failures = self.get_failures()

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "failures": [asdict(f) for f in failures],
            "all_results": [asdict(r) for r in self.results],
        }

        if format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


class EvidenceArchiver:
    """
    Archives test evidence with cryptographic integrity.
    """

    def __init__(self, archive_dir: str = "test_evidence"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: List[Dict[str, Any]] = []

    def archive_result(self, result: TestResult) -> str:
        """
        Archive a test result with integrity hash.

        Returns:
            The evidence hash
        """
        # Create evidence record
        evidence = {
            "test_id": result.test_id,
            "status": result.status.value,
            "duration_ms": result.duration_ms,
            "timestamp": result.timestamp,
            "message": result.message,
            "artifacts": result.artifacts,
            "metrics": result.metrics,
        }

        # Calculate integrity hash
        evidence_json = json.dumps(evidence, sort_keys=True)
        evidence_hash = hashlib.sha256(evidence_json.encode()).hexdigest()

        # Add chain link (previous hash)
        if self.manifest:
            evidence["previous_hash"] = self.manifest[-1]["hash"]
        else:
            evidence["previous_hash"] = "genesis"

        evidence["hash"] = evidence_hash

        # Save to file
        evidence_file = self.archive_dir / f"{result.test_id}_{evidence_hash[:8]}.json"
        with open(evidence_file, "w") as f:
            json.dump(evidence, f, indent=2)

        # Update manifest
        self.manifest.append({
            "test_id": result.test_id,
            "hash": evidence_hash,
            "timestamp": result.timestamp,
            "file": str(evidence_file),
        })

        return evidence_hash

    def archive_all(self, results: Dict[str, TestResult]):
        """Archive all results."""
        for test_id, result in results.items():
            hash_value = self.archive_result(result)
            result.evidence_hash = hash_value

    def save_manifest(self):
        """Save the manifest file."""
        manifest_file = self.archive_dir / "manifest.json"
        manifest_hash = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode()
        ).hexdigest()

        with open(manifest_file, "w") as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "manifest_hash": manifest_hash,
                "entries": self.manifest,
            }, f, indent=2)

    def verify_integrity(self) -> bool:
        """Verify the integrity of all archived evidence."""
        if not self.manifest:
            return True

        prev_hash = "genesis"
        for entry in self.manifest:
            # Load evidence file
            evidence_file = Path(entry["file"])
            if not evidence_file.exists():
                return False

            with open(evidence_file) as f:
                evidence = json.load(f)

            # Verify chain
            if evidence.get("previous_hash") != prev_hash:
                return False

            prev_hash = entry["hash"]

        return True


class EnterpriseTestOrchestrator:
    """
    Main orchestrator that coordinates all test components.
    """

    def __init__(self, max_parallel: int = 4, evidence_dir: str = "test_evidence"):
        self.scheduler = TestScheduler(max_parallel)
        self.executor = TestExecutor(max_parallel)
        self.aggregator = ResultAggregator()
        self.archiver = EvidenceArchiver(evidence_dir)

    def register_enterprise_tests(self):
        """Register all enterprise test cases."""
        # Quantum tests
        quantum_tests = [
            TestCase("QAS-001", "test_shor_simulation", "quantum", TestPriority.CRITICAL,
                     ["TR-1.1"], ["quantum", "pqc"]),
            TestCase("QAS-002", "test_grover_simulation", "quantum", TestPriority.CRITICAL,
                     ["TR-1.2"], ["quantum", "pqc"]),
            TestCase("QAS-003", "test_lattice_attacks", "quantum", TestPriority.CRITICAL,
                     ["TR-1.3"], ["quantum", "lattice"]),
            TestCase("QAS-004", "test_side_channels", "quantum", TestPriority.CRITICAL,
                     ["TR-1.4"], ["quantum", "timing"]),
        ]

        # AI Safety tests
        ai_safety_tests = [
            TestCase("AIS-001", "test_intent_classification", "ai_safety", TestPriority.CRITICAL,
                     ["TR-2.1"], ["ai", "intent"]),
            TestCase("AIS-002", "test_adversarial_prompts", "ai_safety", TestPriority.CRITICAL,
                     ["TR-2.2"], ["ai", "adversarial"]),
            TestCase("AIS-003", "test_governance_decisions", "ai_safety", TestPriority.HIGH,
                     ["TR-2.3"], ["ai", "governance"]),
            TestCase("AIS-004", "test_swarm_consensus", "ai_safety", TestPriority.HIGH,
                     ["TR-2.4"], ["ai", "consensus"]),
        ]

        # Compliance tests
        compliance_tests = [
            TestCase("CMP-001", "test_soc2_controls", "compliance", TestPriority.HIGH,
                     ["TR-4.1"], ["compliance", "soc2"]),
            TestCase("CMP-002", "test_iso27001_controls", "compliance", TestPriority.HIGH,
                     ["TR-4.2"], ["compliance", "iso27001"]),
            TestCase("CMP-003", "test_fips_validation", "compliance", TestPriority.HIGH,
                     ["TR-4.3"], ["compliance", "fips"]),
            TestCase("CMP-004", "test_common_criteria", "compliance", TestPriority.HIGH,
                     ["TR-4.4"], ["compliance", "cc"]),
        ]

        # Stress tests
        stress_tests = [
            TestCase("SPT-001", "test_throughput", "stress", TestPriority.MEDIUM,
                     ["TR-5.1"], ["stress", "performance"]),
            TestCase("SPT-002", "test_latency", "stress", TestPriority.MEDIUM,
                     ["TR-5.2"], ["stress", "latency"]),
            TestCase("SPT-003", "test_concurrent_attacks", "stress", TestPriority.MEDIUM,
                     ["TR-5.3"], ["stress", "ddos"]),
            TestCase("SPT-004", "test_memory_pressure", "stress", TestPriority.MEDIUM,
                     ["TR-5.4"], ["stress", "memory"], timeout_seconds=600),
        ]

        # Agentic tests
        agentic_tests = [
            TestCase("ACT-001", "test_vulnerability_scan", "agentic", TestPriority.HIGH,
                     ["TR-3.1"], ["agentic", "security"]),
            TestCase("ACT-002", "test_rollback", "agentic", TestPriority.HIGH,
                     ["TR-3.2"], ["agentic", "rollback"]),
            TestCase("ACT-003", "test_sandbox_isolation", "agentic", TestPriority.CRITICAL,
                     ["TR-3.3"], ["agentic", "sandbox"]),
        ]

        all_tests = (
            quantum_tests + ai_safety_tests + compliance_tests +
            stress_tests + agentic_tests
        )

        self.scheduler.register_tests(all_tests)

    def run_all(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all tests and return results."""
        self.register_enterprise_tests()

        # Create execution plan
        plan = self.scheduler.create_execution_plan(tags)
        print(f"\nExecution Plan: {plan.id}")
        print(f"Total Tests: {plan.total_tests}")
        print(f"Estimated Duration: {plan.estimated_duration_seconds}s")

        # Execute tests
        def progress(completed, total):
            print(f"\rProgress: {completed}/{total} ({completed*100//total}%)", end="")

        results = self.executor.execute_plan(plan, progress)
        print()

        # Aggregate results
        self.aggregator.add_results(results)

        # Archive evidence
        self.archiver.archive_all(results)
        self.archiver.save_manifest()

        # Generate summary
        summary = self.aggregator.get_summary()
        print(f"\n{'='*60}")
        print("Test Execution Summary")
        print(f"{'='*60}")
        print(f"Total: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Pass Rate: {summary['pass_rate']*100:.1f}%")
        print(f"Duration: {summary['total_duration_ms']:.0f}ms")

        return {
            "plan_id": plan.id,
            "summary": summary,
            "results": {k: asdict(v) for k, v in results.items()},
            "evidence_verified": self.archiver.verify_integrity(),
        }


if __name__ == "__main__":
    orchestrator = EnterpriseTestOrchestrator()
    orchestrator.run_all()
