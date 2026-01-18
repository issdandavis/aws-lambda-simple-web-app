"""
SPT-002: Latency Percentile Tests

Tests system latency at various percentiles.
Target: p99 < 10ms for all critical operations.
"""

import pytest
import time
import hashlib
import secrets
import statistics
from typing import Dict, Any, List, Callable
from dataclasses import dataclass


@dataclass
class LatencyResult:
    """Result of a latency test."""
    operation: str
    samples: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p999_ms: float
    p9999_ms: float


class LatencyBenchmark:
    """Measures operation latency at various percentiles."""

    def __init__(self):
        self.results: List[LatencyResult] = []

    def measure(
        self,
        operation: Callable,
        operation_name: str,
        samples: int = 10000,
        warmup: int = 100,
    ) -> LatencyResult:
        """
        Measure latency of an operation.

        Args:
            operation: Function to measure
            operation_name: Name for reporting
            samples: Number of samples to collect
            warmup: Warmup iterations
        """
        # Warmup
        for _ in range(warmup):
            operation()

        # Collect latency samples
        latencies_ns = []

        for _ in range(samples):
            start = time.perf_counter_ns()
            operation()
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        # Convert to milliseconds
        latencies_ms = [ns / 1_000_000 for ns in latencies_ns]

        # Sort for percentile calculations
        latencies_sorted = sorted(latencies_ms)

        def percentile(data: List[float], p: float) -> float:
            """Calculate percentile."""
            if not data:
                return 0
            k = (len(data) - 1) * (p / 100)
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (data[c] - data[f]) * (k - f)

        result = LatencyResult(
            operation=operation_name,
            samples=samples,
            min_ms=min(latencies_ms),
            max_ms=max(latencies_ms),
            mean_ms=statistics.mean(latencies_ms),
            median_ms=statistics.median(latencies_ms),
            p50_ms=percentile(latencies_sorted, 50),
            p90_ms=percentile(latencies_sorted, 90),
            p99_ms=percentile(latencies_sorted, 99),
            p999_ms=percentile(latencies_sorted, 99.9),
            p9999_ms=percentile(latencies_sorted, 99.99),
        )

        self.results.append(result)
        return result


class MockOperations:
    """Mock operations for latency testing."""

    @staticmethod
    def light_operation() -> bytes:
        """Light operation (hash)."""
        return hashlib.sha256(b"test").digest()

    @staticmethod
    def medium_operation() -> bytes:
        """Medium operation (multiple hashes)."""
        data = b"test"
        for _ in range(10):
            data = hashlib.sha256(data).digest()
        return data

    @staticmethod
    def heavy_operation() -> bytes:
        """Heavy operation (key derivation)."""
        return hashlib.pbkdf2_hmac('sha256', b'password', b'salt', 1000)

    @staticmethod
    def variable_operation() -> bytes:
        """Operation with variable latency."""
        iterations = secrets.randbelow(100) + 1
        data = b"test"
        for _ in range(iterations):
            data = hashlib.sha256(data).digest()
        return data


class TestP50Latency:
    """Tests for p50 (median) latency."""

    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark()

    @pytest.mark.stress
    def test_light_operation_p50(self, benchmark, latency_targets):
        """
        SPT-002: Light operation p50 < 1ms.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "light_operation",
            samples=10000,
        )

        assert result.p50_ms < latency_targets["p50"], \
            f"p50 latency {result.p50_ms:.3f}ms exceeds {latency_targets['p50']}ms"

    @pytest.mark.stress
    def test_medium_operation_p50(self, benchmark):
        """
        Medium operation p50 latency.
        """
        result = benchmark.measure(
            MockOperations.medium_operation,
            "medium_operation",
            samples=5000,
        )

        # Medium ops should complete in reasonable time
        assert result.p50_ms < 5, f"p50 latency {result.p50_ms:.3f}ms too high"


class TestP99Latency:
    """Tests for p99 latency."""

    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark()

    @pytest.mark.stress
    def test_p99_under_10ms(self, benchmark, latency_targets):
        """
        SPT-002: All operations p99 < 10ms.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "p99_test",
            samples=10000,
        )

        assert result.p99_ms < latency_targets["p99"], \
            f"p99 latency {result.p99_ms:.3f}ms exceeds {latency_targets['p99']}ms"

    @pytest.mark.stress
    def test_p99_consistency(self, benchmark):
        """
        P99 latency should be stable across runs.
        """
        p99_values = []

        for i in range(5):
            result = benchmark.measure(
                MockOperations.light_operation,
                f"consistency_{i}",
                samples=5000,
            )
            p99_values.append(result.p99_ms)

        # P99 should be relatively consistent
        stdev = statistics.stdev(p99_values)
        mean = statistics.mean(p99_values)
        cv = stdev / mean if mean > 0 else 0

        assert cv < 0.5, f"p99 latency too variable: CV={cv:.2f}"


class TestTailLatency:
    """Tests for tail latency (p99.9, p99.99)."""

    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark()

    @pytest.mark.stress
    def test_p999_latency(self, benchmark, latency_targets):
        """
        SPT-002: p99.9 < 50ms.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "p999_test",
            samples=10000,
        )

        assert result.p999_ms < latency_targets["p99.9"], \
            f"p99.9 latency {result.p999_ms:.3f}ms exceeds {latency_targets['p99.9']}ms"

    @pytest.mark.stress
    def test_p9999_latency(self, benchmark, latency_targets):
        """
        SPT-002: p99.99 < 100ms.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "p9999_test",
            samples=50000,  # Need more samples for p99.99
        )

        assert result.p9999_ms < latency_targets["p99.99"], \
            f"p99.99 latency {result.p9999_ms:.3f}ms exceeds {latency_targets['p99.99']}ms"

    @pytest.mark.stress
    def test_tail_latency_ratio(self, benchmark):
        """
        Tail latency should not be excessively worse than median.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "tail_ratio_test",
            samples=10000,
        )

        ratio = result.p99_ms / result.p50_ms if result.p50_ms > 0 else float('inf')

        # P99 shouldn't be more than 10x p50
        assert ratio < 10, f"p99/p50 ratio {ratio:.1f}x too high"


class TestLatencyDistribution:
    """Tests for latency distribution characteristics."""

    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark()

    @pytest.mark.stress
    def test_latency_not_bimodal(self, benchmark):
        """
        Latency distribution should not be bimodal.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "bimodal_test",
            samples=10000,
        )

        # For non-bimodal, mean should be close to median
        ratio = result.mean_ms / result.median_ms if result.median_ms > 0 else float('inf')

        # Mean should be within 3x of median
        assert 0.5 < ratio < 3, \
            f"Mean/median ratio {ratio:.2f} suggests bimodal distribution"

    @pytest.mark.stress
    def test_no_outlier_latency_spikes(self, benchmark):
        """
        No extreme outlier latency spikes.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "outlier_test",
            samples=10000,
        )

        # Max should not be more than 100x median
        ratio = result.max_ms / result.median_ms if result.median_ms > 0 else float('inf')

        assert ratio < 100, \
            f"Max/median ratio {ratio:.1f}x indicates outlier spikes"


class TestLatencyUnderLoad:
    """Tests for latency under concurrent load."""

    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark()

    @pytest.mark.stress
    def test_latency_with_background_load(self, benchmark):
        """
        Latency should not degrade significantly under load.
        """
        import threading
        import concurrent.futures

        # Measure baseline latency
        baseline = benchmark.measure(
            MockOperations.light_operation,
            "baseline",
            samples=1000,
        )

        # Create background load
        stop_flag = threading.Event()

        def background_load():
            while not stop_flag.is_set():
                MockOperations.medium_operation()

        # Start background workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(background_load) for _ in range(4)]

            # Measure latency under load
            under_load = benchmark.measure(
                MockOperations.light_operation,
                "under_load",
                samples=1000,
            )

            # Stop background workers
            stop_flag.set()

        # Latency should not increase more than 5x
        ratio = under_load.p99_ms / baseline.p99_ms if baseline.p99_ms > 0 else float('inf')

        print(f"\nBaseline p99: {baseline.p99_ms:.3f}ms")
        print(f"Under load p99: {under_load.p99_ms:.3f}ms")
        print(f"Ratio: {ratio:.2f}x")

        assert ratio < 5, f"Latency degradation {ratio:.2f}x under load"


class TestLatencyPredictability:
    """Tests for latency predictability."""

    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark()

    @pytest.mark.stress
    def test_deterministic_operation_latency(self, benchmark):
        """
        Deterministic operations should have consistent latency.
        """
        result = benchmark.measure(
            MockOperations.light_operation,
            "deterministic_test",
            samples=10000,
        )

        # Standard deviation should be small relative to mean
        # Note: We calculate this from the result fields
        # p90 - p50 gives us an idea of spread
        spread = result.p90_ms - result.p50_ms

        # Spread should be small for deterministic operations
        assert spread < result.median_ms, \
            "Latency spread too large for deterministic operation"

    @pytest.mark.stress
    def test_variable_operation_latency(self, benchmark):
        """
        Variable operations have higher but bounded latency variance.
        """
        result = benchmark.measure(
            MockOperations.variable_operation,
            "variable_test",
            samples=5000,
        )

        # Variable ops should still have bounded tail latency
        assert result.p99_ms < 100, \
            "Variable operation p99 exceeds 100ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
