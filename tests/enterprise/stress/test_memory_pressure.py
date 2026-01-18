"""
SPT-004: Memory Pressure Tests

Tests for memory usage and leak detection.
Target: No memory leaks after 72h operation.
"""

import pytest
import gc
import time
import hashlib
import secrets
import weakref
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage."""
    timestamp: float
    allocated_mb: float
    peak_mb: float
    gc_objects: int


class MemoryMonitor:
    """Monitors memory usage over time."""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()

        current, peak = tracemalloc.get_traced_memory()
        gc.collect()

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=current / (1024 * 1024),
            peak_mb=peak / (1024 * 1024),
            gc_objects=len(gc.get_objects()),
        )

        self.snapshots.append(snapshot)
        return snapshot

    def get_growth_rate(self) -> float:
        """Calculate memory growth rate (MB/hour)."""
        if len(self.snapshots) < 2:
            return 0

        first = self.snapshots[0]
        last = self.snapshots[-1]

        time_diff_hours = (last.timestamp - first.timestamp) / 3600
        memory_diff_mb = last.allocated_mb - first.allocated_mb

        if time_diff_hours <= 0:
            return 0

        return memory_diff_mb / time_diff_hours

    def detect_leak(self, threshold_mb_per_hour: float = 10) -> bool:
        """Detect if there's a memory leak."""
        growth_rate = self.get_growth_rate()
        return growth_rate > threshold_mb_per_hour


class LeakDetector:
    """Detects memory leaks in specific code paths."""

    def __init__(self):
        self.weak_refs: List[weakref.ref] = []

    def track_object(self, obj: Any):
        """Track an object for leak detection."""
        try:
            self.weak_refs.append(weakref.ref(obj))
        except TypeError:
            pass  # Some objects can't be weakly referenced

    def get_leaked_count(self) -> int:
        """Count objects that should have been garbage collected."""
        gc.collect()
        gc.collect()  # Run twice to be sure

        live_count = sum(1 for ref in self.weak_refs if ref() is not None)
        return live_count


class MockSCBECache:
    """Mock cache for testing memory behavior."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, bytes] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[bytes]:
        """Get item from cache."""
        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: bytes):
        """Put item in cache with LRU eviction."""
        if key not in self.cache and len(self.cache) >= self.max_size:
            # Evict oldest
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class TestMemoryLeaks:
    """Tests for memory leak detection."""

    @pytest.fixture
    def monitor(self):
        return MemoryMonitor()

    @pytest.fixture
    def leak_detector(self):
        return LeakDetector()

    @pytest.mark.stress
    def test_no_leak_in_operations(self, monitor):
        """
        SPT-004: No memory leak in normal operations.
        """
        # Take initial snapshot
        initial = monitor.take_snapshot()

        # Perform many operations
        for _ in range(10000):
            data = secrets.token_bytes(1024)
            hashlib.sha256(data).digest()

        gc.collect()

        # Take final snapshot
        final = monitor.take_snapshot()

        # Memory should not grow significantly
        growth = final.allocated_mb - initial.allocated_mb
        assert growth < 10, f"Memory grew by {growth:.2f}MB"

    @pytest.mark.stress
    def test_objects_garbage_collected(self, leak_detector):
        """
        Objects should be garbage collected when no longer referenced.
        """
        # Create and track objects
        for _ in range(1000):
            obj = {"data": secrets.token_bytes(1024)}
            leak_detector.track_object(obj)

        # Objects go out of scope here
        gc.collect()

        # Should have very few (if any) leaked objects
        leaked = leak_detector.get_leaked_count()
        assert leaked < 10, f"Leaked {leaked} objects"

    @pytest.mark.stress
    def test_cache_bounded_memory(self):
        """
        Cache should have bounded memory usage.
        """
        cache = MockSCBECache(max_size=100)

        # Insert more items than max size
        for i in range(1000):
            key = f"key_{i}"
            value = secrets.token_bytes(1024)
            cache.put(key, value)

        # Cache size should be bounded
        assert len(cache.cache) <= 100

    @pytest.mark.stress
    def test_cache_lru_eviction(self):
        """
        LRU eviction should work correctly.
        """
        cache = MockSCBECache(max_size=10)

        # Fill cache
        for i in range(10):
            cache.put(f"key_{i}", b"value")

        # Access first key to make it recently used
        cache.get("key_0")

        # Add new item, should evict key_1 (oldest unaccessed)
        cache.put("key_new", b"new_value")

        assert "key_0" in cache.cache  # Recently accessed
        assert "key_1" not in cache.cache  # Evicted
        assert "key_new" in cache.cache  # New item


class TestMemoryUnderLoad:
    """Tests for memory behavior under load."""

    @pytest.fixture
    def monitor(self):
        return MemoryMonitor()

    @pytest.mark.stress
    def test_memory_stable_under_load(self, monitor):
        """
        Memory should stabilize under sustained load.
        """
        # Take snapshots during load
        for cycle in range(10):
            # Simulate load
            for _ in range(1000):
                data = secrets.token_bytes(4096)
                hashlib.sha256(data).digest()

            gc.collect()
            monitor.take_snapshot()

        # Memory should not grow continuously
        growth_rate = monitor.get_growth_rate()
        assert not monitor.detect_leak(threshold_mb_per_hour=100), \
            f"Memory leak detected: {growth_rate:.2f} MB/hour growth"

    @pytest.mark.stress
    def test_memory_recovery_after_spike(self, monitor):
        """
        Memory should recover after usage spike.
        """
        initial = monitor.take_snapshot()

        # Create memory spike
        large_data = [secrets.token_bytes(1024 * 1024) for _ in range(10)]  # 10MB

        spike = monitor.take_snapshot()

        # Release memory
        del large_data
        gc.collect()
        gc.collect()

        recovered = monitor.take_snapshot()

        # Memory should recover close to initial
        spike_increase = spike.allocated_mb - initial.allocated_mb
        final_increase = recovered.allocated_mb - initial.allocated_mb

        assert final_increase < spike_increase * 0.2, \
            "Memory not recovered after spike"


class TestResourceLimits:
    """Tests for resource limit enforcement."""

    @pytest.mark.stress
    def test_per_request_memory_limit(self):
        """
        Per-request memory should be bounded.
        """
        import tracemalloc
        tracemalloc.start()

        initial, _ = tracemalloc.get_traced_memory()

        # Simulate a single request
        request_data = secrets.token_bytes(1024)
        result = hashlib.sha256(request_data).digest()

        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory used should be small
        used_mb = (current - initial) / (1024 * 1024)
        assert used_mb < 10, f"Request used {used_mb:.2f}MB, exceeds 10MB limit"

    @pytest.mark.stress
    def test_gc_overhead_acceptable(self):
        """
        Garbage collection overhead should be acceptable.
        """
        # Measure GC time
        gc_times = []

        for _ in range(10):
            # Create garbage
            garbage = [secrets.token_bytes(1024) for _ in range(1000)]
            del garbage

            start = time.perf_counter()
            gc.collect()
            gc_times.append(time.perf_counter() - start)

        avg_gc_time = sum(gc_times) / len(gc_times)

        # GC should complete quickly
        assert avg_gc_time < 0.1, f"GC takes {avg_gc_time:.3f}s, too slow"


class TestLongRunningStability:
    """Tests for long-running stability (simulated)."""

    @pytest.fixture
    def monitor(self):
        return MemoryMonitor()

    @pytest.mark.stress
    @pytest.mark.slow
    def test_simulated_72h_operation(self, monitor):
        """
        Simulate 72 hours of operation (compressed).
        """
        # Simulate 72 hours with 1000 iterations representing time periods
        iterations = 1000

        for i in range(iterations):
            # Simulate periodic operations
            for _ in range(100):
                data = secrets.token_bytes(1024)
                hashlib.sha256(data).digest()

            # Periodic cleanup
            if i % 10 == 0:
                gc.collect()
                monitor.take_snapshot()

        # Check for memory leak
        assert not monitor.detect_leak(threshold_mb_per_hour=10), \
            "Memory leak detected in simulated long-running test"

        print(f"\nSimulated 72h test:")
        print(f"  Snapshots: {len(monitor.snapshots)}")
        print(f"  Growth rate: {monitor.get_growth_rate():.2f} MB/hour")

    @pytest.mark.stress
    def test_object_count_stable(self, monitor):
        """
        Number of Python objects should remain stable.
        """
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Many operations
        for _ in range(1000):
            data = secrets.token_bytes(1024)
            result = hashlib.sha256(data).digest()
            del data, result

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow significantly
        growth = final_objects - initial_objects
        assert growth < 1000, f"Object count grew by {growth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
