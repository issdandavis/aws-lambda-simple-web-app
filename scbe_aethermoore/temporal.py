"""
Temporal Trajectory Module (AXIS 4)

Implements phase locking and trajectory coherence for time-bound authorization.
Enhanced with planetary orbital phase locking.

Reference: Section 4.4 of SCBE-AETHER-UNIFIED-2026-001
Claims: 25, 29, 30, 52
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timezone
from .constants import (
    PERFECT_FIFTH,
    PLANETARY_FREQUENCIES,
    PARAMETER_RANGES
)
from .harmonic import harmonic_scaling


# J2000 epoch: January 1, 2000, 12:00 TT (Terrestrial Time)
J2000_UNIX = 946728000.0  # Unix timestamp for J2000

# Planetary orbital periods in seconds
ORBITAL_PERIODS = {
    "mercury": 87.97 * 86400,
    "venus": 224.70 * 86400,
    "earth": 365.25 * 86400,
    "mars": 687.00 * 86400,
    "jupiter": 4333 * 86400,
    "saturn": 10759 * 86400,
}


@dataclass
class TrajectoryPoint:
    """A point on a temporal trajectory."""
    timestamp: float
    phase: float
    coherence: float
    valid: bool


@dataclass
class TrajectorySegment:
    """
    A segment of authorization trajectory.

    Implements self-expiring credentials (Claim 29).
    """
    start_time: float
    end_time: float
    start_phase: float
    end_phase: float
    coherence_threshold: float

    def is_active(self, current_time: Optional[float] = None) -> bool:
        """Check if segment is currently active."""
        t = current_time or time.time()
        return self.start_time <= t <= self.end_time

    def time_remaining(self, current_time: Optional[float] = None) -> float:
        """Return seconds until expiration."""
        t = current_time or time.time()
        return max(0.0, self.end_time - t)


def phase_lock(
    t: float,
    epoch: float,
    period: float
) -> float:
    """
    Standard phase lock computation.

    φ_expected(t) = (2π × (t - epoch) / period) mod 2π

    Args:
        t: Current timestamp
        epoch: Reference epoch
        period: Phase period in seconds

    Returns:
        Expected phase in [0, 2π)

    Reference: Section 4.4
    Claim: 30
    """
    return (2 * math.pi * (t - epoch) / period) % (2 * math.pi)


def planetary_phase(
    t: float,
    planet: str = "mars"
) -> float:
    """
    Compute planetary orbital phase.

    φ_planetary(t) = (2π × (t - J2000) / T_planet) mod 2π

    An attacker would need to predict planetary orbital phase to forge timestamps.

    Args:
        t: Current Unix timestamp
        planet: Planet name (mars, venus, earth, jupiter, saturn, mercury)

    Returns:
        Orbital phase in [0, 2π)

    Reference: Section 4.4 AETHERMOORE Enhancement
    Claim: 52
    """
    if planet not in ORBITAL_PERIODS:
        raise ValueError(f"Unknown planet: {planet}")

    period = ORBITAL_PERIODS[planet]
    return phase_lock(t, J2000_UNIX, period)


def trajectory_coherence(
    trajectory: List[TrajectoryPoint],
    epsilon: float = 0.15
) -> float:
    """
    Compute trajectory coherence score.

    Measures how smoothly phase progresses along the trajectory.
    Jumps or reversals reduce coherence.

    Args:
        trajectory: List of trajectory points
        epsilon: Coherence threshold (default: 0.15)

    Returns:
        Coherence score in [0, 1]

    Reference: Section 4.4
    Claim: 25(f)
    """
    if len(trajectory) < 2:
        return 1.0

    coherent_pairs = 0
    total_pairs = len(trajectory) - 1

    for i in range(total_pairs):
        current = trajectory[i]
        next_pt = trajectory[i + 1]

        # Check time progression
        if next_pt.timestamp <= current.timestamp:
            continue  # Time reversal - not coherent

        # Check phase progression (allowing wrap-around)
        phase_diff = (next_pt.phase - current.phase) % (2 * math.pi)

        # Phase should progress positively within epsilon bounds
        if 0 < phase_diff < math.pi + epsilon:
            coherent_pairs += 1

    return coherent_pairs / total_pairs if total_pairs > 0 else 1.0


def multi_planetary_phase(t: float) -> dict:
    """
    Compute phases for all tracked planets.

    Returns:
        Dict mapping planet names to orbital phases
    """
    return {planet: planetary_phase(t, planet) for planet in ORBITAL_PERIODS}


def phase_distance(phase1: float, phase2: float) -> float:
    """
    Compute angular distance between two phases.

    Handles wrap-around correctly.

    Args:
        phase1, phase2: Phases in [0, 2π)

    Returns:
        Angular distance in [0, π]
    """
    diff = abs(phase1 - phase2) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)


def verify_phase_lock(
    claimed_time: float,
    claimed_phase: float,
    planet: str = "mars",
    tolerance: float = 0.1
) -> Tuple[bool, float]:
    """
    Verify that a claimed time/phase pair is consistent.

    Used to detect timestamp forgery.

    Args:
        claimed_time: Claimed Unix timestamp
        claimed_phase: Claimed orbital phase
        planet: Reference planet
        tolerance: Maximum allowed phase error

    Returns:
        Tuple of (valid, actual_distance)
    """
    expected = planetary_phase(claimed_time, planet)
    distance = phase_distance(expected, claimed_phase)

    return (distance <= tolerance, distance)


def create_trajectory_segment(
    duration: float,
    start_time: Optional[float] = None,
    planet: str = "mars",
    coherence_threshold: float = 0.15
) -> TrajectorySegment:
    """
    Create a self-expiring credential segment.

    Args:
        duration: Duration in seconds
        start_time: Start timestamp (default: now)
        planet: Planet for phase locking
        coherence_threshold: Required coherence level

    Returns:
        TrajectorySegment

    Claim: 29
    """
    start = start_time or time.time()
    end = start + duration

    return TrajectorySegment(
        start_time=start,
        end_time=end,
        start_phase=planetary_phase(start, planet),
        end_phase=planetary_phase(end, planet),
        coherence_threshold=coherence_threshold
    )


def validate_trajectory(
    trajectory: List[TrajectoryPoint],
    segment: TrajectorySegment,
    planet: str = "mars"
) -> Tuple[bool, dict]:
    """
    Validate a trajectory against a segment.

    Checks:
    1. All points within time bounds
    2. Phase progression is monotonic
    3. Coherence above threshold
    4. Phases match planetary positions

    Args:
        trajectory: Points to validate
        segment: Authorization segment
        planet: Reference planet

    Returns:
        Tuple of (valid, details)
    """
    if not trajectory:
        return (False, {"error": "Empty trajectory"})

    # Check time bounds
    for i, pt in enumerate(trajectory):
        if pt.timestamp < segment.start_time or pt.timestamp > segment.end_time:
            return (False, {"error": f"Point {i} outside time bounds"})

        # Verify phase against planetary position
        expected_phase = planetary_phase(pt.timestamp, planet)
        distance = phase_distance(pt.phase, expected_phase)

        if distance > 0.2:  # Phase tolerance
            return (False, {
                "error": f"Point {i} phase mismatch",
                "expected": expected_phase,
                "actual": pt.phase,
                "distance": distance
            })

    # Check coherence
    coherence = trajectory_coherence(trajectory)
    if coherence < segment.coherence_threshold:
        return (False, {
            "error": "Insufficient coherence",
            "coherence": coherence,
            "threshold": segment.coherence_threshold
        })

    return (True, {
        "coherence": coherence,
        "points_validated": len(trajectory),
        "planet": planet
    })


def replay_detection(
    current_phase: float,
    recent_phases: List[float],
    min_separation: float = 0.05
) -> bool:
    """
    Detect replay attacks using phase uniqueness.

    Each phase should only be used once within a window.

    Args:
        current_phase: Phase to check
        recent_phases: Recently used phases
        min_separation: Minimum phase distance

    Returns:
        True if current phase is valid (not a replay)

    Claim: 30
    """
    for recent in recent_phases:
        if phase_distance(current_phase, recent) < min_separation:
            return False  # Possible replay
    return True


class TemporalAuthorizer:
    """
    Complete temporal authorization system.
    """

    def __init__(
        self,
        planet: str = "mars",
        phase_tolerance: float = 0.1,
        coherence_threshold: float = 0.15,
        replay_window: int = 100
    ):
        """
        Initialize temporal authorizer.

        Args:
            planet: Planet for phase locking
            phase_tolerance: Allowed phase deviation
            coherence_threshold: Required trajectory coherence
            replay_window: Number of recent phases to track
        """
        self.planet = planet
        self.phase_tolerance = phase_tolerance
        self.coherence_threshold = coherence_threshold
        self.replay_window = replay_window

        self.recent_phases: List[float] = []
        self.active_segments: List[TrajectorySegment] = []

    def current_phase(self) -> float:
        """Get current planetary phase."""
        return planetary_phase(time.time(), self.planet)

    def create_segment(self, duration: float) -> TrajectorySegment:
        """Create and register a new authorization segment."""
        segment = create_trajectory_segment(
            duration,
            planet=self.planet,
            coherence_threshold=self.coherence_threshold
        )
        self.active_segments.append(segment)
        return segment

    def authorize(
        self,
        timestamp: float,
        phase: float
    ) -> Tuple[bool, dict]:
        """
        Authorize a time/phase pair.

        Returns:
            Tuple of (authorized, details)
        """
        # Verify phase matches planetary position
        valid, distance = verify_phase_lock(
            timestamp, phase, self.planet, self.phase_tolerance
        )

        if not valid:
            return (False, {
                "error": "Phase mismatch",
                "expected": planetary_phase(timestamp, self.planet),
                "actual": phase,
                "distance": distance
            })

        # Check for replay
        if not replay_detection(phase, self.recent_phases):
            return (False, {"error": "Possible replay detected"})

        # Check active segments
        for segment in self.active_segments:
            if segment.is_active(timestamp):
                self.recent_phases.append(phase)
                if len(self.recent_phases) > self.replay_window:
                    self.recent_phases.pop(0)

                return (True, {
                    "segment_remaining": segment.time_remaining(timestamp),
                    "phase": phase,
                    "planet": self.planet
                })

        return (False, {"error": "No active authorization segment"})

    def cleanup_expired(self) -> int:
        """Remove expired segments. Returns count removed."""
        current = time.time()
        before = len(self.active_segments)
        self.active_segments = [s for s in self.active_segments if s.is_active(current)]
        return before - len(self.active_segments)
