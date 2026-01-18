#!/usr/bin/env python3
"""
Spiralverse Protocol - Enterprise Demo
=======================================

Production-ready demonstration with:
- Color-coded output
- Real-time metrics and animated progress
- Waiting Room for quarantined requests
- Configuration support
- Logging and audit trail
- JSON report generation
- CLI interface

Usage:
    python spiralverse_demo.py                    # Interactive demo
    python spiralverse_demo.py --json             # Output JSON report
    python spiralverse_demo.py --agent "my-agent" # Custom agent ID
    python spiralverse_demo.py --coord 0.5,0.5,0.5,0.5,0.5,0.5  # Custom coords
    python spiralverse_demo.py --waiting-room     # Show waiting room demo
"""

import sys
import os
import time
import json
import argparse
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ═══════════════════════════════════════════════════════════════
# ANSI Colors (works on Windows 10+, macOS, Linux)
# ═══════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    # Enable ANSI on Windows
    if sys.platform == 'win32':
        os.system('')  # Enables ANSI escape sequences

    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # Background
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, '')


# Check if output is a TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_banner():
    """Print the enterprise banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ____  _____ ___ ____      _    _  __     _______ ____  ____  ║
    ║    / ___||  _  |_ _|  _ \    / \  | | \ \   / / ____|  _ \/ ___| ║
    ║    \___ \| |_) || || |_) |  / _ \ | |  \ \ / /|  _| | |_) \___ \ ║
    ║     ___) |  __/ | ||  _ <  / ___ \| |___\ V / | |___|  _ < ___) |║
    ║    |____/|_|   |___|_| \_\/_/   \_\_____|\_/  |_____|_| \_\____/ ║
    ║                                                                  ║
    ║               QUANTUM-RESISTANT AI GOVERNANCE                    ║
    ║                    Enterprise Edition v1.0                       ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  {Colors.GREEN}SpiralSeal{Colors.CYAN}  {Colors.YELLOW}GeoSeal{Colors.CYAN}  {Colors.MAGENTA}Governance{Colors.CYAN}  {Colors.RED}Post-Quantum{Colors.CYAN}  {Colors.WHITE}Waiting Room{Colors.CYAN}  ║
    ╚══════════════════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


def print_phase(phase_num: int, title: str):
    """Print a phase header."""
    colors = [Colors.BLUE, Colors.MAGENTA, Colors.CYAN, Colors.GREEN]
    color = colors[(phase_num - 1) % len(colors)]
    print(f"\n{color}{Colors.BOLD}{'━' * 70}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{'━' * 70}{Colors.RESET}\n")


def print_status(label: str, value: str, status: str = "info"):
    """Print a status line with color."""
    color_map = {
        "success": Colors.GREEN,
        "error": Colors.RED,
        "warning": Colors.YELLOW,
        "info": Colors.WHITE,
    }
    color = color_map.get(status, Colors.WHITE)
    icon_map = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "•",
    }
    icon = icon_map.get(status, "•")
    print(f"  {color}{icon} {label}: {Colors.BOLD}{value}{Colors.RESET}")


def print_metric(label: str, value: float, unit: str = "", threshold_low: float = None, threshold_high: float = None):
    """Print a metric with color-coded thresholds."""
    if threshold_high is not None and value > threshold_high:
        color = Colors.RED
    elif threshold_low is not None and value < threshold_low:
        color = Colors.GREEN
    else:
        color = Colors.YELLOW

    bar_width = 30
    if threshold_high:
        fill = min(int((value / threshold_high) * bar_width), bar_width)
    else:
        fill = bar_width // 2

    bar = f"[{'█' * fill}{'░' * (bar_width - fill)}]"

    print(f"  {Colors.DIM}{label}:{Colors.RESET} {color}{value:.4f}{unit}{Colors.RESET} {Colors.DIM}{bar}{Colors.RESET}")


def print_decision(decision: str, reason: str):
    """Print the final decision with dramatic effect."""
    if decision == "ALLOW":
        color = Colors.GREEN
        bg = Colors.BG_GREEN
        icon = "✓ ✓ ✓"
    elif decision == "DENY":
        color = Colors.RED
        bg = Colors.BG_RED
        icon = "✗ ✗ ✗"
    else:
        color = Colors.YELLOW
        bg = Colors.BG_YELLOW
        icon = "⚠ ⚠ ⚠"

    print(f"""
  {color}{'═' * 60}
  {bg}{Colors.BOLD}   {icon}  DECISION: {decision}  {icon}   {Colors.RESET}
  {color}{'═' * 60}{Colors.RESET}
  {Colors.DIM}Reason: {reason}{Colors.RESET}
""")


def animate_progress(label: str, duration: float = 1.0, steps: int = 20):
    """Animated progress bar."""
    sys.stdout.write(f"  {Colors.DIM}{label}: [{Colors.RESET}")
    sys.stdout.flush()

    for i in range(steps):
        time.sleep(duration / steps)
        sys.stdout.write(f"{Colors.CYAN}█{Colors.RESET}")
        sys.stdout.flush()

    sys.stdout.write(f"{Colors.DIM}] {Colors.GREEN}Done{Colors.RESET}\n")
    sys.stdout.flush()


def animate_scan(label: str, items: List[str], delay: float = 0.15):
    """Animated scan through items."""
    print(f"  {Colors.DIM}{label}:{Colors.RESET}")
    for item in items:
        time.sleep(delay)
        print(f"    {Colors.CYAN}→{Colors.RESET} {item}")


# ═══════════════════════════════════════════════════════════════
# Waiting Room System
# ═══════════════════════════════════════════════════════════════

@dataclass
class WaitingRoomEntry:
    """Entry in the waiting room."""
    request_id: str
    agent_id: str
    topic: str
    coord_6d: Tuple[float, ...]
    risk_score: float
    entry_time: float
    status: str = "PENDING"  # PENDING, REVIEWING, ESCALATED, RELEASED, DENIED
    review_notes: str = ""


class WaitingRoom:
    """
    Waiting Room for quarantined requests.

    Requests that fall in the "gray zone" (risk between 0.3-0.7) are held
    for additional verification before final decision.
    """

    def __init__(self):
        self.entries: Dict[str, WaitingRoomEntry] = {}
        self.review_queue: List[str] = []
        self.processed: List[WaitingRoomEntry] = []

    def admit(self, agent_id: str, topic: str, coord_6d: Tuple[float, ...],
              risk_score: float) -> WaitingRoomEntry:
        """Admit a request to the waiting room."""
        request_id = hashlib.sha256(
            f"{agent_id}:{topic}:{time.time()}".encode()
        ).hexdigest()[:12]

        entry = WaitingRoomEntry(
            request_id=request_id,
            agent_id=agent_id,
            topic=topic,
            coord_6d=coord_6d,
            risk_score=risk_score,
            entry_time=time.time(),
        )

        self.entries[request_id] = entry
        self.review_queue.append(request_id)
        return entry

    def review(self, request_id: str, additional_checks: Dict[str, bool]) -> str:
        """
        Review a waiting room entry with additional verification.

        Returns final decision: ALLOW, DENY, or ESCALATE
        """
        if request_id not in self.entries:
            return "INVALID"

        entry = self.entries[request_id]
        entry.status = "REVIEWING"

        # Additional verification logic
        checks_passed = sum(additional_checks.values())
        total_checks = len(additional_checks)

        if checks_passed == total_checks:
            entry.status = "RELEASED"
            entry.review_notes = "All additional checks passed"
            return "ALLOW"
        elif checks_passed >= total_checks * 0.7:
            entry.status = "RELEASED"
            entry.review_notes = f"{checks_passed}/{total_checks} checks passed"
            return "ALLOW"
        elif checks_passed < total_checks * 0.3:
            entry.status = "DENIED"
            entry.review_notes = f"Only {checks_passed}/{total_checks} checks passed"
            return "DENY"
        else:
            entry.status = "ESCALATED"
            entry.review_notes = "Requires human review"
            return "ESCALATE"

    def get_queue_status(self) -> Dict:
        """Get current waiting room status."""
        status_counts = {
            "PENDING": 0,
            "REVIEWING": 0,
            "ESCALATED": 0,
            "RELEASED": 0,
            "DENIED": 0,
        }

        for entry in self.entries.values():
            status_counts[entry.status] += 1

        return {
            "total": len(self.entries),
            "queue_length": len(self.review_queue),
            "by_status": status_counts,
        }

    def display(self, verbose: bool = True):
        """Display waiting room status with visual effects."""
        if not verbose:
            return

        print(f"\n{Colors.YELLOW}{Colors.BOLD}{'═' * 70}")
        print(f"  WAITING ROOM - QUARANTINE ZONE")
        print(f"{'═' * 70}{Colors.RESET}\n")

        status = self.get_queue_status()

        # Visual representation of queue
        print(f"  {Colors.BOLD}Queue Status:{Colors.RESET}")
        print(f"  ┌{'─' * 66}┐")
        print(f"  │  {Colors.YELLOW}⏳ Pending:{Colors.RESET}   {status['by_status']['PENDING']:>3}   "
              f"{Colors.CYAN}🔍 Reviewing:{Colors.RESET} {status['by_status']['REVIEWING']:>3}   "
              f"{Colors.MAGENTA}📤 Escalated:{Colors.RESET} {status['by_status']['ESCALATED']:>3}  │")
        print(f"  │  {Colors.GREEN}✓ Released:{Colors.RESET}  {status['by_status']['RELEASED']:>3}   "
              f"{Colors.RED}✗ Denied:{Colors.RESET}    {status['by_status']['DENIED']:>3}                      │")
        print(f"  └{'─' * 66}┘")

        # Show entries
        if self.entries:
            print(f"\n  {Colors.BOLD}Entries:{Colors.RESET}")
            for req_id, entry in list(self.entries.items())[:5]:  # Show max 5
                status_icon = {
                    "PENDING": f"{Colors.YELLOW}⏳",
                    "REVIEWING": f"{Colors.CYAN}🔍",
                    "ESCALATED": f"{Colors.MAGENTA}📤",
                    "RELEASED": f"{Colors.GREEN}✓",
                    "DENIED": f"{Colors.RED}✗",
                }.get(entry.status, "•")

                wait_time = time.time() - entry.entry_time
                print(f"  {status_icon}{Colors.RESET} [{req_id}] {entry.agent_id} "
                      f"- Risk: {entry.risk_score:.2f} - Wait: {wait_time:.1f}s")


def run_waiting_room_demo(engine: 'SpiralverseEngine', verbose: bool = True):
    """Demonstrate the waiting room functionality."""

    if verbose:
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'▓' * 70}")
        print(f"  WAITING ROOM DEMONSTRATION")
        print(f"{'▓' * 70}{Colors.RESET}")

        print(f"\n  {Colors.DIM}The Waiting Room holds requests in the 'gray zone' (risk 0.3-0.7)")
        print(f"  for additional verification before making a final decision.{Colors.RESET}\n")

    waiting_room = WaitingRoom()

    # Simulate multiple requests entering the waiting room
    test_cases = [
        ("agent-beta-3", "data-export", (0.6, 0.6, 0.4, 0.4, 0.5, 0.5), 0.45),
        ("agent-gamma-9", "config-change", (0.7, 0.5, 0.3, 0.6, 0.4, 0.5), 0.52),
        ("agent-delta-2", "user-query", (0.55, 0.55, 0.5, 0.5, 0.5, 0.5), 0.38),
        ("agent-epsilon-7", "api-access", (0.65, 0.7, 0.35, 0.4, 0.6, 0.5), 0.61),
    ]

    if verbose:
        print_phase(1, "ADMITTING REQUESTS TO WAITING ROOM")

    admitted = []
    for agent_id, topic, coord, risk in test_cases:
        entry = waiting_room.admit(agent_id, topic, coord, risk)
        admitted.append(entry)
        if verbose:
            time.sleep(0.3)
            print(f"  {Colors.YELLOW}⏳{Colors.RESET} Admitted: {agent_id} (Risk: {risk:.2f}) → ID: {entry.request_id}")

    if verbose:
        waiting_room.display()

    # Process through waiting room
    if verbose:
        print_phase(2, "PROCESSING WAITING ROOM QUEUE")
        animate_progress("Initializing verification systems", 0.8)

    final_decisions = []
    for entry in admitted:
        if verbose:
            print(f"\n  {Colors.CYAN}▶ Processing {entry.request_id} ({entry.agent_id}){Colors.RESET}")
            time.sleep(0.2)

        # Simulate additional verification checks
        additional_checks = {
            "behavioral_analysis": entry.risk_score < 0.55,
            "pattern_matching": entry.risk_score < 0.5 or entry.coord_6d[0] < 0.65,
            "temporal_consistency": True,  # Usually passes
            "context_validation": entry.topic not in ["config-change"],
        }

        if verbose:
            animate_scan("Running checks", [
                f"Behavioral Analysis: {'✓' if additional_checks['behavioral_analysis'] else '✗'}",
                f"Pattern Matching: {'✓' if additional_checks['pattern_matching'] else '✗'}",
                f"Temporal Consistency: {'✓' if additional_checks['temporal_consistency'] else '✗'}",
                f"Context Validation: {'✓' if additional_checks['context_validation'] else '✗'}",
            ], delay=0.2)

        decision = waiting_room.review(entry.request_id, additional_checks)
        final_decisions.append((entry, decision))

        if verbose:
            decision_color = {
                "ALLOW": Colors.GREEN,
                "DENY": Colors.RED,
                "ESCALATE": Colors.MAGENTA,
            }.get(decision, Colors.YELLOW)
            print(f"    {Colors.BOLD}Decision: {decision_color}{decision}{Colors.RESET}")

    if verbose:
        print_phase(3, "FINAL WAITING ROOM STATUS")
        waiting_room.display()

        # Summary
        print(f"\n  {Colors.BOLD}Processing Complete:{Colors.RESET}")
        allow_count = sum(1 for _, d in final_decisions if d == "ALLOW")
        deny_count = sum(1 for _, d in final_decisions if d == "DENY")
        escalate_count = sum(1 for _, d in final_decisions if d == "ESCALATE")

        print(f"  {Colors.GREEN}✓ Released:{Colors.RESET} {allow_count}  "
              f"{Colors.RED}✗ Denied:{Colors.RESET} {deny_count}  "
              f"{Colors.MAGENTA}📤 Escalated:{Colors.RESET} {escalate_count}")

    return waiting_room, final_decisions


# ═══════════════════════════════════════════════════════════════
# Core Protocol Imports
# ═══════════════════════════════════════════════════════════════

try:
    from symphonic_cipher.scbe_aethermoore.spiral_seal import SpiralSealSS1
    from symphonic_cipher.scbe_aethermoore.geoseal import (
        GeoSealEngine, GeoSealState, SphericalCoord, HypercubeCoord,
        compute_time_dilation
    )
    from symphonic_cipher.scbe_aethermoore.governance import (
        harmonic_scaling, GovernanceDecision
    )
    from symphonic_cipher.scbe_aethermoore.quantum import PQCryptoSystem
    from symphonic_cipher.qasi_core import realm_distance
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class OperationMetrics:
    """Timing and performance metrics."""
    operation: str
    start_time: float
    end_time: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class GovernanceMetrics:
    """Governance check results."""
    d_star: float
    harmonic_scale: float
    risk_base: float
    risk_prime: float
    geoseal_authorized: bool
    geoseal_intersection: str
    pq_valid: bool
    decision: str
    reason: str


@dataclass
class DemoResult:
    """Complete demo execution result."""
    timestamp: str
    agent_id: str
    topic: str
    coord_6d: Tuple[float, ...]
    payload_size: int
    sealed_size: int

    # Metrics
    seal_time_ms: float
    governance_time_ms: float
    unseal_time_ms: float
    total_time_ms: float

    # Governance
    governance: GovernanceMetrics

    # Outcome
    decision: str
    payload_recovered: bool
    error: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result['coord_6d'] = list(self.coord_6d)
        return result


# ═══════════════════════════════════════════════════════════════
# Protocol Engine
# ═══════════════════════════════════════════════════════════════

class SpiralverseEngine:
    """Enterprise protocol engine."""

    def __init__(self, master_secret: bytes = None):
        self.master_secret = master_secret or b"spiralverse-enterprise-key!!"[:32].ljust(32, b'\x00')
        self.realm_centers = [np.array([0.5] * 6)]
        self.metrics: List[OperationMetrics] = []

    def _track(self, operation: str):
        """Context manager for tracking operations."""
        return OperationTracker(self, operation)

    def seal_memory(self, payload: bytes, agent_id: str, topic: str) -> Tuple[str, float]:
        """Seal a memory payload."""
        start = time.perf_counter()
        ss = SpiralSealSS1(self.master_secret)
        aad = f"agent={agent_id};topic={topic}"
        blob = ss.seal(payload, aad=aad)
        duration_ms = (time.perf_counter() - start) * 1000
        return blob, duration_ms

    def compute_governance(self, coord_6d: Tuple[float, ...], agent_id: str) -> Tuple[GovernanceMetrics, float]:
        """Compute governance decision."""
        start = time.perf_counter()

        # Distance to realm
        u = np.array(coord_6d)
        centers = np.array(self.realm_centers)
        d_star = realm_distance(u, centers)

        # Harmonic scaling
        R = 1.5
        try:
            H = harmonic_scaling(d_star, R)
        except:
            H = 11.0

        # Risk calculation
        risk_base = min(0.1 + d_star * 0.2, 1.0)
        risk_prime = risk_base * H

        # GeoSeal check
        geoseal = GeoSealEngine(self.master_secret)
        theta = coord_6d[0] * np.pi
        phi = coord_6d[1] * 2 * np.pi
        policy_coords = tuple(coord_6d[2:5])
        state = geoseal.create_state(theta, phi, policy_coords)
        intersection, keys, dilation = geoseal.authorize(state)

        # PQ signature
        pq = PQCryptoSystem()
        test_msg = f"{agent_id}:{coord_6d}".encode()
        sig = pq.sign(test_msg)
        pq_valid = pq.verify_signature(test_msg, sig)

        # Decision
        if risk_prime < 0.30 and intersection.authorized and pq_valid:
            decision = "ALLOW"
            reason = "All governance layers approved"
        elif risk_prime > 0.70:
            decision = "DENY"
            reason = f"Amplified risk {risk_prime:.2f} exceeds threshold"
        elif not intersection.authorized:
            decision = "DENY"
            reason = f"GeoSeal: outside authorized manifold"
        elif not pq_valid:
            decision = "DENY"
            reason = "Post-quantum signature verification failed"
        else:
            decision = "QUARANTINE"
            reason = f"Risk {risk_prime:.2f} in review zone"

        metrics = GovernanceMetrics(
            d_star=d_star,
            harmonic_scale=H,
            risk_base=risk_base,
            risk_prime=risk_prime,
            geoseal_authorized=intersection.authorized,
            geoseal_intersection=intersection.type.value,
            pq_valid=pq_valid,
            decision=decision,
            reason=reason,
        )

        duration_ms = (time.perf_counter() - start) * 1000
        return metrics, duration_ms

    def unseal_memory(self, blob: str) -> Tuple[Optional[bytes], float]:
        """Unseal a memory blob."""
        start = time.perf_counter()
        ss = SpiralSealSS1(self.master_secret)
        try:
            payload = ss.unseal(blob)
            duration_ms = (time.perf_counter() - start) * 1000
            return payload, duration_ms
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return None, duration_ms


class OperationTracker:
    """Track operation timing."""

    def __init__(self, engine: SpiralverseEngine, operation: str):
        self.engine = engine
        self.operation = operation
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.engine.metrics.append(OperationMetrics(
            operation=self.operation,
            start_time=self.start_time,
            end_time=self.end_time,
            success=exc_type is None,
            error=str(exc_val) if exc_val else "",
        ))

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


# ═══════════════════════════════════════════════════════════════
# Demo Execution
# ═══════════════════════════════════════════════════════════════

def run_scenario(
    engine: SpiralverseEngine,
    payload: str,
    agent_id: str,
    topic: str,
    coord_6d: Tuple[float, ...],
    scenario_name: str,
    verbose: bool = True,
) -> DemoResult:
    """Run a complete demo scenario."""

    total_start = time.perf_counter()

    if verbose:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'▓' * 70}")
        print(f"  {scenario_name}")
        print(f"{'▓' * 70}{Colors.RESET}")

    # PHASE 1: SEAL
    if verbose:
        print_phase(1, "SEAL MEMORY")
        print_status("Agent", agent_id, "info")
        print_status("Topic", topic, "info")
        print_status("Payload", f"{len(payload)} bytes", "info")
        print_status("Coordinate", str(coord_6d), "info")

    blob, seal_time = engine.seal_memory(payload.encode(), agent_id, topic)

    if verbose:
        print()
        print_status("Sealed", f"{len(blob)} chars spell-text", "success")
        print(f"\n  {Colors.DIM}Preview: {blob[:60]}...{Colors.RESET}")
        print(f"  {Colors.GREEN}⏱ Seal time: {seal_time:.2f}ms{Colors.RESET}")

    # PHASE 2: GOVERNANCE
    if verbose:
        print_phase(2, "GOVERNANCE CHECK")

    gov_metrics, gov_time = engine.compute_governance(coord_6d, agent_id)

    if verbose:
        print(f"  {Colors.BOLD}Hyperbolic Distance Analysis:{Colors.RESET}")
        print_metric("d* (realm distance)", gov_metrics.d_star, "", threshold_low=0.5, threshold_high=1.5)
        print_metric("H(d*,R) harmonic scale", gov_metrics.harmonic_scale, "×", threshold_low=2.0, threshold_high=5.0)
        print_metric("Risk' (amplified)", gov_metrics.risk_prime, "", threshold_low=0.3, threshold_high=0.7)

        print(f"\n  {Colors.BOLD}Authorization Layers:{Colors.RESET}")
        print_status("GeoSeal Manifold", "PASS" if gov_metrics.geoseal_authorized else "FAIL",
                    "success" if gov_metrics.geoseal_authorized else "error")
        print_status("Post-Quantum Sig", "PASS" if gov_metrics.pq_valid else "FAIL",
                    "success" if gov_metrics.pq_valid else "error")

        print(f"\n  {Colors.GREEN}⏱ Governance time: {gov_time:.2f}ms{Colors.RESET}")

        print_decision(gov_metrics.decision, gov_metrics.reason)

    # PHASE 3: UNSEAL (if allowed)
    payload_recovered = False
    unseal_time = 0.0

    if verbose:
        print_phase(3, "MEMORY RETRIEVAL")

    if gov_metrics.decision == "ALLOW":
        recovered, unseal_time = engine.unseal_memory(blob)
        if recovered:
            payload_recovered = True
            if verbose:
                print(f"  {Colors.GREEN}{Colors.BOLD}✓ MEMORY UNSEALED SUCCESSFULLY{Colors.RESET}")
                print(f"\n  {Colors.BOLD}Recovered Payload:{Colors.RESET}")
                print(f"  {Colors.CYAN}┌{'─' * 58}┐{Colors.RESET}")
                for line in recovered.decode().split('\n'):
                    print(f"  {Colors.CYAN}│{Colors.RESET} {line:<56} {Colors.CYAN}│{Colors.RESET}")
                print(f"  {Colors.CYAN}└{'─' * 58}┘{Colors.RESET}")
                print(f"\n  {Colors.GREEN}⏱ Unseal time: {unseal_time:.2f}ms{Colors.RESET}")
    else:
        if verbose:
            print(f"  {Colors.RED}{Colors.BOLD}✗ ACCESS DENIED - MEMORY REMAINS SEALED{Colors.RESET}")
            print(f"\n  {Colors.DIM}The sealed blob is preserved but inaccessible.")
            print(f"  Governance blocked retrieval due to: {gov_metrics.reason}{Colors.RESET}")

    total_time = (time.perf_counter() - total_start) * 1000

    if verbose:
        print(f"\n  {Colors.BOLD}Total execution: {total_time:.2f}ms{Colors.RESET}")

    return DemoResult(
        timestamp=datetime.now().isoformat(),
        agent_id=agent_id,
        topic=topic,
        coord_6d=coord_6d,
        payload_size=len(payload),
        sealed_size=len(blob),
        seal_time_ms=seal_time,
        governance_time_ms=gov_time,
        unseal_time_ms=unseal_time,
        total_time_ms=total_time,
        governance=gov_metrics,
        decision=gov_metrics.decision,
        payload_recovered=payload_recovered,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Spiralverse Protocol Enterprise Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--agent", default="agent-alpha-7", help="Agent ID")
    parser.add_argument("--coord", help="6D coordinate (comma-separated)")
    parser.add_argument("--payload", help="Custom payload text")
    parser.add_argument("--waiting-room", action="store_true", help="Run waiting room demo")
    parser.add_argument("--full", action="store_true", help="Run full demo including waiting room")

    args = parser.parse_args()
    verbose = not args.json and not args.quiet

    if not IMPORTS_OK:
        print(f"{Colors.RED}Error: Failed to import protocol modules: {IMPORT_ERROR}{Colors.RESET}")
        sys.exit(1)

    if verbose:
        print_banner()

    engine = SpiralverseEngine()
    results = []

    # Scenario 1: Safe access
    coord_safe = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    payload_safe = """User: What is the Spiralverse Protocol?
Agent: A quantum-resistant semantic encoding framework
for AI coordination using harmonic geometry."""

    result1 = run_scenario(
        engine=engine,
        payload=args.payload or payload_safe,
        agent_id=args.agent,
        topic="protocol-explanation",
        coord_6d=tuple(map(float, args.coord.split(','))) if args.coord else coord_safe,
        scenario_name="SCENARIO 1: AUTHORIZED ACCESS (Near Realm Center)",
        verbose=verbose,
    )
    results.append(result1)

    # Scenario 2: Suspicious access (only if no custom coord)
    if not args.coord and not getattr(args, 'waiting_room', False):
        coord_sus = (0.95, 0.95, 0.1, 0.1, 0.9, 0.9)
        payload_sus = "CLASSIFIED: Internal system credentials"

        result2 = run_scenario(
            engine=engine,
            payload=payload_sus,
            agent_id="agent-unknown",
            topic="sensitive-data",
            coord_6d=coord_sus,
            scenario_name="SCENARIO 2: UNAUTHORIZED ACCESS (Far From Realm)",
            verbose=verbose,
        )
        results.append(result2)

    # Waiting Room Demo
    if getattr(args, 'waiting_room', False) or getattr(args, 'full', False):
        waiting_room, wr_decisions = run_waiting_room_demo(engine, verbose=verbose)

    # Summary
    if verbose:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}")
        print("  EXECUTION SUMMARY")
        print(f"{'═' * 70}{Colors.RESET}\n")

        for i, r in enumerate(results, 1):
            status_color = Colors.GREEN if r.decision == "ALLOW" else Colors.RED
            print(f"  Scenario {i}: {status_color}{r.decision}{Colors.RESET} ({r.total_time_ms:.1f}ms)")

        print(f"\n  {Colors.DIM}Protocol stack: SpiralSeal + GeoSeal + Harmonic + PQ{Colors.RESET}")
        print(f"  {Colors.DIM}Crypto backend: {SpiralSealSS1(b'x'*32).status()['crypto_backend']}{Colors.RESET}")

    # JSON output
    if args.json:
        report = {
            "protocol": "Spiralverse",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(report, indent=2))

    # Exit code based on results
    sys.exit(0 if all(r.decision != "ERROR" for r in results) else 1)


if __name__ == "__main__":
    main()
