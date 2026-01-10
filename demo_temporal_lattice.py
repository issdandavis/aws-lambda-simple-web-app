#!/usr/bin/env python3
"""
Temporal Lattice Demonstration

Shows how equations can start UNSTABLE and crystallize on ARRIVAL
when all 7 vertices align and dual lattice verifies.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scbe_aethermoore.temporal_lattice import (
    TemporalEquation,
    StabilizationState,
    create_temporal_equation,
    oscillate_equation,
    attempt_crystallization,
    time_dilation_error,
    visualize_temporal_equation,
    explain_temporal_lattice,
    TimeAxisConfig,
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_explain():
    """Show plain English explanation."""
    print_section("THE CONCEPT: Time as a Stabilization Axis")
    print(explain_temporal_lattice())


def demo_7_vertices():
    """Show how the 7 vertices work."""
    print_section("THE 7 VERTICES")

    print("""
    Every temporal equation has 7 "corners" that must align:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │                        TIME (when)                          │
    │                           ▲                                 │
    │                           │                                 │
    │         X ◄───────────────┼───────────────► Y               │
    │        (behavior)         │              (behavior)         │
    │                           │                                 │
    │                           ▼                                 │
    │                        Z (behavior)                         │
    │                                                             │
    │    + POLICY (what rules)                                    │
    │    + RING (trust level)                                     │
    │    + INTENT (why)                                           │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Think of it as a 7-dimensional cube.
    ALL 7 corners must "click" into place for the equation to stabilize.
    """)

    # Create sample equation
    context = np.array([100.0, 42.0, 2.0, 0.8, 0.3, 0.9])
    policy = {"tier": 0.3, "intent": 0.5, "data_class": 0.2, "safety": 0.8}
    ring_position = 0.2
    intent = "sil'kor/nav'een"
    target_time = time.time() + 2.0  # 2 seconds from now
    shared_secret = b"shared_secret_for_demo_32bytes"
    signing_key = b"signing_key_for_dilithium_demo"

    equation = create_temporal_equation(
        context, policy, ring_position, intent, target_time,
        shared_secret, signing_key
    )

    print("  SAMPLE EQUATION CREATED:")
    print(f"    Target time: {target_time:.2f}")
    print(f"    Current state: {equation.state.value}")
    print()
    print("  Vertices:")
    for v in equation.vertices:
        bar_len = int(v.value * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {v.name:12s}: [{bar}] {v.value:.4f}")


def demo_oscillation():
    """Show the oscillation process."""
    print_section("OSCILLATION: Unstable → Oscillating → Crystallized")

    print("""
    The equation doesn't start stable. It VIBRATES until conditions align:

    Step 1: UNSTABLE (water)
            〰️〰️〰️〰️〰️〰️〰️〰️
            Can't grab it, too fluid

    Step 2: OSCILLATING (jello)
            ∿∿∿∿∿∿∿∿∿∿∿∿
            Starting to hold shape, still wiggles

    Step 3: CRYSTALLIZED (ice)
            ◈◈◈◈◈◈◈◈◈◈◈◈
            Solid! Now you can use it

    Step X: COLLAPSED (error)
            ✗✗✗✗✗✗✗✗✗✗✗✗
            Too many oscillations, never stabilized
    """)

    # Create and oscillate
    context = np.array([100.0, 42.0, 2.0, 0.8, 0.3, 0.9])
    policy = {"tier": 0.3, "intent": 0.5}
    target_time = time.time() + 0.5

    equation = create_temporal_equation(
        context, policy, 0.2, "test/intent", target_time,
        b"secret32bytesecret32bytesecret!", b"sign32bytesign32bytesign32bytes"
    )

    print("  WATCHING OSCILLATION:")
    print(f"    Starting state: {equation.state.value}")
    print()

    for i in range(10):
        state, score = oscillate_equation(equation, context, time.time())
        state_char = {
            StabilizationState.UNSTABLE: "〰️",
            StabilizationState.OSCILLATING: "∿",
            StabilizationState.CRYSTALLIZED: "◈",
            StabilizationState.COLLAPSED: "✗"
        }.get(state, "?")

        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    Cycle {i+1:2d}: {state_char} [{bar}] {score:.2%}")

        if state in [StabilizationState.CRYSTALLIZED, StabilizationState.COLLAPSED]:
            break

    print(f"\n    Final state: {equation.state.value}")


def demo_dual_lattice():
    """Show dual lattice verification."""
    print_section("DUAL LATTICE: Kyber + Dilithium")

    print("""
    TWO separate quantum-safe systems must BOTH agree:

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │   ┌──────────────────┐        ┌──────────────────┐            │
    │   │      KYBER       │        │    DILITHIUM     │            │
    │   │   (Lattice KEM)  │        │ (Lattice Sigs)   │            │
    │   │                  │        │                  │            │
    │   │  Creates shared  │        │  Signs the       │            │
    │   │  secret key      │        │  whole thing     │            │
    │   │                  │        │                  │            │
    │   │  Based on:       │        │  Based on:       │            │
    │   │  Module-LWE      │        │  Module-LWE +    │            │
    │   │  (hard lattice   │        │  Module-SIS      │            │
    │   │   problem)       │        │  (different      │            │
    │   │                  │        │   hard problems) │            │
    │   └────────┬─────────┘        └────────┬─────────┘            │
    │            │                           │                       │
    │            └───────────┬───────────────┘                       │
    │                        │                                       │
    │                        ▼                                       │
    │              ┌─────────────────┐                               │
    │              │  BOTH MUST PASS │                               │
    │              │    to unlock    │                               │
    │              └─────────────────┘                               │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    WHY TWO?
    ─────────
    If quantum computers crack Kyber someday:
      → Dilithium signature still protects authenticity
      → Attacker can't forge the "who signed this"

    If quantum computers crack Dilithium someday:
      → Kyber key exchange still protects secrecy
      → Attacker can't read the contents

    To break the system, you need to crack BOTH.
    That's exponentially harder.
    """)


def demo_time_dilation_error():
    """Show the time dilation trapdoor."""
    print_section("TIME DILATION TRAPDOOR")

    print("""
    If time skew is too large, the equation "errors out":

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │   Normal time:      ──────────────────────────────────►       │
    │                     Equation oscillates, may crystallize       │
    │                                                                │
    │   Small skew:       ─────────〰️──────────────────────►       │
    │                     Slight wobble, still works                 │
    │                                                                │
    │   Large skew:       ─────❌                                   │
    │                     ERROR! Time dilation → ∞                   │
    │                     Like falling into a black hole             │
    │                     Equation NEVER crystallizes                │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    This is a TRAPDOOR for attackers:
    - If they try to manipulate timestamps
    - If they try to replay old equations
    - If they're in a different time zone than expected
    → They get trapped in infinite dilation (nothing works)
    """)

    # Demonstrate
    context = np.array([100.0, 42.0, 2.0, 0.8, 0.3, 0.9])
    equation = create_temporal_equation(
        context, {"tier": 0.5}, 0.3, "test", time.time() + 1.0,
        b"secret32bytesecret32bytesecret!", b"sign32bytesign32bytesign32bytes"
    )

    test_cases = [
        ("Perfect time", equation.target_time),
        ("0.5s early", equation.target_time - 0.5),
        ("0.9s late", equation.target_time + 0.9),
        ("2s late (ERROR)", equation.target_time + 2.0),
        ("10s late (ERROR)", equation.target_time + 10.0),
    ]

    print("  TIME SKEW TESTS:")
    print()
    for name, observed in test_cases:
        is_error, dilation = time_dilation_error(equation, observed, equation.target_time)
        if is_error:
            status = "✗ ERROR (trapped)"
            dilation_str = "∞"
        else:
            status = "✓ OK"
            dilation_str = f"{dilation:.2f}x"

        print(f"    {name:20s}: {status:20s} Dilation: {dilation_str}")


def demo_full_crystallization():
    """Show full crystallization attempt."""
    print_section("FULL CRYSTALLIZATION ATTEMPT")

    print("  Creating temporal equation...")
    print()

    context = np.array([100.0, 42.0, 2.0, 0.85, 0.25, 0.92])
    policy = {"tier": 0.3, "intent": 0.5, "data_class": 0.2, "safety": 0.8}

    # Target 1 second from now
    target_time = time.time() + 1.0

    equation = create_temporal_equation(
        context, policy, 0.15, "sil'kor/nav'een", target_time,
        b"shared_secret_for_demo_32bytes", b"signing_key_for_dilithium_demo"
    )

    print(visualize_temporal_equation(equation))
    print()

    print("  Attempting crystallization...")
    print("  (Waiting for target time window...)")
    print()

    # Wait for target time
    wait_time = target_time - time.time()
    if wait_time > 0:
        time.sleep(min(wait_time, 0.5))  # Don't wait too long for demo

    # Attempt crystallization
    config = TimeAxisConfig(
        oscillation_period=0.05,
        max_drift=2.0,
        crystallization_window=1.0
    )

    success, value = attempt_crystallization(
        equation, context,
        b"shared_secret_for_demo_32bytes",
        config
    )

    print(visualize_temporal_equation(equation))
    print()

    if success:
        print(f"  ◈ CRYSTALLIZED! Value: {value[:16].hex()}...")
    else:
        print(f"  ✗ Failed to crystallize. State: {equation.state.value}")


def demo_attack_scenario():
    """Show how attacks fail."""
    print_section("ATTACK SCENARIO: Why This Is Secure")

    print("""
    ATTACK 1: Wrong Context
    ────────────────────────
    Attacker has the equation but different behavior.
    → Vertices don't align
    → Equation never crystallizes
    → Just get noise

    ATTACK 2: Wrong Time
    ────────────────────────
    Attacker tries to use equation at different time.
    → Time dilation trapdoor activates
    → Infinite dilation = frozen
    → Nothing happens

    ATTACK 3: Crack Kyber
    ────────────────────────
    Hypothetical: Quantum computer breaks Kyber.
    → Dilithium signature still required
    → Can't forge the signature
    → Attack fails

    ATTACK 4: Crack Dilithium
    ────────────────────────
    Hypothetical: Quantum computer breaks Dilithium.
    → Kyber shared secret still required
    → Can't decrypt without it
    → Attack fails

    ATTACK 5: Replay Old Equation
    ────────────────────────
    Attacker records equation, replays later.
    → Time vertex doesn't align
    → Oscillation fails
    → Just get noise

    THE DEFENSE:
    ─────────────
    • 7 vertices must ALL align (multiplicative security)
    • Dual lattice = need to crack BOTH (exponential hardness)
    • Time trapdoor = wrong timing = infinite freeze
    • Fail-to-noise = attacker can't even tell they failed
    """)


def main():
    print("\n" + "=" * 70)
    print("     TEMPORAL LATTICE DEMONSTRATION")
    print("     'Equations that stabilize on arrival'")
    print("=" * 70)

    demo_explain()
    demo_7_vertices()
    demo_oscillation()
    demo_dual_lattice()
    demo_time_dilation_error()
    demo_full_crystallization()
    demo_attack_scenario()

    print_section("SUMMARY")
    print("""
    WHAT YOU INVENTED:

    1. TIME AS A DIMENSION
       Equations don't need to be stable at the start.
       They crystallize on arrival when conditions align.

    2. THE 7 VERTICES
       Time + 3D behavior + policy + ring + intent
       All 7 must align for the equation to solidify.

    3. DUAL LATTICE SECURITY
       Kyber (key exchange) + Dilithium (signatures)
       Both must verify. Crack one, the other still protects.

    4. TIME DILATION TRAPDOOR
       Wrong timing = infinite dilation = frozen/trapped
       Attackers can't manipulate time without triggering it.

    5. OSCILLATION TO CRYSTALLIZATION
       The math "vibrates" between states until all conditions
       synchronize, then locks into solid usable form.

    IN ONE SENTENCE:
    ─────────────────
    "A quantum-safe equation that floats unstable through time
     until the exact moment when all 7 dimensions align and
     both lattice systems verify, then crystallizes solid."
    """)

    print("=" * 70)
    print("     DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
