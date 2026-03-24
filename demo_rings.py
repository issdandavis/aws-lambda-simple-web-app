#!/usr/bin/env python3
"""
Concentric Ring System Demonstration

Shows the multi-ring trust topology in action with plain English explanations.
Think of it as airport security - different zones with different levels of scrutiny.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scbe_aethermoore.geoseal import (
    GeoSealConfig,
    SphereConfig,
    ConcentricRingSystem,
    RingConfig,
    TimeDilationConfig,
    project_to_sphere,
    classify_ring,
    compute_ring_time_dilation,
    derive_ring_keys,
    compute_radial_distance,
    visualize_rings,
    CellMembership,
    PathType,
)


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}\n")


def print_subsection(title):
    print(f"\n  --- {title} ---\n")


def demo_explain_rings():
    """Explain the ring concept in simple terms."""
    print_section("WHAT ARE CONCENTRIC RINGS? (Plain English)")

    print("""
    Imagine AIRPORT SECURITY with multiple zones:

    +-----------------------------------------------------------+
    |                                                           |
    |     EXTERIOR (Ring 4): Public area outside security       |
    |         - Anyone can be here                              |
    |         - Maximum scrutiny to enter                       |
    |         - Slowest processing, most ID checks              |
    |                                                           |
    |     +-----------------------------------------------+     |
    |     |                                               |     |
    |     |   BOUNDARY (Ring 3): Just past security      |     |
    |     |       - You've been checked once             |     |
    |     |       - Still getting extra scrutiny         |     |
    |     |                                               |     |
    |     |   +---------------------------------------+   |     |
    |     |   |                                       |   |     |
    |     |   |  VERIFIED (Ring 2): Gate area        |   |     |
    |     |   |      - You have a boarding pass      |   |     |
    |     |   |      - Moderate trust level          |   |     |
    |     |   |                                       |   |     |
    |     |   |   +-------------------------------+   |   |     |
    |     |   |   |                               |   |   |     |
    |     |   |   | TRUSTED (Ring 1): Lounge     |   |   |     |
    |     |   |   |     - You're a known flyer   |   |   |     |
    |     |   |   |     - Fast-track access      |   |   |     |
    |     |   |   |                               |   |   |     |
    |     |   |   |   +-------------------+       |   |   |     |
    |     |   |   |   |                   |       |   |   |     |
    |     |   |   |   | CORE (Ring 0)    |       |   |   |     |
    |     |   |   |   |   Cockpit crew   |       |   |   |     |
    |     |   |   |   |   Maximum trust  |       |   |   |     |
    |     |   |   |   |   Zero delay     |       |   |   |     |
    |     |   |   |   +-------------------+       |   |   |     |
    |     |   |   +-------------------------------+   |   |     |
    |     |   +---------------------------------------+   |     |
    |     +-----------------------------------------------+     |
    +-----------------------------------------------------------+

    WHY RINGS?

    1. GRADUATED TRUST
       Not everyone needs the same level of access.
       Your "normal" behavior puts you in a home ring.
       Unusual behavior moves you outward to slower rings.

    2. TIME IS SECURITY
       Inner rings = FAST (milliseconds)
       Outer rings = SLOW (seconds, more checks)
       Attackers get stuck in "time tar"

    3. DIFFERENT KEYS FOR DIFFERENT RINGS
       Even with the password, you need to be in the right ring.
       A key from Ring 1 doesn't work in Ring 3.

    4. TRANSITION ZONES
       The edges between rings are special.
       Extra scrutiny when crossing boundaries.
       Like the door between airport zones.
    """)


def demo_ring_classification():
    """Show how different behaviors land in different rings."""
    print_section("STEP 1: WHERE DO YOU LAND?")

    ring_system = ConcentricRingSystem()
    sphere_config = SphereConfig(n_dims=3, level=4, feature_indices=(3, 4, 5))

    # Define center (the "ideal" trusted state)
    center = np.array([1.0, 0.0, 0.0])  # North pole = ideal

    # Different behavioral contexts
    scenarios = [
        {
            "name": "Perfect Behavior",
            "desc": "Calm, stable, exactly as expected",
            "context": np.array([100.0, 42.0, 0.5, 0.99, 0.10, 0.99]),
            "expected_ring": "core"
        },
        {
            "name": "Normal Operation",
            "desc": "Slightly off baseline, still good",
            "context": np.array([100.0, 42.0, 1.0, 0.85, 0.25, 0.92]),
            "expected_ring": "trusted"
        },
        {
            "name": "Mild Anomaly",
            "desc": "Something's a bit off, worth watching",
            "context": np.array([100.0, 42.0, 3.0, 0.70, 0.45, 0.75]),
            "expected_ring": "verified"
        },
        {
            "name": "Stressed System",
            "desc": "High load, entropy dropping",
            "context": np.array([100.0, 42.0, 5.0, 0.55, 0.70, 0.50]),
            "expected_ring": "boundary"
        },
        {
            "name": "Full Alarm",
            "desc": "Something is very wrong",
            "context": np.array([100.0, 42.0, 9.0, 0.30, 0.95, 0.20]),
            "expected_ring": "exterior"
        },
    ]

    for scenario in scenarios:
        u = project_to_sphere(scenario["context"], sphere_config)
        ring_mem = classify_ring(u, ring_system, center)

        print(f"  {scenario['name']}")
        print(f"    Description: {scenario['desc']}")
        print(f"    Sphere point: ({u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f})")
        print(f"    Radial distance: {ring_mem.radial_distance:.4f}")
        print(f"    --> Ring {ring_mem.ring_index}: {ring_mem.ring_name.upper()}")
        print(f"    Trust level: {ring_mem.trust_level:.2f}")
        print(f"    In transition zone: {'YES' if ring_mem.is_transition_zone else 'NO'}")
        print()


def demo_time_dilation_by_ring():
    """Show how time works differently in each ring."""
    print_section("STEP 2: TIME MOVES DIFFERENTLY IN EACH RING")

    ring_system = ConcentricRingSystem()
    time_config = TimeDilationConfig()
    sphere_config = SphereConfig(n_dims=3, level=4, feature_indices=(3, 4, 5))
    center = np.array([1.0, 0.0, 0.0])

    print("  RING-BY-RING TIME DILATION:\n")
    print("  Ring        | Latency Budget | PoW Bits | Notes")
    print("  ------------|----------------|----------|------------------")

    # Sample contexts that land in each ring
    test_contexts = [
        np.array([100.0, 42.0, 0.5, 0.99, 0.05, 0.99]),  # Core
        np.array([100.0, 42.0, 1.0, 0.85, 0.20, 0.90]),  # Trusted
        np.array([100.0, 42.0, 2.5, 0.70, 0.40, 0.70]),  # Verified
        np.array([100.0, 42.0, 5.0, 0.50, 0.65, 0.45]),  # Boundary
        np.array([100.0, 42.0, 8.0, 0.30, 0.90, 0.20]),  # Exterior
    ]

    for ctx in test_contexts:
        u = project_to_sphere(ctx, sphere_config)
        ring_mem = classify_ring(u, ring_system, center)
        latency, pow_bits = compute_ring_time_dilation(ring_mem, ring_system, time_config)

        ring = ring_system.get_ring(ring_mem.ring_index)
        notes = ""
        if ring.requires_attestation:
            notes = "Needs attestation"
        if ring_mem.is_transition_zone:
            notes += " [TRANSITION]"

        print(f"  {ring_mem.ring_name:11s} | {latency:10.1f} ms  | {pow_bits:8d} | {notes}")

    print("""

  WHAT THIS MEANS:

  - CORE users: ~10ms response, no proof-of-work needed
    Like walking through a door you have a key for.

  - TRUSTED users: ~30ms response, still no PoW
    Fast lane, but not instant.

  - VERIFIED users: ~60ms response, 1 bit of PoW
    A small puzzle to solve before proceeding.

  - BOUNDARY users: ~100ms response, 3 bits of PoW
    Noticeable delay, real computational work.

  - EXTERIOR users: ~200ms response, 6 bits of PoW
    Significant delay, meaningful computational barrier.

  ATTACKERS trying to brute-force from EXTERIOR get:
  - 200ms per attempt
  - 64x computational slowdown from PoW
  - Each wrong attempt stays in slow zone

  LEGITIMATE USERS who drift outward:
  - Get automatic slowdown (time to investigate)
  - Can move back inward as trust rebuilds
    """)


def demo_ring_specific_keys():
    """Show how each ring gets different cryptographic keys."""
    print_section("STEP 3: DIFFERENT RINGS, DIFFERENT KEYS")

    ring_system = ConcentricRingSystem()
    sphere_config = SphereConfig(n_dims=3, level=4, feature_indices=(3, 4, 5))
    center = np.array([1.0, 0.0, 0.0])

    shared_secret = b"shared_secret_from_kyber_kem_32b"

    # Mock membership for demonstration
    base_membership = CellMembership(
        h=42, z=128, L_s=4, L_c=4,
        in_allowed_sphere=True, in_allowed_cube=True, margin=0.5
    )

    print("  SAME shared secret, DIFFERENT rings:\n")

    test_contexts = [
        ("core", np.array([100.0, 42.0, 0.5, 0.99, 0.05, 0.99])),
        ("verified", np.array([100.0, 42.0, 3.0, 0.70, 0.45, 0.70])),
        ("exterior", np.array([100.0, 42.0, 8.0, 0.30, 0.90, 0.20])),
    ]

    for name, ctx in test_contexts:
        u = project_to_sphere(ctx, sphere_config)
        ring_mem = classify_ring(u, ring_system, center)

        keys = derive_ring_keys(shared_secret, ring_mem, base_membership, PathType.INTERIOR)

        print(f"  RING: {ring_mem.ring_name.upper()}")
        print(f"    K_ring:     {keys['K_ring'][:8].hex()}...")
        print(f"    K_msg_ring: {keys['K_msg_ring'][:8].hex()}...")
        print()

    print("""
  WHY THIS MATTERS:

  1. KEY ISOLATION
     A key derived in Ring 0 (core) is USELESS in Ring 3 (boundary).
     Even if an attacker steals a key, it only works at that exact position.

  2. RING BINDING
     The ring index and name are mixed into key derivation.
     Moving between rings = completely different keys.

  3. RADIAL BINDING
     Even within a ring, your exact position matters.
     The radial distance is baked into the key.

  4. DEFENSE IN DEPTH
     An attacker would need to:
     - Be in the right ring
     - At the right radial position
     - With the right sphere cell
     - With the right cube cell
     - With the right path type
     - All at once!
    """)


def demo_transition_zones():
    """Show special handling at ring boundaries."""
    print_section("STEP 4: THE EDGES (TRANSITION ZONES)")

    ring_system = ConcentricRingSystem()
    time_config = TimeDilationConfig()
    sphere_config = SphereConfig(n_dims=3, level=4, feature_indices=(3, 4, 5))
    center = np.array([1.0, 0.0, 0.0])

    print("""
  At the EDGE between rings, extra caution is applied:

  Ring Boundary     | What Happens
  ------------------|----------------------------------------
  Core -> Trusted   | Minor slowdown, watching for drift
  Trusted -> Verified | First attestation required
  Verified -> Boundary | Significant delay increase
  Boundary -> Exterior | Maximum scrutiny begins

  EXAMPLE - Points near the Trusted/Verified boundary:
    """)

    # Points near the 0.3 boundary (Trusted -> Verified)
    boundary_contexts = [
        ("Deep in Trusted", np.array([100.0, 42.0, 1.0, 0.87, 0.18, 0.93])),
        ("Edge of Trusted", np.array([100.0, 42.0, 1.5, 0.82, 0.26, 0.88])),
        ("Edge of Verified", np.array([100.0, 42.0, 2.0, 0.78, 0.32, 0.82])),
        ("Deep in Verified", np.array([100.0, 42.0, 2.8, 0.72, 0.40, 0.74])),
    ]

    for name, ctx in boundary_contexts:
        u = project_to_sphere(ctx, sphere_config)
        ring_mem = classify_ring(u, ring_system, center)
        latency, pow_bits = compute_ring_time_dilation(ring_mem, ring_system, time_config)

        transition_marker = " [TRANSITION ZONE]" if ring_mem.is_transition_zone else ""
        print(f"  {name}:")
        print(f"    Ring: {ring_mem.ring_name}, Depth: {ring_mem.depth_in_ring:.2f}{transition_marker}")
        print(f"    Latency: {latency:.1f}ms, PoW: {pow_bits} bits")
        print()


def demo_custom_ring_config():
    """Show how to customize the ring system."""
    print_section("STEP 5: CUSTOM RING CONFIGURATIONS")

    print("""
  You can define your OWN ring system for different use cases:
    """)

    # Financial system - more paranoid
    financial_rings = ConcentricRingSystem(rings=[
        RingConfig("core", 0.0, 0.05, 1.0, 0.05, 0, False),      # Very tight core
        RingConfig("trusted", 0.05, 0.15, 0.9, 0.2, 0, False),   # Small trusted zone
        RingConfig("verified", 0.15, 0.35, 0.7, 0.5, 2, True),   # Medium verified
        RingConfig("reviewed", 0.35, 0.6, 0.5, 1.0, 4, True),    # New: reviewed zone
        RingConfig("boundary", 0.6, 0.85, 0.3, 2.0, 6, True),    # Larger boundary
        RingConfig("exterior", 0.85, 1.0, 0.1, 5.0, 10, True),   # Very slow exterior
    ])

    print("  FINANCIAL SYSTEM (6 rings, paranoid):")
    print("  " + "-" * 55)
    for i, ring in enumerate(financial_rings.rings):
        print(f"  Ring {i} [{ring.name:10s}]: r=[{ring.inner_radius:.2f}-{ring.outer_radius:.2f}] "
              f"trust={ring.trust_level:.1f} latency_mult={ring.latency_multiplier:.1f}x")

    print()

    # IoT system - more permissive
    iot_rings = ConcentricRingSystem(rings=[
        RingConfig("core", 0.0, 0.2, 1.0, 0.1, 0, False),        # Larger core
        RingConfig("trusted", 0.2, 0.5, 0.8, 0.3, 0, False),     # Large trusted
        RingConfig("standard", 0.5, 0.8, 0.5, 0.7, 1, True),     # Standard ops
        RingConfig("exterior", 0.8, 1.0, 0.2, 1.5, 3, True),     # Smaller exterior
    ])

    print("  IOT SYSTEM (4 rings, permissive):")
    print("  " + "-" * 55)
    for i, ring in enumerate(iot_rings.rings):
        print(f"  Ring {i} [{ring.name:10s}]: r=[{ring.inner_radius:.2f}-{ring.outer_radius:.2f}] "
              f"trust={ring.trust_level:.1f} latency_mult={ring.latency_multiplier:.1f}x")


def demo_visual_ring_map():
    """Show a visual representation of the ring system."""
    print_section("VISUAL: THE CONCENTRIC RING MAP")

    ring_system = ConcentricRingSystem()
    sphere_config = SphereConfig(n_dims=3, level=4, feature_indices=(3, 4, 5))
    center = np.array([1.0, 0.0, 0.0])

    # Place a sample point
    sample_context = np.array([100.0, 42.0, 2.5, 0.72, 0.38, 0.76])
    u = project_to_sphere(sample_context, sphere_config)
    ring_mem = classify_ring(u, ring_system, center)

    print(visualize_rings(ring_mem, ring_system))


def main():
    print("\n" + "=" * 65)
    print("     CONCENTRIC RING SYSTEM DEMONSTRATION")
    print("     'Security through layered geometry'")
    print("=" * 65)

    demo_explain_rings()
    demo_ring_classification()
    demo_time_dilation_by_ring()
    demo_ring_specific_keys()
    demo_transition_zones()
    demo_custom_ring_config()
    demo_visual_ring_map()

    print_section("SUMMARY: WHAT THE RINGS ADD")

    print("""
    The CONCENTRIC RING SYSTEM adds LAYERED SECURITY:

    1. GRADUATED RESPONSE
       Not binary (in/out) but continuous (5+ rings).
       Your position determines your treatment.

    2. AUTOMATIC THROTTLING
       Drifting outward = automatic slowdown.
       No manual intervention needed.

    3. RING-BOUND KEYS
       Cryptographic keys are TIED to ring position.
       Stolen keys don't work in other rings.

    4. TRANSITION MONITORING
       Ring boundaries are special checkpoints.
       Crossing triggers extra scrutiny.

    5. CUSTOMIZABLE TOPOLOGY
       Different applications get different ring layouts.
       Financial = tight, IoT = loose, etc.

    COMBINED WITH GEOSEAL:
    - Sphere (behavior) --> which RING you're in
    - Cube (policy) --> what you're ALLOWED to do
    - Path (interior/exterior) --> HOW FAST it happens
    - Ring position --> WHAT KEYS you get

    Four-dimensional security: Behavior + Policy + Path + Ring
    """)

    print("=" * 65)
    print("     DEMONSTRATION COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
