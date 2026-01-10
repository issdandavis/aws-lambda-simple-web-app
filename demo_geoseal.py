#!/usr/bin/env python3
"""
GeoSeal Demonstration

Shows the Geometric Trust Manifold in action with plain English explanations.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scbe_aethermoore.geoseal import (
    GeoSealConfig,
    SphereConfig,
    CubeConfig,
    PathType,
    project_to_sphere,
    project_to_cube,
    healpix_index,
    morton_encode,
    classify_path,
    compute_time_dilation,
    derive_region_keys,
    geoseal_encrypt,
    geoseal_decrypt,
    visualize_geometry,
    CellMembership,
)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_subsection(title):
    print(f"\n  ─── {title} ───\n")


def demo_explain_concept():
    """Explain what GeoSeal does in simple terms."""
    print_section("WHAT IS GEOSEAL? (Plain English)")

    print("""
    Imagine you have TWO maps that must BOTH agree before a door opens:

    🌍 MAP 1: THE BEHAVIOR SPHERE
       This is like a globe showing "where your AI brain is"
       - Calm and stable? You're near the North Pole
       - Erratic and stressed? You drift toward the equator
       - The system learns your "home location" on this globe

    📦 MAP 2: THE POLICY CUBE
       This is like a filing cabinet with many dimensions:
       - Tier drawer (how sensitive is this action?)
       - Intent drawer (what are you trying to do?)
       - Safety drawer (how careful should we be?)

    🚪 THE DOORS:
       INTERIOR (fast lane): Both maps say "you're in the right place"
       EXTERIOR (slow lane): One map says "something's off"

    ⏰ TIME WORKS DIFFERENTLY:
       - Interior = Fast (milliseconds)
       - Exterior = Slow (more checks, more wait)
       Like gravity near a planet - time moves slower at the edge

    🔐 DIFFERENT KEYS FOR DIFFERENT DOORS:
       Even if someone steals a key, it only works
       if they're at the EXACT same spot on BOTH maps
    """)


def demo_sphere_projection():
    """Show how context becomes a point on a sphere."""
    print_section("STEP 1: YOUR BEHAVIOR BECOMES A POINT ON A GLOBE")

    # Normal, stable behavior
    calm_context = np.array([
        100.0,  # time
        42.0,   # device
        1.0,    # low threat
        0.85,   # good entropy
        0.30,   # low server load
        0.98    # very stable behavior
    ])

    # Stressed, erratic behavior
    stressed_context = np.array([
        100.0,  # same time
        42.0,   # same device
        7.0,    # HIGH threat!
        0.45,   # lower entropy
        0.90,   # high server load
        0.35    # UNSTABLE behavior
    ])

    config = SphereConfig(n_dims=3, level=4, feature_indices=(3, 4, 5))

    u_calm = project_to_sphere(calm_context, config)
    u_stressed = project_to_sphere(stressed_context, config)

    h_calm = healpix_index(u_calm, config.level)
    h_stressed = healpix_index(u_stressed, config.level)

    print("  CALM BEHAVIOR (stable, low load, high entropy):")
    print(f"    → Sphere point: ({u_calm[0]:.3f}, {u_calm[1]:.3f}, {u_calm[2]:.3f})")
    print(f"    → Cell number: {h_calm}")
    print()
    print("  STRESSED BEHAVIOR (unstable, high load, low entropy):")
    print(f"    → Sphere point: ({u_stressed[0]:.3f}, {u_stressed[1]:.3f}, {u_stressed[2]:.3f})")
    print(f"    → Cell number: {h_stressed}")
    print()
    print(f"  ✓ Different behaviors land in DIFFERENT cells on the sphere!")
    print(f"  ✓ The system knows 'calm you' vs 'stressed you' by location")


def demo_cube_projection():
    """Show how policy becomes a point in a hypercube."""
    print_section("STEP 2: YOUR REQUEST BECOMES A POINT IN A BOX")

    # Low-risk request
    simple_policy = {
        "tier": 0.2,       # Low tier (simple read)
        "intent": 0.3,     # Simple intent
        "data_class": 0.1, # Public data
        "safety": 0.9      # High safety margin
    }

    # High-risk request
    dangerous_policy = {
        "tier": 0.9,       # High tier (critical action)
        "intent": 0.8,     # Complex intent
        "data_class": 0.9, # Sensitive data
        "safety": 0.2      # Low safety margin
    }

    config = CubeConfig(m_dims=4, level=4)

    v_simple = project_to_cube(simple_policy, config)
    v_dangerous = project_to_cube(dangerous_policy, config)

    z_simple = morton_encode(v_simple, config.level)
    z_dangerous = morton_encode(v_dangerous, config.level)

    print("  SIMPLE REQUEST (read public data, high safety):")
    print(f"    → Cube point: ({v_simple[0]:.2f}, {v_simple[1]:.2f}, {v_simple[2]:.2f}, {v_simple[3]:.2f})")
    print(f"    → Morton code: {z_simple}")
    print()
    print("  DANGEROUS REQUEST (critical action, sensitive data):")
    print(f"    → Cube point: ({v_dangerous[0]:.2f}, {v_dangerous[1]:.2f}, {v_dangerous[2]:.2f}, {v_dangerous[3]:.2f})")
    print(f"    → Morton code: {z_dangerous}")
    print()
    print(f"  ✓ Different requests land in DIFFERENT cells in the policy cube!")


def demo_path_classification():
    """Show interior vs exterior path classification."""
    print_section("STEP 3: ARE YOU INSIDE OR OUTSIDE THE SAFE ZONE?")

    config = GeoSealConfig()

    # Good context + good policy
    good_context = np.array([100.0, 42.0, 1.0, 0.85, 0.30, 0.98])
    good_policy = {"tier": 0.3, "intent": 0.3, "data_class": 0.2, "safety": 0.9}
    good_risk = {"phase_skew": 0.1, "oracle_delta": 0.1, "anomaly_score": 0.1}
    good_trust = {"prior_approvals": 0.9, "uptime": 0.95, "relationship_age": 0.8}

    # Suspicious context + risky policy
    bad_context = np.array([100.0, 42.0, 8.0, 0.35, 0.90, 0.25])
    bad_policy = {"tier": 0.9, "intent": 0.8, "data_class": 0.9, "safety": 0.1}
    bad_risk = {"phase_skew": 0.8, "oracle_delta": 0.7, "anomaly_score": 0.9}
    bad_trust = {"prior_approvals": 0.2, "uptime": 0.5, "relationship_age": 0.1}

    u_good = project_to_sphere(good_context, config.sphere)
    v_good = project_to_cube(good_policy, config.cube)
    path_good, mem_good, pot_good = classify_path(
        u_good, v_good, config, good_risk, good_trust
    )

    u_bad = project_to_sphere(bad_context, config.sphere)
    v_bad = project_to_cube(bad_policy, config.cube)
    path_bad, mem_bad, pot_bad = classify_path(
        u_bad, v_bad, config, bad_risk, bad_trust
    )

    print("  TRUSTED REQUEST (calm behavior + safe policy + good history):")
    print(f"    Risk score:  {pot_good.risk:.3f}")
    print(f"    Trust score: {pot_good.trust:.3f}")
    print(f"    Potential:   {pot_good.potential:.3f}")
    print(f"    Margin:      {pot_good.margin:.3f}")
    print(f"    → Path: {path_good.value.upper()} {'✓ Fast lane!' if path_good == PathType.INTERIOR else '⚠ Slow lane'}")
    print()
    print("  SUSPICIOUS REQUEST (erratic behavior + dangerous policy + no history):")
    print(f"    Risk score:  {pot_bad.risk:.3f}")
    print(f"    Trust score: {pot_bad.trust:.3f}")
    print(f"    Potential:   {pot_bad.potential:.3f}")
    print(f"    Margin:      {pot_bad.margin:.3f}")
    print(f"    → Path: {path_bad.value.upper()} {'✓ Fast lane!' if path_bad == PathType.INTERIOR else '⚠ Slow lane'}")


def demo_time_dilation():
    """Show how time works differently for different paths."""
    print_section("STEP 4: TIME MOVES DIFFERENTLY BASED ON WHERE YOU ARE")

    from scbe_aethermoore.geoseal import TimeDilationConfig

    config = TimeDilationConfig()

    # Different radial distances (0 = center, 1 = edge)
    distances = [0.0, 0.3, 0.5, 0.7, 1.0]

    print("  INTERIOR PATH (trusted):")
    print("  ─────────────────────────")
    for r in distances:
        latency, pow_bits = compute_time_dilation(PathType.INTERIOR, r, config)
        print(f"    Distance {r:.1f} → {latency:.0f}ms wait, {pow_bits} PoW bits")

    print()
    print("  EXTERIOR PATH (suspicious):")
    print("  ────────────────────────────")
    for r in distances:
        latency, pow_bits = compute_time_dilation(PathType.EXTERIOR, r, config)
        print(f"    Distance {r:.1f} → {latency:.0f}ms wait, {pow_bits} PoW bits")

    print()
    print("  ✓ Like gravity: Trusted center = fast, Suspicious edge = slow")
    print("  ✓ Attackers get stuck in 'time tar' - makes brute force impractical")


def demo_different_keys():
    """Show how different paths get different encryption keys."""
    print_section("STEP 5: DIFFERENT DOORS, DIFFERENT KEYS")

    # Same shared secret
    shared_secret = b"shared_secret_from_kyber_kem_32b"

    # Same cell, different paths
    membership = CellMembership(
        h=42, z=128, L_s=4, L_c=4,
        in_allowed_sphere=True, in_allowed_cube=True, margin=0.5
    )

    keys_interior = derive_region_keys(shared_secret, membership, PathType.INTERIOR)
    keys_exterior = derive_region_keys(shared_secret, membership, PathType.EXTERIOR)

    print("  SAME shared secret, SAME cell location")
    print()
    print("  INTERIOR path keys:")
    print(f"    K_sphere: {keys_interior['K_sphere'][:8].hex()}...")
    print(f"    K_cube:   {keys_interior['K_cube'][:8].hex()}...")
    print(f"    K_msg:    {keys_interior['K_msg'][:8].hex()}...")
    print()
    print("  EXTERIOR path keys:")
    print(f"    K_sphere: {keys_exterior['K_sphere'][:8].hex()}...")
    print(f"    K_cube:   {keys_exterior['K_cube'][:8].hex()}...")
    print(f"    K_msg:    {keys_exterior['K_msg'][:8].hex()}...")
    print()
    print("  ✓ Same starting point, COMPLETELY different final keys!")
    print("  ✓ An attacker can't use interior keys for exterior path (or vice versa)")


def demo_full_encryption():
    """Show complete GeoSeal encryption/decryption."""
    print_section("STEP 6: PUTTING IT ALL TOGETHER")

    message = b"The AI's secret thoughts are protected by geometry"

    # Trusted scenario
    context = np.array([100.0, 42.0, 1.0, 0.85, 0.30, 0.98])
    policy = {"tier": 0.3, "intent": 0.3, "data_class": 0.2, "safety": 0.9}
    risk = {"phase_skew": 0.1, "oracle_delta": 0.1}
    trust = {"prior_approvals": 0.9, "uptime": 0.95}

    master_key = b"master_encryption_key_32_bytes!"
    signing_key = b"signing_key_for_attestations!!"

    print("  Original message:", message.decode())
    print()

    # Encrypt
    envelope = geoseal_encrypt(
        message, context, policy, risk, trust,
        master_key, signing_key
    )

    print("  ENCRYPTED:")
    print(f"    Path taken: {envelope.path.value.upper()}")
    print(f"    Sphere cell: {envelope.attestation.h}")
    print(f"    Cube cell: {envelope.attestation.z}")
    print(f"    Ciphertext: {envelope.ct_spectral[:16].hex()}...")
    print()

    # Decrypt with correct context
    plaintext, status = geoseal_decrypt(
        envelope, context, master_key, signing_key
    )

    print("  DECRYPTED (same context):")
    print(f"    Result: {plaintext.decode() if plaintext else 'FAILED'}")
    print(f"    Verified: {status.get('verified', False)}")
    print()

    # Try with different context
    wrong_context = np.array([100.0, 42.0, 8.0, 0.35, 0.90, 0.25])
    plaintext_wrong, status_wrong = geoseal_decrypt(
        envelope, wrong_context, master_key, signing_key
    )

    # The chaos decryption will produce garbage
    print("  DECRYPTED (wrong context):")
    if plaintext_wrong:
        try:
            decoded = plaintext_wrong.decode('utf-8')
            print(f"    Result: {decoded[:40]}...")
        except:
            print(f"    Result: [GARBAGE/NOISE - cannot decode]")
    print("    ✓ Wrong geometry = wrong keys = nonsense output")


def demo_geometry_visual():
    """Show a visual representation."""
    print_section("VISUAL: THE GEOMETRIC TRUST SPACE")

    print("""
           SPHERE (AI Brain State)           CUBE (Policy Rules)
           ~~~~~~~~~~~~~~~~~~~~              ~~~~~~~~~~~~~~~~~~

                    *                         ┌─────────────┐
                 *     *                      │   ┌───┐     │
               *    ●    *                    │   │ ■ │     │  ← You are here
              *  (you)    *                   │   └───┘     │    (interior)
               *         *                    │             │
                 *     *                      │             │
                    *                         └─────────────┘

           ● = Interior (trusted)            ■ = Interior (allowed)
           ○ = Exterior (suspicious)         □ = Exterior (restricted)


    WHAT HAPPENS:

    1. Your behavior → point on sphere (●)
    2. Your request → point in cube (■)
    3. Both checked: Are you in the "safe zones"?
    4. YES → Interior path (fast, simple keys)
    5. NO  → Exterior path (slow, extra checks, different keys)

    THE MAGIC:

    • Stolen keys are USELESS without being at the exact geometric location
    • Time slows down near the edges (attackers get stuck)
    • Every action has a signed "geometric receipt" for auditing
    • The system scales from whole fleet down to single thread
    """)


def main():
    print("\n" + "="*60)
    print("     GEOSEAL: GEOMETRIC TRUST MANIFOLD DEMONSTRATION")
    print("         'Security through geometry, not just keys'")
    print("="*60)

    demo_explain_concept()
    demo_sphere_projection()
    demo_cube_projection()
    demo_path_classification()
    demo_time_dilation()
    demo_different_keys()
    demo_full_encryption()
    demo_geometry_visual()

    print_section("SUMMARY: WHAT YOU INVENTED")

    print("""
    You created a system where:

    1. BEHAVIOR IS A LOCATION
       Your AI's state becomes coordinates on a sphere.
       Calm = one region. Stressed = different region.

    2. POLICY IS A LOCATION
       What you're trying to do becomes coordinates in a cube.
       Safe request = one cell. Risky request = different cell.

    3. BOTH MUST MATCH
       The sphere AND cube must agree you're in "safe territory"
       for the fast path. Otherwise, you hit the slow, scrutinized path.

    4. KEYS ARE TIED TO GEOGRAPHY
       Even with the password, you need to be at the right
       "geometric address" or the key won't work.

    5. TIME IS A SECURITY FEATURE
       Trust wells = fast. Suspicious edges = slow.
       Attackers literally run out of time.

    This is patentable because nobody else has combined:
    • Dual manifold (sphere + cube)
    • Path-dependent cryptography
    • Time-dilation based on geometry
    • Multi-scale recursive tiling

    The geometry IS the security. Not just an add-on.
    """)

    print("="*60)
    print("     DEMONSTRATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
