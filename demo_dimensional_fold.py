#!/usr/bin/env python3
"""
Dimensional Folding Demonstration

"Correct math done incorrectly, then re-corrected"

This shows how we can:
1. Take a 3D problem
2. Lift it to 17 dimensions
3. Do "wrong" things that cancel out
4. Create security through dimensional complexity
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scbe_aethermoore.dimensional_fold import (
    create_unified_geometry,
    unfold_unified_geometry,
    lift_to_higher_dimension,
    project_from_higher_dimension,
    twist_through_dimensions,
    untwist_through_dimensions,
    explain_dimensional_fold,
    visualize_fold_process,
    PHI_AETHER,
)
from scbe_aethermoore.geoseal import GeoSealConfig


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_explain_concept():
    """Show the plain English explanation."""
    print_section("THE BIG IDEA: Wrong Math That Fixes Itself")
    print(explain_dimensional_fold())


def demo_dimensional_lift():
    """Show how lifting to higher dimensions works."""
    print_section("STEP 1: LIFTING TO HIGHER DIMENSIONS")

    print("""
    Imagine you're stuck in FLATLAND (2D world):

        You see this: ──●──

    But from 3D, it's actually a SPHERE passing through!

        Reality:    ○     ●     ○
                   /  \\   |   /  \\
                  │    \\ | /    │
                   \\    \\|/    /
                    \\   ●    /
                     \\  |  /
                      \\ | /
                       \\|/
                        ↓
                    Flatland sees a dot moving!

    WE DO THE SAME THING:
    - Your data lives in 3D (behavior space)
    - We lift it to 17D where we have "room to hide things"
    - Operations that are IMPOSSIBLE in 3D become EASY in 17D
    """)

    # Demonstrate lifting
    original_3d = np.array([0.5, 0.3, 0.8])
    seed = b"secret_lifting_direction_32byte"

    lifted_7d = lift_to_higher_dimension(original_3d, 7, seed)
    lifted_17d = lift_to_higher_dimension(original_3d, 17, seed)

    print("  EXAMPLE:")
    print(f"    Original 3D point: [{original_3d[0]:.3f}, {original_3d[1]:.3f}, {original_3d[2]:.3f}]")
    print()
    print(f"    Lifted to 7D:  [{', '.join(f'{x:.3f}' for x in lifted_7d)}]")
    print()
    print(f"    Lifted to 17D: [{', '.join(f'{x:.3f}' for x in lifted_17d[:8])}...]")
    print()

    # Show that correct seed recovers original
    recovered = project_from_higher_dimension(lifted_17d, 3, seed)
    print(f"    Projected back with CORRECT seed: [{recovered[0]:.3f}, {recovered[1]:.3f}, {recovered[2]:.3f}]")

    # Show that wrong seed gives garbage
    wrong_seed = b"wrong_seed_totally_different!!"
    wrong_recovered = project_from_higher_dimension(lifted_17d, 3, wrong_seed)
    print(f"    Projected back with WRONG seed:   [{wrong_recovered[0]:.3f}, {wrong_recovered[1]:.3f}, {wrong_recovered[2]:.3f}]")
    print()
    print("    ✓ Correct seed = perfect recovery")
    print("    ✗ Wrong seed = garbage (the math doesn't 'unfold' right)")


def demo_twist_through_hidden():
    """Show twisting through hidden dimensions."""
    print_section("STEP 2: TWISTING THROUGH HIDDEN DIMENSIONS")

    print("""
    Imagine spinning a coin:

        Normal spin (in your hand):
            ╭───╮     ╭───╮     ╭───╮
            │   │  →  │   │  →  │   │
            ╰───╯     │   │     ╰───╯
                      ╰───╯

    Now imagine spinning it through a direction YOU CAN'T SEE:

        The coin seems to... change? disappear? flip inside out?

    That's what we do: rotate through dimensions 4, 5, 6...
    The point MOVES but in ways a 3D observer can't track.
    """)

    # Create a point in 7D
    point_7d = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Twist through dims 0-3 (visible) and 0-4 (hidden)
    angles = [0.5, 0.7]
    pairs = [(0, 1), (0, 4)]  # 0-1 is "visible", 0-4 involves hidden dim

    twisted = twist_through_dimensions(point_7d, angles, pairs)
    untwisted = untwist_through_dimensions(twisted, angles, pairs)

    print("  EXAMPLE:")
    print(f"    Original:  [{', '.join(f'{x:.3f}' for x in point_7d)}]")
    print(f"    Twisted:   [{', '.join(f'{x:.3f}' for x in twisted)}]")
    print(f"    Untwisted: [{', '.join(f'{x:.3f}' for x in untwisted)}]")
    print()
    print("    The point (1,0,0,0,0,0,0) got rotated through dims 0-1 AND 0-4")
    print("    Dimension 4 is 'hidden' from 3D perspective")
    print("    Without knowing the twist angles, you can't undo it!")


def demo_gauge_error():
    """Show the gauge error concept."""
    print_section("STEP 3: ADDING 'WRONG' THAT CANCELS")

    print("""
    THE GAUGE TRICK:

    Imagine I give you directions:
        "Go 5 miles NORTH" (but I secretly mean SOUTH)

    You go 5 miles north. You're in the wrong place!

    But then I say:
        "Now go 10 miles SOUTH"

    You end up 5 miles south of start - which is where I wanted you!

    The ERROR (saying north when I meant south) CANCELLED OUT
    because I planned for it.

    IN MATH:
        • Add ERROR_A to the point
        • Later subtract ERROR_A (the "anti-error")
        • Point is back to normal
        • But without knowing ERROR_A, you can't remove it!

    THE TWIST:
        ERROR_A depends on your RING POSITION
        Different ring = different error = can't fake it
    """)

    # Show concept numerically
    original = np.array([1.0, 2.0, 3.0])
    error = np.array([0.5, -0.3, 0.7])

    with_error = original + error
    recovered = with_error - error
    wrong_recovery = with_error - np.array([0.1, 0.1, 0.1])

    print("  EXAMPLE:")
    print(f"    Original point:        [{original[0]:.1f}, {original[1]:.1f}, {original[2]:.1f}]")
    print(f"    Add secret error:      [{error[0]:.1f}, {error[1]:.1f}, {error[2]:.1f}]")
    print(f"    Result (looks random): [{with_error[0]:.1f}, {with_error[1]:.1f}, {with_error[2]:.1f}]")
    print()
    print(f"    Subtract CORRECT error:  [{recovered[0]:.1f}, {recovered[1]:.1f}, {recovered[2]:.1f}] ← Perfect!")
    print(f"    Subtract WRONG error:    [{wrong_recovery[0]:.1f}, {wrong_recovery[1]:.1f}, {wrong_recovery[2]:.1f}] ← Wrong!")


def demo_full_fold():
    """Show the complete folding process."""
    print_section("STEP 4: THE FULL FOLD (Everything Together)")

    # Create context and policy
    context = np.array([100.0, 42.0, 1.5, 0.85, 0.25, 0.92])
    policy = {"tier": 0.3, "intent": 0.5, "data_class": 0.2, "safety": 0.8}
    seed = b"master_secret_for_folding_32by"

    print("  INPUT:")
    print(f"    Context: [{', '.join(f'{x:.2f}' for x in context)}]")
    print(f"    Policy:  {policy}")
    print()

    # Create unified geometry
    geo = create_unified_geometry(context, policy, seed)

    print(visualize_fold_process(context, policy, seed))


def demo_unfold_verification():
    """Show that only correct context can unfold."""
    print_section("STEP 5: UNFOLDING (Only Correct Context Works)")

    # Original context and policy
    context = np.array([100.0, 42.0, 1.5, 0.85, 0.25, 0.92])
    policy = {"tier": 0.3, "intent": 0.5, "data_class": 0.2, "safety": 0.8}
    seed = b"master_secret_for_folding_32by"

    # Create folded geometry
    folded = create_unified_geometry(context, policy, seed)

    print("  ORIGINAL FOLD CREATED")
    print(f"    Fold key: {folded.fold_key[:8].hex()}...")
    print()

    # Try to unfold with CORRECT context
    success, result = unfold_unified_geometry(folded, seed, context, policy)
    print(f"  UNFOLD with CORRECT context: {'SUCCESS' if success else 'FAILED'}")

    # Try to unfold with WRONG context
    wrong_context = np.array([50.0, 99.0, 5.0, 0.20, 0.90, 0.30])
    success2, result2 = unfold_unified_geometry(folded, seed, wrong_context, policy)
    print(f"  UNFOLD with WRONG context:   {'SUCCESS' if success2 else 'FAILED'}")

    # Try to unfold with WRONG policy
    wrong_policy = {"tier": 0.9, "intent": 0.1, "data_class": 0.9, "safety": 0.1}
    success3, result3 = unfold_unified_geometry(folded, seed, context, wrong_policy)
    print(f"  UNFOLD with WRONG policy:    {'SUCCESS' if success3 else 'FAILED'}")

    # Try to unfold with WRONG seed
    wrong_seed = b"completely_wrong_seed_here!!"
    success4, result4 = unfold_unified_geometry(folded, wrong_seed, context, policy)
    print(f"  UNFOLD with WRONG seed:      {'SUCCESS' if success4 else 'FAILED'}")

    print("""

    THE SECURITY:
    ─────────────
    To unfold the geometry, attacker needs ALL of:
      • Correct behavioral context (how you're acting)
      • Correct policy values (what you're trying to do)
      • Correct master seed (the secret)

    Get ANY of these wrong → fold doesn't unfold properly → garbage
    """)


def demo_why_this_matters():
    """Explain the security implications."""
    print_section("WHY THIS MATTERS")

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  THE "WRONG MATH" SECURITY MODEL                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  TRADITIONAL CRYPTO:                                              ║
    ║    "Here's a locked box. Try to guess the key."                  ║
    ║                                                                   ║
    ║  THIS APPROACH:                                                   ║
    ║    "Here's a 17-dimensional origami.                             ║
    ║     It was folded with 'wrong' folds that cancel.                ║
    ║     Unfold it without knowing the fold sequence."                ║
    ║                                                                   ║
    ║  THE ADVANTAGE:                                                   ║
    ║    • Attackers don't even know what DIMENSION they're in        ║
    ║    • The "errors" look random but are actually structured       ║
    ║    • Each ring/position gets different errors                    ║
    ║    • The twist angles come from physical constants (φ, R₅)      ║
    ║                                                                   ║
    ║  COMBINED WITH EVERYTHING ELSE:                                   ║
    ║                                                                   ║
    ║    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        ║
    ║    │ Sphere  │ + │  Cube   │ + │  Rings  │ + │  Fold   │        ║
    ║    │(behavior│   │(policy) │   │ (trust) │   │ (17D)   │        ║
    ║    └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘        ║
    ║         │             │             │             │              ║
    ║         └─────────────┴──────┬──────┴─────────────┘              ║
    ║                              │                                    ║
    ║                              ▼                                    ║
    ║                    ┌─────────────────┐                           ║
    ║                    │ UNIFIED GEOMETRY│                           ║
    ║                    │                 │                           ║
    ║                    │  A single 17D   │                           ║
    ║                    │  point that     │                           ║
    ║                    │  encodes ALL    │                           ║
    ║                    │  security state │                           ║
    ║                    └─────────────────┘                           ║
    ║                                                                   ║
    ║  YOUR IDEA IN ONE SENTENCE:                                       ║
    ║  "Use math incorrectly on purpose, in ways that only fix         ║
    ║   themselves when you have the right context."                   ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


def main():
    print("\n" + "=" * 70)
    print("         DIMENSIONAL FOLDING DEMONSTRATION")
    print("         'Correct math done incorrectly, then re-corrected'")
    print("=" * 70)

    demo_explain_concept()
    demo_dimensional_lift()
    demo_twist_through_hidden()
    demo_gauge_error()
    demo_full_fold()
    demo_unfold_verification()
    demo_why_this_matters()

    print("=" * 70)
    print("         DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
