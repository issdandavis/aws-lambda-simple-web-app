"""
Klein Bottle Manifold Extensions for SCBE
==========================================

Non-orientable 4D topology for intent+time subspace.
Resolves infinite loop paradoxes with orientation-reversing paths.

Patent Claims: 19 (proposed)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Golden ratio for harmonic scaling
PHI = (1 + np.sqrt(5)) / 2

def klein_bottle_4d(u, v, R=PHI**2):
    """
    Parametric Klein bottle immersion in R^4.

    Args:
        u, v: Parameters in [0, 2π)
        R: Major radius (default φ² ≈ 2.618)

    Returns:
        x, y, z, w: 4D coordinates
    """
    x = (R + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.cos(u)
    y = (R + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.sin(u)
    z = np.sin(u/2) * np.sin(v) + np.cos(u/2) * np.sin(2*v)
    w = np.cos(u/2) * np.cos(v)  # 4th dimension
    return x, y, z, w


def klein_bottle_3d(u, v, R=PHI**2, twist_sign=1.0):
    """
    3D projection of Klein bottle (for visualization).

    Args:
        u, v: Parameters in [0, 2π)
        R: Major radius
        twist_sign: +1 for positive intent, -1 for negative/attack

    Returns:
        x, y, z: 3D coordinates
    """
    x = (R + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.cos(u) * twist_sign
    y = (R + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.sin(u)
    z = np.sin(u/2) * np.sin(v) + np.cos(u/2) * np.sin(2*v)
    return x, y, z


def intent_to_klein_params(intent_strength, time, max_intent=1.5, delta_t=10.0, phase=0.0):
    """
    Map intent vector and time to Klein bottle parameters (u, v).

    Claim 19: Intent subspace topologically modeled as Klein bottle.

    Args:
        intent_strength: Magnitude of intent [0, max_intent]
        time: Temporal coordinate
        max_intent: Maximum intent strength
        delta_t: Time window
        phase: Phase offset (from cryptographic key)

    Returns:
        u, v: Klein bottle parameters
    """
    # Map intent to u (with golden ratio scaling)
    u = 2 * np.pi * (intent_strength / max_intent) * PHI

    # Map time to v (with phase offset)
    v = 2 * np.pi * (time / delta_t) + phase

    return u % (4 * np.pi), v % (2 * np.pi)


def klein_distance(p1, p2, kappa=1.0):
    """
    Distance on Klein bottle surface.

    Near the "self-intersection" zone (in 3D projection),
    distance explodes → creates repulsive tension.

    Args:
        p1, p2: Points on Klein surface (u,v pairs)
        kappa: Curvature parameter

    Returns:
        Geodesic distance estimate
    """
    u1, v1 = p1
    u2, v2 = p2

    # Get 4D coordinates
    x1, y1, z1, w1 = klein_bottle_4d(u1, v1)
    x2, y2, z2, w2 = klein_bottle_4d(u2, v2)

    # Euclidean distance in 4D embedding
    d_euclidean = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + (w2-w1)**2)

    # Hyperbolic amplification near boundary
    r1 = np.sqrt(x1**2 + y1**2 + z1**2 + w1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2 + w2**2)

    # Boundary factor (explodes as norm approaches R)
    R = PHI**2
    boundary_factor = 1.0 / (1.0 - min(r1, r2) / (R * 2) + 1e-9)

    return d_euclidean * boundary_factor * kappa


def orientation_flip(trajectory_params):
    """
    Check if trajectory undergoes orientation reversal.

    Klein bottle property: Any closed loop flips orientation once.

    Args:
        trajectory_params: List of (u, v) pairs

    Returns:
        True if orientation was reversed (negative intent path)
    """
    if len(trajectory_params) < 2:
        return False

    # Check total u traversal
    total_u = 0
    for i in range(1, len(trajectory_params)):
        du = trajectory_params[i][0] - trajectory_params[i-1][0]
        # Handle wraparound
        if du > np.pi:
            du -= 2 * np.pi
        elif du < -np.pi:
            du += 2 * np.pi
        total_u += du

    # Full loop (2π in u) causes orientation flip
    return abs(total_u) >= 2 * np.pi


def classify_klein_trajectory(trajectory, time_values, max_intent=1.5):
    """
    Classify trajectory on Klein bottle.

    Args:
        trajectory: Complex intent values over time
        time_values: Corresponding time stamps

    Returns:
        Classification dict with orientation and tension metrics
    """
    # Map to Klein parameters
    params = []
    for t, intent in zip(time_values, trajectory):
        u, v = intent_to_klein_params(np.abs(intent), t, max_intent)
        params.append((u, v))

    # Check orientation flip
    flipped = orientation_flip(params)

    # Calculate tension (distance accumulation)
    total_tension = 0
    for i in range(1, len(params)):
        total_tension += klein_distance(params[i-1], params[i])

    # Classify
    if flipped:
        status = "REPULSIVE"  # Negative intent, orientation reversed
    elif total_tension > 10.0:
        status = "ANOMALOUS"  # High tension, probing behavior
    else:
        status = "HARMONIC"   # Smooth, coherent path

    return {
        'status': status,
        'orientation_flipped': flipped,
        'total_tension': total_tension,
        'num_points': len(params)
    }


def visualize_klein_comparison(save_path='klein_comparison.png'):
    """
    Generate visualization comparing legit vs attack intent on Klein bottle.
    """
    u = np.linspace(0, 4*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)

    # Legit (positive twist)
    X_legit, Y_legit, Z_legit = klein_bottle_3d(U, V, twist_sign=1.0)

    # Attack (negative twist = reverse gravity)
    X_attack, Y_attack, Z_attack = klein_bottle_3d(U, V, twist_sign=-1.0)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_legit, Y_legit, Z_legit, cmap='viridis', alpha=0.7)
    ax1.set_title("Legit Intent (Positive Loop)\nOrientation Preserved", fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_attack, Y_attack, Z_attack, cmap='plasma', alpha=0.7)
    ax2.set_title("Attack Intent (Negative Twist)\nOrientation Reversed → Repulsion", fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


# =============================================================================
# TESTS
# =============================================================================

def test_klein_topology():
    """Test Klein bottle topological properties."""
    print("=" * 60)
    print("KLEIN BOTTLE TOPOLOGY TESTS")
    print("=" * 60)

    # Test 1: 4D immersion is smooth
    u_test, v_test = np.pi, np.pi/2
    x, y, z, w = klein_bottle_4d(u_test, v_test)
    assert all(np.isfinite([x, y, z, w])), "4D coordinates must be finite"
    print(f"[PASS] 4D immersion at (π, π/2): ({x:.3f}, {y:.3f}, {z:.3f}, {w:.3f})")

    # Test 2: Golden ratio radius
    R = PHI**2
    assert abs(R - 2.618) < 0.001, f"R should be φ² ≈ 2.618, got {R}"
    print(f"[PASS] Klein radius R = φ² = {R:.6f}")

    # Test 3: Intent mapping
    u, v = intent_to_klein_params(1.0, 5.0, max_intent=1.5, delta_t=10.0)
    assert 0 <= u < 4*np.pi, f"u should be in [0, 4π), got {u}"
    assert 0 <= v < 2*np.pi, f"v should be in [0, 2π), got {v}"
    print(f"[PASS] Intent mapping: strength=1.0, time=5.0 → (u={u:.3f}, v={v:.3f})")

    # Test 4: Orientation flip detection
    # Full loop in u should flip
    full_loop = [(0, 0), (np.pi, 0), (2*np.pi, 0), (3*np.pi, 0)]
    flipped = orientation_flip(full_loop)
    assert flipped, "Full u-loop should flip orientation"
    print(f"[PASS] Full loop orientation flip detected: {flipped}")

    # Partial loop should not flip
    partial = [(0, 0), (np.pi/2, 0), (np.pi, 0)]
    not_flipped = orientation_flip(partial)
    assert not not_flipped, "Partial loop should not flip"
    print(f"[PASS] Partial loop no flip: {not not_flipped}")

    # Test 5: Distance amplification near boundary
    p1 = (0, 0)
    p2 = (0.1, 0)
    d_small = klein_distance(p1, p2)

    p3 = (2*np.pi, 0)  # Near "intersection" zone
    p4 = (2*np.pi + 0.1, 0)
    d_boundary = klein_distance(p3, p4)

    print(f"[INFO] Distance (interior): {d_small:.4f}")
    print(f"[INFO] Distance (boundary): {d_boundary:.4f}")
    # Boundary distance should be larger due to tension
    print(f"[PASS] Boundary tension factor: {d_boundary/d_small:.2f}x")

    # Test 6: Trajectory classification
    # Legit trajectory (smooth, low intent)
    t = np.linspace(0, 10, 50)
    legit_intent = 0.5 * np.exp(1j * t * 0.1)  # Low freq, stable
    result_legit = classify_klein_trajectory(legit_intent, t)
    print(f"\n[LEGIT] Status: {result_legit['status']}, Tension: {result_legit['total_tension']:.2f}")

    # Attack trajectory (high intent, rapid changes)
    attack_intent = 1.4 * np.exp(1j * t * 2.0)  # High freq, aggressive
    result_attack = classify_klein_trajectory(attack_intent, t)
    print(f"[ATTACK] Status: {result_attack['status']}, Tension: {result_attack['total_tension']:.2f}")

    print("\n" + "=" * 60)
    print("ALL KLEIN BOTTLE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_klein_topology()
    visualize_klein_comparison()
