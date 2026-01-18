#!/usr/bin/env python3
"""
Vertical Wall Visual Demo
=========================

Visual demonstration of the Harmonic Scaling Law H(d*) = exp(d*²)
and the Anti-Fragile Stiffness Psi(P) = 1 + (max-1) * tanh(beta * P)

The "vertical wall" makes attacks geometrically impossible as distance
from the trusted center increases.

Run: python demos/vertical_wall_demo.py
     python demos/vertical_wall_demo.py -n (non-interactive)
"""

import math
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Gradient for risk levels
    SAFE = "\033[38;5;46m"      # Green
    CAUTION = "\033[38;5;226m"  # Yellow
    DANGER = "\033[38;5;208m"   # Orange
    CRITICAL = "\033[38;5;196m" # Red

    # Special
    CYAN = "\033[38;5;51m"
    PURPLE = "\033[38;5;93m"
    GOLD = "\033[38;5;220m"
    BOX = "\033[38;5;240m"


def harmonic_scaling(d_star: float) -> float:
    """H(d*) = exp(d*²) - The Vertical Wall"""
    return math.exp(d_star ** 2)


def anti_fragile_stiffness(pressure: float, psi_max: float = 2.0, beta: float = 3.0) -> float:
    """Psi(P) = 1 + (psi_max - 1) * tanh(beta * P)"""
    return 1.0 + (psi_max - 1.0) * math.tanh(beta * pressure)


def print_header(title: str):
    width = 70
    print(f"\n{C.BOX}{'=' * width}{C.RESET}")
    print(f"{C.BOLD}{title.center(width)}{C.RESET}")
    print(f"{C.BOX}{'=' * width}{C.RESET}\n")


def get_risk_color(h_value: float) -> str:
    """Get color based on risk level."""
    if h_value < 3:
        return C.SAFE
    elif h_value < 20:
        return C.CAUTION
    elif h_value < 500:
        return C.DANGER
    else:
        return C.CRITICAL


def demo_vertical_wall():
    """Visualize the vertical wall H(d*) = exp(d*²)."""
    print_header("THE VERTICAL WALL: H(d*) = exp(d*²)")

    print(f"""
    {C.BOLD}The Harmonic Scaling Function{C.RESET}

    Risk amplification grows {C.CRITICAL}EXPONENTIALLY{C.RESET} as distance
    from the trusted center increases.

    At the boundary (d* → ∞), risk becomes {C.CRITICAL}INFINITE{C.RESET}.

    """)

    # Distance vs Risk table
    print(f"  {C.BOLD}d* (distance)     H(d*) (risk)      Visual{C.RESET}")
    print(f"  {C.BOX}{'─' * 60}{C.RESET}")

    distances = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    max_bar = 50

    for d in distances:
        h = harmonic_scaling(d)
        color = get_risk_color(h)

        # Logarithmic bar scaling
        if h > 1:
            bar_len = min(max_bar, int(math.log10(h) * 12))
        else:
            bar_len = 1

        bar = "█" * bar_len

        # Status
        if d < 1.0:
            status = "SAFE"
        elif d < 2.0:
            status = "CAUTION"
        elif d < 3.0:
            status = "DANGER"
        else:
            status = "IMPOSSIBLE"

        print(f"  {d:>5.1f}            {color}{h:>14,.2f}{C.RESET}    {color}{bar}{C.RESET} {status}")

    print(f"\n  {C.DIM}Risk EXPLODES near the boundary - attacks become impossible!{C.RESET}")


def demo_vertical_wall_ascii():
    """ASCII art visualization of the wall."""
    print_header("VISUAL: THE WALL")

    # Create ASCII representation of exp(x²)
    height = 20
    width = 60

    print(f"  {C.BOLD}H(d*){C.RESET}")
    print(f"    ▲")

    # Calculate points
    points = []
    for col in range(width):
        d_star = col / (width - 1) * 4  # d* from 0 to 4
        h = harmonic_scaling(d_star)
        log_h = math.log10(h + 1) if h > 0 else 0
        row = int(log_h * 3)  # Scale for display
        points.append((col, min(row, height - 1)))

    # Draw from top to bottom
    for row in range(height - 1, -1, -1):
        line = "    │"
        for col in range(width):
            _, point_row = points[col]
            if point_row >= row:
                # Color based on position
                d_star = col / (width - 1) * 4
                h = harmonic_scaling(d_star)
                color = get_risk_color(h)
                line += f"{color}█{C.RESET}"
            else:
                line += " "

        # Y-axis labels
        if row == height - 1:
            line += f" ← {C.CRITICAL}∞ (IMPOSSIBLE){C.RESET}"
        elif row == height // 2:
            line += f" ← {C.DANGER}~55 (DANGER){C.RESET}"
        elif row == 2:
            line += f" ← {C.SAFE}~1 (SAFE){C.RESET}"

        print(line)

    # X-axis
    print(f"    └{'─' * width}▶ d*")
    print(f"     0           1           2           3           4")
    print(f"     {C.SAFE}[SAFE]{C.RESET}      {C.CAUTION}[CAUTION]{C.RESET}  {C.DANGER}[DANGER]{C.RESET}   {C.CRITICAL}[WALL]{C.RESET}")


def demo_anti_fragile():
    """Visualize anti-fragile stiffness."""
    print_header("ANTI-FRAGILE STIFFNESS: Ψ(P)")

    print(f"""
    {C.BOLD}The System Gets STRONGER Under Attack{C.RESET}

    Like a non-Newtonian fluid:
    - Walk slowly → feet sink in
    - Run fast → surface becomes {C.CYAN}SOLID{C.RESET}

    Ψ(P) = 1 + (Ψ_max - 1) × tanh(β × P)

    """)

    print(f"  {C.BOLD}P (attack)   Ψ (stiffness)   Response{C.RESET}")
    print(f"  {C.BOX}{'─' * 55}{C.RESET}")

    pressures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for p in pressures:
        psi = anti_fragile_stiffness(p)
        bar_len = int((psi - 1.0) * 40)
        bar = "█" * bar_len

        if p < 0.3:
            color = C.SAFE
            status = "Normal"
        elif p < 0.6:
            color = C.CAUTION
            status = "Hardening"
        elif p < 0.8:
            color = C.DANGER
            status = "Reinforced"
        else:
            color = C.CYAN
            status = "MAXIMUM"

        print(f"  {p:>4.1f}         {color}{psi:.4f}{C.RESET}        {color}{bar}{C.RESET} {status}")

    print(f"\n  {C.DIM}Attack harder → System becomes STRONGER!{C.RESET}")


def demo_breathing_space():
    """Visualize the breathing Poincaré ball."""
    print_header("BREATHING SPACE")

    print(f"""
    {C.BOLD}The Poincaré Ball "Breathes" Based on Threat{C.RESET}

    • {C.SAFE}b < 1{C.RESET}: Space contracts (low threat, easier travel)
    • b = 1: Identity (no change)
    • {C.CRITICAL}b > 1{C.RESET}: Space expands (high threat, harder travel)

    """)

    # ASCII Poincaré disk
    radius = 8

    for breath_label, breath_val, color in [("LOW THREAT", 0.7, C.SAFE),
                                             ("NORMAL", 1.0, C.CAUTION),
                                             ("HIGH THREAT", 1.5, C.CRITICAL)]:
        print(f"    {C.BOLD}{breath_label}{C.RESET} (b = {breath_val})")

        scaled_radius = int(radius * breath_val)

        for y in range(-scaled_radius, scaled_radius + 1):
            line = "    "
            for x in range(-scaled_radius * 2, scaled_radius * 2 + 1):
                dist = math.sqrt((x/2)**2 + y**2)
                if abs(dist - scaled_radius) < 0.8:
                    line += f"{color}○{C.RESET}"
                elif dist < scaled_radius:
                    if abs(x) < 1 and abs(y) < 1:
                        line += f"{C.GOLD}●{C.RESET}"  # Center
                    elif dist < scaled_radius * 0.3:
                        line += f"{C.SAFE}·{C.RESET}"
                    elif dist < scaled_radius * 0.6:
                        line += f"{C.CAUTION}·{C.RESET}"
                    else:
                        line += f"{color}·{C.RESET}"
                else:
                    line += " "
            print(line)
        print()


def demo_composite_attack():
    """Simulate an attack and show risk calculation."""
    print_header("COMPOSITE RISK SIMULATION")

    print(f"""
    {C.BOLD}Risk' = B × H(d*) × T × I{C.RESET}

    B = Base behavioral risk [0, 1]
    H(d*) = Harmonic scaling (distance)
    T = Time penalty factor ≥ 1
    I = Intent suspicion factor ≥ 1

    """)

    # Simulate attack scenarios
    scenarios = [
        ("Normal user", 0.1, 0.5, 1.0, 1.0),
        ("Curious user", 0.3, 1.0, 1.0, 1.2),
        ("Suspicious activity", 0.5, 1.5, 1.5, 1.5),
        ("Clear attack", 0.8, 2.0, 2.0, 2.0),
        ("Maximum threat", 1.0, 3.0, 3.0, 3.0),
    ]

    print(f"  {C.BOLD}Scenario              B      d*     T      I      Risk     Decision{C.RESET}")
    print(f"  {C.BOX}{'─' * 70}{C.RESET}")

    for name, b, d_star, t, intent in scenarios:
        h = harmonic_scaling(d_star)
        risk = b * h * t * intent

        if risk < 1.0:
            decision = f"{C.SAFE}ALLOW{C.RESET}"
        elif risk < 5.0:
            decision = f"{C.CAUTION}WARN{C.RESET}"
        elif risk < 50.0:
            decision = f"{C.DANGER}DENY{C.RESET}"
        else:
            decision = f"{C.CRITICAL}BLOCK{C.RESET}"

        color = get_risk_color(risk)
        print(f"  {name:<20}  {b:.1f}    {d_star:.1f}    {t:.1f}    {intent:.1f}    {color}{risk:>8.2f}{C.RESET}   {decision}")

    print(f"\n  {C.DIM}Any single bad factor can push risk over the threshold!{C.RESET}")


def demo_attack_path():
    """Visualize an attacker's path hitting the wall."""
    print_header("ATTACK PATH VISUALIZATION")

    print(f"""
    {C.BOLD}Attacker Attempting to Reach Protected Resource{C.RESET}

    Starting from outside, trying to reach the center...

    """)

    width = 60

    # Stages of attack
    stages = [
        (0.5, "Probing...", C.SAFE),
        (1.0, "Attempting access...", C.CAUTION),
        (1.5, "Pushing deeper...", C.CAUTION),
        (2.0, "Risk escalating!", C.DANGER),
        (2.5, "HITTING WALL!", C.CRITICAL),
        (3.0, "💥 REJECTED 💥", C.CRITICAL),
    ]

    for d_star, message, color in stages:
        h = harmonic_scaling(d_star)
        position = int((4 - d_star) / 4 * (width - 10))

        # Draw the path
        line = "  "
        line += f"{C.SAFE}●{C.RESET}"  # Protected resource
        line += f"{C.SAFE}━{C.RESET}" * (position - 1)

        if d_star < 2.5:
            line += f"{color}◆{C.RESET}"  # Attacker position
        else:
            line += f"{color}✖{C.RESET}"  # Blocked

        line += " " * (width - position - 10)
        line += f"{C.BOX}│{C.RESET}"  # Wall

        print(f"{line}  {message}")
        print(f"  {' ' * (position + 2)}d*={d_star:.1f}  H={h:,.0f}")
        print()

    print(f"  {C.SAFE}●{C.RESET} = Protected resource")
    print(f"  {C.BOX}│{C.RESET} = The Vertical Wall (d* ≈ 4)")
    print(f"\n  {C.DIM}The wall is {C.CRITICAL}MATHEMATICALLY IMPOSSIBLE{C.DIM} to cross!{C.RESET}")


def run_demo(interactive: bool = True):
    """Run the full visual demo."""
    print(f"""
{C.BOLD}
 ██╗   ██╗███████╗██████╗ ████████╗██╗ ██████╗ █████╗ ██╗
 ██║   ██║██╔════╝██╔══██╗╚══██╔══╝██║██╔════╝██╔══██╗██║
 ██║   ██║█████╗  ██████╔╝   ██║   ██║██║     ███████║██║
 ╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║   ██║██║     ██╔══██║██║
  ╚████╔╝ ███████╗██║  ██║   ██║   ██║╚██████╗██║  ██║███████╗
   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝
                    ██╗    ██╗ █████╗ ██╗     ██╗
                    ██║    ██║██╔══██╗██║     ██║
                    ██║ █╗ ██║███████║██║     ██║
                    ██║███╗██║██╔══██║██║     ██║
                    ╚███╔███╔╝██║  ██║███████╗███████╗
                     ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝╚══════╝
{C.RESET}
    {C.DIM}H(d*) = exp(d*²) - Making attacks geometrically impossible{C.RESET}
    """)

    def wait():
        if interactive:
            try:
                input(f"\n{C.DIM}Press Enter to continue...{C.RESET}")
            except EOFError:
                pass

    demo_vertical_wall()
    wait()

    demo_vertical_wall_ascii()
    wait()

    demo_anti_fragile()
    wait()

    demo_breathing_space()
    wait()

    demo_composite_attack()
    wait()

    demo_attack_path()

    print(f"\n{C.SAFE}{C.BOLD}Demo Complete!{C.RESET}")
    print(f"\n{C.DIM}The Vertical Wall stands eternal.")
    print(f"Attacks become mathematically impossible.{C.RESET}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vertical Wall Visual Demo")
    parser.add_argument("--no-interactive", "-n", action="store_true",
                       help="Run without waiting for user input")
    args = parser.parse_args()

    try:
        run_demo(interactive=not args.no_interactive)
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Demo interrupted.{C.RESET}")
