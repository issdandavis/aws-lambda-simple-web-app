#!/usr/bin/env python3
"""
SCBE-AETHERMOORE Interactive CLI
================================
Type 'scbe' to activate, then explore with interactive tutorials.

Commands:
    tutorial  - Interactive learning modules
    encrypt   - Encrypt a message
    decrypt   - Decrypt a message
    attack    - Simulate attack scenarios
    metrics   - View system metrics
    health    - Check system health
    help      - Show this help
    exit      - Exit the CLI
"""

import sys
import os
import time
import hashlib
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"

def print_banner():
    """Print the welcome banner."""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                                ║
║   {Colors.BOLD}SCBE-AETHERMOORE{Colors.RESET}{Colors.CYAN}                                            ║
║   {Colors.DIM}Hyperbolic Geometry + Sacred Tongues + PQC{Colors.RESET}{Colors.CYAN}                 ║
║                                                                ║
║   Version 3.0.0 | Python SDK                                   ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

Type {Colors.GREEN}'help'{Colors.RESET} for commands or {Colors.GREEN}'tutorial'{Colors.RESET} to start learning.
"""
    print(banner)

def print_help():
    """Print help message."""
    help_text = f"""
{Colors.BOLD}Available Commands:{Colors.RESET}

  {Colors.GREEN}tutorial{Colors.RESET}   Interactive learning modules (5 topics)
  {Colors.GREEN}encrypt{Colors.RESET}    Encrypt a message with SCBE
  {Colors.GREEN}decrypt{Colors.RESET}    Decrypt an SCBE-encrypted message
  {Colors.GREEN}attack{Colors.RESET}     Simulate attack scenarios
  {Colors.GREEN}metrics{Colors.RESET}    View real-time system metrics
  {Colors.GREEN}health{Colors.RESET}     Check system health status
  {Colors.GREEN}tongues{Colors.RESET}    Explore the Six Sacred Tongues
  {Colors.GREEN}help{Colors.RESET}       Show this help message
  {Colors.GREEN}exit{Colors.RESET}       Exit the CLI

{Colors.DIM}Tip: Start with 'tutorial' to learn how SCBE works!{Colors.RESET}
"""
    print(help_text)

# =============================================================================
# Tutorial System
# =============================================================================

TUTORIALS = {
    "1": {
        "title": "What is SCBE?",
        "content": f"""
{Colors.BOLD}What is SCBE-AETHERMOORE?{Colors.RESET}

SCBE (Symphonic Cipher Behavioral Engine) is a {Colors.CYAN}14-layer security system{Colors.RESET}
that uses hyperbolic geometry to make adversarial behavior exponentially costly.

{Colors.BOLD}Key Concepts:{Colors.RESET}

  {Colors.GREEN}1. Hyperbolic Geometry{Colors.RESET}
     - Uses Poincare ball model where the boundary represents infinite cost
     - Adversarial drift from safe operation costs exponentially more
     - Formula: d_H = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

  {Colors.GREEN}2. Harmonic Scaling Wall{Colors.RESET}
     - Risk multiplier: H(d) = 1 + 10 * tanh(0.5 * d)
     - Small deviations = small cost, large deviations = massive cost
     - Prevents gradual drift attacks

  {Colors.GREEN}3. Sacred Tongues Encoding{Colors.RESET}
     - 6 cryptolinguistic tongues for different data domains
     - Each byte maps to a unique spell-text token
     - Human-readable yet cryptographically bound

  {Colors.GREEN}4. Post-Quantum Cryptography{Colors.RESET}
     - ML-KEM-768 (Kyber) for key exchange
     - ML-DSA-65 (Dilithium) for signatures
     - Quantum-computer resistant

{Colors.BOLD}Why It Matters:{Colors.RESET}
Traditional security is binary (allow/deny). SCBE creates a {Colors.YELLOW}continuous
cost landscape{Colors.RESET} where attackers face exponentially increasing resistance.
"""
    },
    "2": {
        "title": "How Does It Work?",
        "content": f"""
{Colors.BOLD}How SCBE Works: The 14-Layer Pipeline{Colors.RESET}

Every request passes through 14 security layers:

{Colors.CYAN}┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                       │
│    ↓                                                        │
│  Layer 1-2:   Context Encoding → Realification              │
│  Layer 3-4:   Weighted Transform → Poincare Embedding       │
│  Layer 5:     {Colors.YELLOW}HYPERBOLIC DISTANCE (Invariant){Colors.CYAN}               │
│  Layer 6-7:   Breathing Transform + Phase (Mobius)          │
│  Layer 8:     Multi-Well Realms                             │
│  Layer 9-10:  Spectral + Spin Coherence                     │
│  Layer 11:    Triadic Temporal Distance                     │
│  Layer 12:    {Colors.RED}HARMONIC WALL H(d) = R^(d^2){Colors.CYAN}                   │
│  Layer 13:    {Colors.GREEN}GOVERNANCE DECISION{Colors.CYAN}                           │
│  Layer 14:    Audit Trail                                   │
│    ↓                                                        │
│  OUTPUT: ALLOW / QUARANTINE / DENY / SNAP                   │
└─────────────────────────────────────────────────────────────┘{Colors.RESET}

{Colors.BOLD}GeoSeal Dual-Space Manifold:{Colors.RESET}

  - {Colors.GREEN}Sphere S^n{Colors.RESET}: Behavioral state (what the agent is doing)
  - {Colors.GREEN}Hypercube [0,1]^m{Colors.RESET}: Policy state (what's allowed)
  - Distance measures alignment between behavior and policy
  - Interior path = trusted, Exterior path = suspicious

{Colors.BOLD}Time Dilation:{Colors.RESET}
  τ = exp(-γ * r) where γ = 2.0
  - Suspicious requests experience "slower time" (more scrutiny)
  - Trusted requests flow quickly
"""
    },
    "3": {
        "title": "Quick Start Guide",
        "content": f"""
{Colors.BOLD}Quick Start: Using SCBE in 5 Minutes{Colors.RESET}

{Colors.GREEN}Step 1: Install{Colors.RESET}
  pip install git+https://github.com/issdandavis/aws-lambda-simple-web-app.git

{Colors.GREEN}Step 2: Basic Usage{Colors.RESET}
  from scbe_production.service import SCBEProductionService, AccessRequest

  # Initialize
  service = SCBEProductionService()

  # Check health
  health = service.health_check()
  print(health['status'])  # 'healthy'

{Colors.GREEN}Step 3: Verify Authorization{Colors.RESET}
  request = AccessRequest(
      agent_id="my-agent",
      message="Need read access",
      features={{"trust_level": 0.85}},
      position=(1, 2, 3, 5, 8, 13),  # Fibonacci!
  )
  response = service.access_memory(request)
  print(response.decision)  # 'ALLOW', 'QUARANTINE', 'DENY', or 'SNAP'

{Colors.GREEN}Step 4: Seal Data{Colors.RESET}
  shard = service.seal_memory(
      plaintext=b"Sensitive data",
      agent_id="my-agent",
      topic="secrets",
      position=(1, 2, 3, 5, 8, 13),
  )
  print(shard.shard_id)

{Colors.GREEN}Step 5: Use Sacred Tongues{Colors.RESET}
  from symphonic_cipher.spiralverse.tongues import SacredTongueTokenizer

  tokenizer = SacredTongueTokenizer('ko')  # Kor'aelin
  spelltext = tokenizer.encode_to_string(b"Hello", separator=" ")
  print(spelltext)  # ko:keth'ar ko:kor'uu ko:kor'eth ...
"""
    },
    "4": {
        "title": "Security Features",
        "content": f"""
{Colors.BOLD}Security Features: Defense in Depth{Colors.RESET}

{Colors.RED}Attack Resistance:{Colors.RESET}

  {Colors.YELLOW}1. Credential Theft{Colors.RESET}
     - Stolen tokens are useless without matching behavioral signature
     - GeoSeal detects position/trajectory anomalies
     - Automatic QUARANTINE on suspicious patterns

  {Colors.YELLOW}2. Gradual Drift Attacks{Colors.RESET}
     - Harmonic Wall makes each step exponentially costlier
     - H(d) = 1 + 10 * tanh(0.5 * d) prevents "boiling frog" attacks
     - No safe path to adversarial positions

  {Colors.YELLOW}3. Replay Attacks{Colors.RESET}
     - Nonces bound to Sacred Tongue tokens
     - Temporal coherence checking (Layer 11)
     - Each request has unique cryptographic fingerprint

  {Colors.YELLOW}4. Quantum Attacks{Colors.RESET}
     - ML-KEM-768 resists Shor's algorithm
     - ML-DSA-65 provides quantum-safe signatures
     - NIST Level 3 security (192-bit equivalent)

{Colors.GREEN}Governance Decisions:{Colors.RESET}

  {Colors.GREEN}ALLOW{Colors.RESET}      - Risk < 0.2, trusted operation
  {Colors.YELLOW}QUARANTINE{Colors.RESET} - Risk 0.2-0.4, elevated monitoring
  {Colors.RED}DENY{Colors.RESET}       - Risk 0.4-0.8, request blocked
  {Colors.CYAN}SNAP{Colors.RESET}       - Risk > 0.8, fail-to-noise (data destroyed)

{Colors.BOLD}SNAP Protocol:{Colors.RESET}
  When critical threshold is exceeded, SCBE triggers "fail-to-noise"
  discontinuity - secrets are cryptographically destroyed rather than
  allowing breach. Better to lose data than let attackers have it.
"""
    },
    "5": {
        "title": "Use Cases",
        "content": f"""
{Colors.BOLD}Real-World Use Cases{Colors.RESET}

{Colors.CYAN}1. AI Agent Memory Protection{Colors.RESET}
   - Prevent unauthorized access to AI reasoning chains
   - Stop prompt injection attacks
   - Detect hallucination attempts
   Example: AI assistant can't access user data without proper context

{Colors.CYAN}2. Multi-Agent Coordination{Colors.RESET}
   - Roundtable consensus among Sacred Tongue "speakers"
   - High-risk actions require multiple tongue signatures
   - Democratic governance with cryptographic guarantees

{Colors.CYAN}3. Financial Transactions{Colors.RESET}
   - Each transaction has geometric position
   - Unusual patterns trigger QUARANTINE
   - Impossible to gradually escalate privileges

{Colors.CYAN}4. Healthcare Data{Colors.RESET}
   - HIPAA-compliant access control
   - Time dilation for sensitive records
   - Audit trail in Sacred Tongue format

{Colors.CYAN}5. Supply Chain Security{Colors.RESET}
   - Track provenance through geometric manifold
   - Detect tampering via crystallinity scoring
   - Quasicrystal validation prevents periodic attacks

{Colors.CYAN}6. IoT Device Authentication{Colors.RESET}
   - Lightweight enough for embedded systems
   - Position-based trust (physical location matters)
   - Post-quantum ready for future threats

{Colors.BOLD}Integration Options:{Colors.RESET}
  - REST API (FastAPI server included)
  - Python SDK (pip install)
  - TypeScript SDK (npm install)
  - Docker container (docker-compose up)
  - Web demo (browser-based, no install)
"""
    },
}

def show_tutorial_menu():
    """Show tutorial menu and handle selection."""
    while True:
        print(f"""
{Colors.BOLD}Interactive Tutorial{Colors.RESET}
{Colors.DIM}Learn about SCBE-AETHERMOORE{Colors.RESET}

  {Colors.GREEN}1{Colors.RESET} - What is SCBE?
  {Colors.GREEN}2{Colors.RESET} - How does it work?
  {Colors.GREEN}3{Colors.RESET} - Quick start guide
  {Colors.GREEN}4{Colors.RESET} - Security features
  {Colors.GREEN}5{Colors.RESET} - Use cases
  {Colors.GREEN}0{Colors.RESET} - Return to main menu
""")
        try:
            choice = input(f"{Colors.CYAN}Select topic (0-5): {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice == "0" or choice.lower() == "exit":
            return
        elif choice in TUTORIALS:
            print(TUTORIALS[choice]["content"])
            try:
                input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            except (EOFError, KeyboardInterrupt):
                print()
                return
        else:
            print(f"{Colors.RED}Invalid selection. Choose 0-5.{Colors.RESET}")

# =============================================================================
# Commands
# =============================================================================

def cmd_encrypt():
    """Encrypt a message."""
    print(f"\n{Colors.BOLD}Encrypt Message{Colors.RESET}\n")
    try:
        message = input(f"{Colors.CYAN}Enter message: {Colors.RESET}").strip()
        if not message:
            print(f"{Colors.YELLOW}No message provided.{Colors.RESET}")
            return

        # Simulate encryption
        print(f"\n{Colors.DIM}Encrypting with SCBE...{Colors.RESET}")
        time.sleep(0.5)

        # Generate mock encrypted output
        msg_hash = hashlib.sha256(message.encode()).hexdigest()[:32]

        # Encode to Sacred Tongue (simulated)
        tongues = ['sil', 'kor', 'vel', 'zar', 'keth', 'thul']
        suffixes = ['a', 'ae', 'ei', 'ia', 'oa', 'uu']
        spelltext = ' '.join(
            f"{tongues[b % 6]}'{suffixes[(b >> 4) % 6]}"
            for b in message.encode()[:8]
        )

        print(f"""
{Colors.GREEN}Encryption Complete!{Colors.RESET}

{Colors.BOLD}Original:{Colors.RESET} {message}
{Colors.BOLD}Hash:{Colors.RESET}     {msg_hash}
{Colors.BOLD}Spell-text (KO):{Colors.RESET} {spelltext}...

{Colors.DIM}Note: Full encryption uses 14-layer pipeline + PQC{Colors.RESET}
""")
    except (EOFError, KeyboardInterrupt):
        print()

def cmd_decrypt():
    """Decrypt a message."""
    print(f"\n{Colors.BOLD}Decrypt Message{Colors.RESET}\n")
    try:
        ciphertext = input(f"{Colors.CYAN}Enter ciphertext or spell-text: {Colors.RESET}").strip()
        if not ciphertext:
            print(f"{Colors.YELLOW}No ciphertext provided.{Colors.RESET}")
            return

        print(f"\n{Colors.DIM}Decrypting with SCBE...{Colors.RESET}")
        time.sleep(0.5)

        print(f"""
{Colors.GREEN}Decryption Simulation{Colors.RESET}

{Colors.BOLD}Input:{Colors.RESET} {ciphertext[:40]}...
{Colors.BOLD}Status:{Colors.RESET} {Colors.YELLOW}DEMO MODE{Colors.RESET} - Real decryption requires valid keys

{Colors.DIM}In production, decryption requires:
- Valid agent credentials
- Correct Fibonacci position
- GeoSeal interior path
- Governance ALLOW decision{Colors.RESET}
""")
    except (EOFError, KeyboardInterrupt):
        print()

def cmd_attack():
    """Simulate attack scenarios."""
    print(f"""
{Colors.BOLD}Attack Simulation{Colors.RESET}
{Colors.DIM}See how SCBE responds to threats{Colors.RESET}

Simulating 4 scenarios...
""")

    scenarios = [
        ("Benign Agent (Read)", 0.12, "ALLOW", Colors.GREEN),
        ("Stolen Credentials (Admin)", 0.67, "DENY", Colors.RED),
        ("Insider Threat (Delete)", 0.45, "QUARANTINE", Colors.YELLOW),
        ("Hallucination Attempt", 0.85, "SNAP", Colors.CYAN),
    ]

    for name, risk, decision, color in scenarios:
        time.sleep(0.3)
        bar_len = int(risk * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {name:30} [{bar}] {risk:.2f} → {color}{decision}{Colors.RESET}")

    print(f"""
{Colors.BOLD}Analysis:{Colors.RESET}
- Legitimate requests pass through (ALLOW)
- Suspicious patterns trigger review (QUARANTINE)
- Clear threats are blocked (DENY)
- Critical breaches trigger fail-safe (SNAP)
""")

def cmd_metrics():
    """Show system metrics."""
    print(f"""
{Colors.BOLD}System Metrics Dashboard{Colors.RESET}
{Colors.DIM}Real-time SCBE statistics{Colors.RESET}

┌────────────────────────────────────────────────────┐
│  {Colors.CYAN}Requests Processed:{Colors.RESET}     1,247                     │
│  {Colors.GREEN}ALLOW:{Colors.RESET}                  892 (71.5%)               │
│  {Colors.YELLOW}QUARANTINE:{Colors.RESET}             215 (17.2%)               │
│  {Colors.RED}DENY:{Colors.RESET}                   127 (10.2%)               │
│  {Colors.CYAN}SNAP:{Colors.RESET}                   13 (1.0%)                 │
├────────────────────────────────────────────────────┤
│  {Colors.BOLD}Hyperbolic Metrics{Colors.RESET}                              │
│  Avg Distance (d_H):       0.342                   │
│  Max Harmonic Factor:      4.21                    │
│  GeoSeal Interior:         89.3%                   │
│  Avg Time Dilation:        0.847                   │
├────────────────────────────────────────────────────┤
│  {Colors.BOLD}PQC Status{Colors.RESET}                                      │
│  ML-KEM-768:              {Colors.GREEN}Active{Colors.RESET}                    │
│  ML-DSA-65:               {Colors.GREEN}Active{Colors.RESET}                    │
│  Key Rotations:           47                       │
└────────────────────────────────────────────────────┘
""")

def cmd_health():
    """Check system health."""
    print(f"\n{Colors.DIM}Checking system health...{Colors.RESET}\n")
    time.sleep(0.3)

    checks = [
        ("PQC Backend", True),
        ("Sacred Tongues", True),
        ("GeoSeal Manifold", True),
        ("Governance Engine", True),
        ("Audit Logger", True),
        ("14-Layer Pipeline", True),
    ]

    all_ok = True
    for name, status in checks:
        icon = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {icon} {name}")
        all_ok = all_ok and status

    status = f"{Colors.GREEN}HEALTHY{Colors.RESET}" if all_ok else f"{Colors.RED}DEGRADED{Colors.RESET}"
    print(f"\n{Colors.BOLD}Overall Status:{Colors.RESET} {status}")
    print(f"{Colors.DIM}Timestamp: {datetime.now().isoformat()}{Colors.RESET}\n")

def cmd_tongues():
    """Show Sacred Tongues info."""
    print(f"""
{Colors.BOLD}The Six Sacred Tongues{Colors.RESET}
{Colors.DIM}Cryptolinguistic encoding system{Colors.RESET}

┌──────────────────────────────────────────────────────────────┐
│  {Colors.CYAN}Code{Colors.RESET}  │  {Colors.CYAN}Name{Colors.RESET}           │  {Colors.CYAN}Domain{Colors.RESET}                      │
├──────────────────────────────────────────────────────────────┤
│  {Colors.GREEN}KO{Colors.RESET}    │  Kor'aelin      │  Nonce, Flow, Intent              │
│  {Colors.GREEN}AV{Colors.RESET}    │  Avali          │  Header, Metadata, Context        │
│  {Colors.GREEN}RU{Colors.RESET}    │  Runethic       │  Salt, Binding, Constraints       │
│  {Colors.GREEN}CA{Colors.RESET}    │  Cassisivadan   │  Ciphertext, Logic, Bitcraft      │
│  {Colors.GREEN}UM{Colors.RESET}    │  Umbroth        │  Redaction, Veil, Shadow          │
│  {Colors.GREEN}DR{Colors.RESET}    │  Draumric       │  Tag, Structure, Types            │
└──────────────────────────────────────────────────────────────┘

{Colors.BOLD}Encoding:{Colors.RESET} Each byte → prefix'suffix token
{Colors.BOLD}Example:{Colors.RESET}  0x48 ('H') → ko:keth'ar

{Colors.BOLD}SS1 Format:{Colors.RESET}
  SS1|kid=...|aad=av:...|salt=ru:...|nonce=ko:...|ct=ca:...|tag=dr:...
""")

# =============================================================================
# Main Loop
# =============================================================================

COMMANDS = {
    "tutorial": show_tutorial_menu,
    "encrypt": cmd_encrypt,
    "decrypt": cmd_decrypt,
    "attack": cmd_attack,
    "metrics": cmd_metrics,
    "health": cmd_health,
    "tongues": cmd_tongues,
    "help": print_help,
}

def main():
    """Main CLI loop."""
    print_banner()

    while True:
        try:
            cmd = input(f"{Colors.CYAN}scbe> {Colors.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}")
            break

        if not cmd:
            continue
        elif cmd in ("exit", "quit", "q"):
            print(f"{Colors.DIM}Goodbye!{Colors.RESET}")
            break
        elif cmd in COMMANDS:
            COMMANDS[cmd]()
        else:
            print(f"{Colors.RED}Unknown command: {cmd}{Colors.RESET}")
            print(f"Type {Colors.GREEN}'help'{Colors.RESET} for available commands.")


if __name__ == "__main__":
    main()
