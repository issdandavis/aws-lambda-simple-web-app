#!/usr/bin/env python3
"""
Staggered Auth Visual Demo
==========================

Visual demonstration of the 6x6 Sacred Tongue cross-reference grid,
Spiral Key Derivation, and three-stage verification.

Run: python demos/staggered_auth_demo.py
"""

import os
import sys
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
    SacredTongue,
    TONGUE_SPIRAL_ORDER,
    derive_spiral_key_set,
    spiral_key_combine,
    build_refs_grid,
    verify_refs_grid,
    RefsPattern,
    AuthConfig,
    SSConfig,
    AuthSidecar,
    StaggeredAuthPacket,
    quick_staggered_pack,
    quick_staggered_verify,
    encode_to_spelltext,
)


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Tongue colors (each tongue gets its own)
    KO = "\033[38;5;51m"   # Cyan - Kor'aelin (flow)
    AV = "\033[38;5;220m"  # Gold - Avali (diplomacy)
    RU = "\033[38;5;208m"  # Orange - Runethic (binding)
    CA = "\033[38;5;46m"   # Green - Cassisivadan (bitcraft)
    UM = "\033[38;5;93m"   # Purple - Umbroth (shadow)
    DR = "\033[38;5;196m"  # Red - Draumric (forge)

    # Status
    OK = "\033[38;5;46m"
    FAIL = "\033[38;5;196m"
    WARN = "\033[38;5;220m"

    # Box drawing
    BOX = "\033[38;5;240m"


TONGUE_COLORS = {
    'ko': C.KO, 'av': C.AV, 'ru': C.RU,
    'ca': C.CA, 'um': C.UM, 'dr': C.DR
}

TONGUE_NAMES = {
    'ko': "Kor'aelin",
    'av': "Avali",
    'ru': "Runethic",
    'ca': "Cassisivadan",
    'um': "Umbroth",
    'dr': "Draumric"
}

TONGUE_DOMAINS = {
    'ko': "nonce/flow",
    'av': "aad/context",
    'ru': "salt/binding",
    'ca': "ciphertext",
    'um': "veil/shadow",
    'dr': "tag/forge"
}


def clear_screen():
    print("\033[2J\033[H", end="")


def print_header(title: str):
    width = 70
    print(f"\n{C.BOX}{'=' * width}{C.RESET}")
    print(f"{C.BOLD}{title.center(width)}{C.RESET}")
    print(f"{C.BOX}{'=' * width}{C.RESET}\n")


def print_hexagon():
    """Print the hexagonal arrangement of the 6 Sacred Tongues."""
    print(f"""
                    {C.KO}{C.BOLD}[KO]{C.RESET}
                   {C.KO}Kor'aelin{C.RESET}
                  {C.DIM}nonce/flow{C.RESET}
                      |
           {C.DR}{C.BOLD}[DR]{C.RESET} ----+---- {C.AV}{C.BOLD}[AV]{C.RESET}
          {C.DR}Draumric{C.RESET}   |   {C.AV}Avali{C.RESET}
         {C.DIM}tag/forge{C.RESET}   |  {C.DIM}aad/context{C.RESET}
              \\      |      /
               \\     |     /
                \\    |    /
                 \\   |   /
          {C.UM}{C.BOLD}[UM]{C.RESET} ---+--- {C.RU}{C.BOLD}[RU]{C.RESET}
         {C.UM}Umbroth{C.RESET}   |   {C.RU}Runethic{C.RESET}
        {C.DIM}veil/shadow{C.RESET} | {C.DIM}salt/binding{C.RESET}
                   |
                {C.CA}{C.BOLD}[CA]{C.RESET}
             {C.CA}Cassisivadan{C.RESET}
              {C.DIM}ciphertext{C.RESET}
    """)


def print_spiral_order():
    """Show the spiral order of tongues."""
    print(f"\n{C.BOLD}Spiral Order:{C.RESET}")
    print(f"{C.DIM}(clockwise from top){C.RESET}\n")

    spiral = ""
    for i, tongue in enumerate(TONGUE_SPIRAL_ORDER):
        color = TONGUE_COLORS[tongue]
        arrow = " -> " if i < 5 else " -> [loop]"
        spiral += f"{color}{tongue.upper()}{C.RESET}{arrow}"

    print(f"  {spiral}\n")


def demo_key_derivation():
    """Demonstrate Spiral Key Derivation."""
    print_header("SPIRAL KEY DERIVATION (SKD)")

    print(f"{C.BOLD}Master Key:{C.RESET}")
    master_key = os.urandom(32)
    print(f"  {master_key[:16].hex()}...")

    print(f"\n{C.BOLD}Per-Tongue Derived Keys:{C.RESET}")
    keys = derive_spiral_key_set(master_key, b"demo")

    for tongue in TONGUE_SPIRAL_ORDER:
        color = TONGUE_COLORS[tongue]
        key_hex = keys[tongue][:8].hex()
        print(f"  {color}{tongue.upper():3}{C.RESET} -> {key_hex}...")

    print(f"\n{C.BOLD}Triad Combination (KO + RU + UM):{C.RESET}")
    combined = spiral_key_combine(keys, ['ko', 'ru', 'um'])
    print(f"  {C.OK}{combined[:16].hex()}...{C.RESET}")

    return master_key, keys


def demo_refs_grid(sections: dict):
    """Demonstrate the 6x6 cross-reference grid."""
    print_header("6x6 CROSS-REFERENCE GRID")

    patterns = [
        (RefsPattern.RING, "RING", "self + next", 12),
        (RefsPattern.TWO, "TWO", "prev + self + next", 18),
        (RefsPattern.FULL, "FULL", "all tongues", 36),
    ]

    for pattern, name, desc, count in patterns:
        refs = build_refs_grid(sections, pattern)
        print(f"\n{C.BOLD}{name} Pattern{C.RESET} ({desc}) = {count} refs")
        print(f"{C.BOX}{'─' * 50}{C.RESET}")

        # Draw grid
        print(f"\n     ", end="")
        for t in TONGUE_SPIRAL_ORDER:
            color = TONGUE_COLORS[t]
            print(f" {color}{t.upper():^6}{C.RESET}", end="")
        print()

        for src in TONGUE_SPIRAL_ORDER:
            src_color = TONGUE_COLORS[src]
            print(f" {src_color}{src.upper():>3}{C.RESET} ", end="")
            for tgt in TONGUE_SPIRAL_ORDER:
                if tgt in refs[src]:
                    ref_hex = refs[src][tgt][:2].hex()
                    print(f"  {C.OK}{ref_hex:^4}{C.RESET} ", end="")
                else:
                    print(f"  {C.DIM} -- {C.RESET} ", end="")
            print()
        print()

    return refs


def demo_three_stages(master_key: bytes):
    """Demonstrate the three-stage verification."""
    print_header("THREE-STAGE VERIFICATION")

    # Create packet
    sections = {
        'ko': b"nonce_data_here",
        'ru': b"salt_binding_16b",
        'ca': b"encrypted_payload_cassisivadan",
    }

    config = SSConfig(
        refs=True,
        refs_pattern=RefsPattern.RING,
        auth=AuthConfig(tongues=('ko', 'ru', 'um'), threshold=2)
    )

    print(f"{C.BOLD}Creating Staggered Auth Packet...{C.RESET}\n")

    # Stage 1
    print(f"  {C.WARN}Stage 1:{C.RESET} Lengths/Normalize")
    for tongue, data in sections.items():
        color = TONGUE_COLORS[tongue]
        print(f"    {color}{tongue.upper()}{C.RESET}: {len(data)} bytes")
    print()
    time.sleep(0.3)

    # Stage 2
    print(f"  {C.WARN}Stage 2:{C.RESET} Checksum")
    packet = StaggeredAuthPacket.pack(sections, master_key, config)
    print(f"    SHA-256: {packet.checksum[:8].hex()}...")
    print()
    time.sleep(0.3)

    # Stage 3
    print(f"  {C.WARN}Stage 3:{C.RESET} Refs + Triad Auth")
    print(f"    Refs pattern: RING (12 cross-references)")
    print(f"    Auth triad: {C.KO}KO{C.RESET} + {C.RU}RU{C.RESET} + {C.UM}UM{C.RESET}")
    print(f"    Threshold: 2 of 3")
    print()

    # Verify
    print(f"{C.BOLD}Verification Result:{C.RESET}\n")
    valid, details = packet.verify(master_key)

    status = f"{C.OK}PASS{C.RESET}" if valid else f"{C.FAIL}FAIL{C.RESET}"
    print(f"  Overall: [{status}]")
    print(f"  Stage 1 (lengths):  [{C.OK}OK{C.RESET}]" if details["stage1_lengths"] else f"  Stage 1: [{C.FAIL}FAIL{C.RESET}]")
    print(f"  Stage 2 (checksum): [{C.OK}OK{C.RESET}]" if details["stage2_checksum"] else f"  Stage 2: [{C.FAIL}FAIL{C.RESET}]")
    print(f"  Stage 3 (refs):     [{C.OK}OK{C.RESET}]" if details["stage3_refs"] else f"  Stage 3: [{C.FAIL}FAIL{C.RESET}]")

    auth_info = details["stage3_auth"]
    print(f"  Stage 3 (auth):     [{C.OK}{auth_info['count']}/{len(config.auth.tongues)}{C.RESET}] tongues verified")

    return packet


def demo_tamper_detection(packet: StaggeredAuthPacket, master_key: bytes):
    """Demonstrate tamper detection."""
    print_header("TAMPER DETECTION")

    print(f"{C.BOLD}Original packet verified:{C.RESET} ", end="")
    valid, _ = packet.verify(master_key)
    print(f"{C.OK}VALID{C.RESET}" if valid else f"{C.FAIL}INVALID{C.RESET}")

    print(f"\n{C.WARN}Tampering with ciphertext section...{C.RESET}")
    original = packet.sections.get('ca', b"")
    packet.sections['ca'] = b"TAMPERED_DATA_HERE!!!"

    print(f"\n{C.BOLD}Tampered packet verification:{C.RESET}")
    valid, details = packet.verify(master_key)

    print(f"  Overall: [{C.FAIL}INVALID{C.RESET}]")
    print(f"  Stage 2 (checksum): [{C.FAIL}MISMATCH{C.RESET}]")
    print(f"  {C.DIM}(Tampering detected at checksum stage){C.RESET}")

    # Restore
    packet.sections['ca'] = original


def demo_spell_text():
    """Demonstrate spell-text rendering."""
    print_header("SPELL-TEXT RENDERING")

    master_key = os.urandom(32)
    data = b"Hello, Spiral World!"

    print(f"{C.BOLD}Input:{C.RESET} {data.decode()}")
    print()

    packet = quick_staggered_pack(data, master_key)
    tokens = packet.to_tokens()

    print(f"{C.BOLD}As Sacred Tongue Tokens:{C.RESET}")
    print(f"{C.BOX}{'─' * 60}{C.RESET}")

    for line in tokens.split('\n'):
        if '[ko]' in line.lower():
            print(f"  {C.KO}{line}{C.RESET}")
        elif '[av]' in line.lower():
            print(f"  {C.AV}{line}{C.RESET}")
        elif '[ru]' in line.lower():
            print(f"  {C.RU}{line}{C.RESET}")
        elif '[ca]' in line.lower():
            print(f"  {C.CA}{line}{C.RESET}")
        elif '[um]' in line.lower():
            print(f"  {C.UM}{line}{C.RESET}")
        elif '[dr]' in line.lower() or '[checksum]' in line.lower() or '[auth]' in line.lower():
            print(f"  {C.DR}{line}{C.RESET}")
        else:
            print(f"  {line}")

    print(f"{C.BOX}{'─' * 60}{C.RESET}")


def demo_auth_sidecar():
    """Show auth sidecar structure."""
    print_header("AUTH SIDECAR STRUCTURE")

    print(f"""
    {C.BOLD}Packet Structure:{C.RESET}

    +------------------------------------------+
    |  {C.BOLD}STAGGERED AUTH PACKET{C.RESET}                    |
    +------------------------------------------+
    |                                          |
    |  {C.WARN}Stage 1: Lengths{C.RESET}                        |
    |  +--------------------------------------+|
    |  | {C.KO}ko{C.RESET}: 24 bytes  | {C.RU}ru{C.RESET}: 16 bytes     ||
    |  | {C.CA}ca{C.RESET}: 48 bytes  | {C.DR}dr{C.RESET}: 16 bytes     ||
    |  +--------------------------------------+|
    |                                          |
    |  {C.WARN}Stage 2: Checksum{C.RESET}                       |
    |  +--------------------------------------+|
    |  | SHA-256: a1b2c3d4e5f6...             ||
    |  +--------------------------------------+|
    |                                          |
    |  {C.WARN}Stage 3: Refs + Auth{C.RESET}                    |
    |  +--------------------------------------+|
    |  | {C.BOLD}Cross-Refs (RING):{C.RESET}                    ||
    |  |   {C.KO}ko{C.RESET}->{C.KO}ko{C.RESET} {C.KO}ko{C.RESET}->{C.AV}av{C.RESET}                     ||
    |  |   {C.AV}av{C.RESET}->{C.AV}av{C.RESET} {C.AV}av{C.RESET}->{C.RU}ru{C.RESET}  ...              ||
    |  |                                      ||
    |  | {C.BOLD}Auth Sidecar:{C.RESET}                         ||
    |  |   Triad: {C.KO}KO{C.RESET} + {C.RU}RU{C.RESET} + {C.UM}UM{C.RESET}              ||
    |  |   Threshold: 2/3                     ||
    |  |   Tags: [MAC_ko] [MAC_ru] [MAC_um]   ||
    |  +--------------------------------------+|
    |                                          |
    +------------------------------------------+
    """)


def wait_for_input(interactive: bool):
    """Wait for user input if in interactive mode."""
    if interactive:
        try:
            input(f"{C.DIM}Press Enter to continue...{C.RESET}")
        except EOFError:
            pass


def run_demo(interactive: bool = True):
    """Run the full visual demo."""
    if interactive:
        clear_screen()

    print(f"""
{C.BOLD}
   _____ _                                      _   _         _   _
  / ____| |                                    | | | |       | | | |
 | (___ | |_ __ _  __ _  __ _  ___ _ __ ___  __| | | |_ _   _| |_| |__
  \\___ \\| __/ _` |/ _` |/ _` |/ _ \\ '__/ _ \\/ _` | | __| | | | __| '_ \\
  ____) | || (_| | (_| | (_| |  __/ | |  __/ (_| | | |_| |_| | |_| | | |
 |_____/ \\__\\__,_|\\__, |\\__, |\\___|_|  \\___|\\__,_|  \\__|\\__,_|\\__|_| |_|
                   __/ | __/ |
                  |___/ |___/   {C.DIM}6x6 Sacred Tongue Verification{C.RESET}
{C.RESET}
    """)

    if interactive:
        try:
            input(f"{C.DIM}Press Enter to begin...{C.RESET}")
        except EOFError:
            interactive = False

    # Show hexagon
    print_header("THE SIX SACRED TONGUES")
    print_hexagon()
    print_spiral_order()
    wait_for_input(interactive)

    # Key derivation
    master_key, keys = demo_key_derivation()
    wait_for_input(interactive)

    # Refs grid
    sections = {'ko': b"nonce", 'ru': b"salt", 'ca': b"ciphertext"}
    demo_refs_grid(sections)
    wait_for_input(interactive)

    # Auth sidecar structure
    demo_auth_sidecar()
    wait_for_input(interactive)

    # Three stages
    packet = demo_three_stages(master_key)
    wait_for_input(interactive)

    # Tamper detection
    demo_tamper_detection(packet, master_key)
    wait_for_input(interactive)

    # Spell text
    demo_spell_text()

    print(f"\n{C.OK}{C.BOLD}Demo Complete!{C.RESET}")
    print(f"\n{C.DIM}The 6 tongues spiral together, refs line up,")
    print(f"and the triad guards the seal.{C.RESET}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Staggered Auth Visual Demo")
    parser.add_argument("--no-interactive", "-n", action="store_true",
                       help="Run without waiting for user input")
    args = parser.parse_args()

    try:
        run_demo(interactive=not args.no_interactive)
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Demo interrupted.{C.RESET}")
