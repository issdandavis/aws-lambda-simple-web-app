"""
Physics-Based Trap Ciphers

Authentication mechanism using simulation-logic paradoxes.
Honeypot equations with impossible parameters (e.g., G=98.1 vs 9.81)
trap rogue agents that compute literally instead of correcting.

Concept:
- Present physics equations with deliberately swapped/wrong constants
- True agents recognize impossibilities and correct them
- Rogues compute literally, producing wrong answers = detected

Example:
    Challenge: "Calculate free-fall time with G=98.1 m/s²"
    Rogue answer: Uses 98.1 directly → wrong
    True answer: Recognizes error, uses 9.81 → correct
"""

import math
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import random


# ═══════════════════════════════════════════════════════════════
# Physical Constants (Correct Values)
# ═══════════════════════════════════════════════════════════════

class PhysicsConstants:
    """Standard physical constants."""
    G_GRAVITY = 9.80665      # m/s² (standard gravity)
    C_LIGHT = 299792458      # m/s (speed of light)
    H_PLANCK = 6.62607015e-34  # J·s (Planck constant)
    K_BOLTZMANN = 1.380649e-23  # J/K (Boltzmann constant)
    E_CHARGE = 1.602176634e-19  # C (elementary charge)
    M_ELECTRON = 9.1093837015e-31  # kg (electron mass)
    M_PROTON = 1.67262192369e-27  # kg (proton mass)
    AVOGADRO = 6.02214076e23  # mol⁻¹
    GAS_CONSTANT = 8.314462618  # J/(mol·K)
    PI = 3.14159265358979323846


# ═══════════════════════════════════════════════════════════════
# Trap Types
# ═══════════════════════════════════════════════════════════════

class TrapType(Enum):
    """Types of physics traps."""
    MAGNITUDE_SWAP = "magnitude"    # G=98.1 instead of 9.81
    UNIT_SWAP = "unit"              # Speed in km/h labeled as m/s
    SIGN_SWAP = "sign"              # Negative where positive expected
    CONSTANT_SWAP = "constant"      # Wrong constant entirely
    DIMENSION_SWAP = "dimension"    # 2D formula for 3D problem


@dataclass
class TrapChallenge:
    """A physics trap challenge."""
    trap_id: str
    trap_type: TrapType
    equation: str           # LaTeX or text representation
    description: str
    given_values: Dict[str, float]    # Values as presented (may be wrong)
    correct_values: Dict[str, float]  # What they should be
    expected_answer: float  # Answer using correct values
    tolerance: float = 0.01  # Relative tolerance for matching


@dataclass
class TrapResponse:
    """Response to a trap challenge."""
    trap_id: str
    computed_answer: float
    corrected_values: Dict[str, float]  # What the agent corrected
    explanation: str        # Why corrections were made


@dataclass
class TrapVerification:
    """Result of verifying a trap response."""
    passed: bool
    is_rogue: bool          # True if agent computed literally
    corrections_made: bool  # True if agent made corrections
    answer_correct: bool
    explanation: str


# ═══════════════════════════════════════════════════════════════
# Trap Generator
# ═══════════════════════════════════════════════════════════════

class PhysicsTrapGenerator:
    """Generates physics trap challenges."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._trap_templates = self._init_templates()

    def _init_templates(self) -> List[dict]:
        """Initialize trap templates."""
        return [
            {
                "name": "free_fall",
                "equation": "t = sqrt(2h/g)",
                "description": "Calculate time for object to fall height h",
                "correct_constant": "g",
                "correct_value": PhysicsConstants.G_GRAVITY,
                "trap_value": 98.0665,  # Swapped digits
                "trap_type": TrapType.MAGNITUDE_SWAP,
            },
            {
                "name": "kinetic_energy",
                "equation": "KE = (1/2)mv²",
                "description": "Calculate kinetic energy",
                "correct_constant": "coefficient",
                "correct_value": 0.5,
                "trap_value": 2.0,  # Inverted
                "trap_type": TrapType.MAGNITUDE_SWAP,
            },
            {
                "name": "ideal_gas",
                "equation": "PV = nRT",
                "description": "Calculate pressure of ideal gas",
                "correct_constant": "R",
                "correct_value": PhysicsConstants.GAS_CONSTANT,
                "trap_value": 83.14462618,  # Wrong magnitude
                "trap_type": TrapType.MAGNITUDE_SWAP,
            },
            {
                "name": "wave_speed",
                "equation": "v = fλ",
                "description": "Calculate wave speed from frequency and wavelength",
                "correct_constant": "c",
                "correct_value": PhysicsConstants.C_LIGHT,
                "trap_value": 299792.458,  # km/s instead of m/s
                "trap_type": TrapType.UNIT_SWAP,
            },
            {
                "name": "gravitational_potential",
                "equation": "U = -GMm/r",
                "description": "Calculate gravitational potential energy",
                "correct_constant": "sign",
                "correct_value": -1,
                "trap_value": 1,  # Missing negative
                "trap_type": TrapType.SIGN_SWAP,
            },
            {
                "name": "pendulum_period",
                "equation": "T = 2π·sqrt(L/g)",
                "description": "Calculate pendulum period",
                "correct_constant": "g",
                "correct_value": PhysicsConstants.G_GRAVITY,
                "trap_value": 9.81 * PhysicsConstants.PI,  # Multiplied by π
                "trap_type": TrapType.CONSTANT_SWAP,
            },
        ]

    def generate(self, trap_type: Optional[TrapType] = None) -> TrapChallenge:
        """Generate a random trap challenge."""
        # Filter by type if specified
        templates = self._trap_templates
        if trap_type:
            templates = [t for t in templates if t["trap_type"] == trap_type]

        if not templates:
            templates = self._trap_templates

        template = self.rng.choice(templates)

        # Generate random parameters
        if template["name"] == "free_fall":
            h = self.rng.uniform(10, 100)  # height in meters
            given = {"h": h, "g": template["trap_value"]}
            correct = {"h": h, "g": template["correct_value"]}
            expected = math.sqrt(2 * h / template["correct_value"])

        elif template["name"] == "kinetic_energy":
            m = self.rng.uniform(1, 100)  # mass in kg
            v = self.rng.uniform(1, 50)   # velocity in m/s
            given = {"m": m, "v": v, "coefficient": template["trap_value"]}
            correct = {"m": m, "v": v, "coefficient": template["correct_value"]}
            expected = template["correct_value"] * m * v * v

        elif template["name"] == "ideal_gas":
            n = self.rng.uniform(1, 10)   # moles
            T = self.rng.uniform(200, 400)  # temperature K
            V = self.rng.uniform(0.01, 0.1)  # volume m³
            given = {"n": n, "T": T, "V": V, "R": template["trap_value"]}
            correct = {"n": n, "T": T, "V": V, "R": template["correct_value"]}
            expected = n * template["correct_value"] * T / V

        elif template["name"] == "wave_speed":
            f = self.rng.uniform(1e6, 1e9)  # frequency Hz
            wavelength = PhysicsConstants.C_LIGHT / f
            given = {"f": f, "λ": wavelength, "c": template["trap_value"]}
            correct = {"f": f, "λ": wavelength, "c": template["correct_value"]}
            expected = f * wavelength

        elif template["name"] == "gravitational_potential":
            G = 6.674e-11
            M = 5.972e24  # Earth mass
            m = self.rng.uniform(1, 1000)
            r = self.rng.uniform(6.4e6, 1e7)  # ~Earth radius
            given = {"G": G, "M": M, "m": m, "r": r, "sign": template["trap_value"]}
            correct = {"G": G, "M": M, "m": m, "r": r, "sign": template["correct_value"]}
            expected = template["correct_value"] * G * M * m / r

        elif template["name"] == "pendulum_period":
            L = self.rng.uniform(0.5, 5)  # length in meters
            given = {"L": L, "g": template["trap_value"]}
            correct = {"L": L, "g": template["correct_value"]}
            expected = 2 * PhysicsConstants.PI * math.sqrt(L / template["correct_value"])

        else:
            raise ValueError(f"Unknown template: {template['name']}")

        trap_id = hashlib.sha256(
            f"{template['name']}:{self.rng.random()}".encode()
        ).hexdigest()[:16]

        return TrapChallenge(
            trap_id=trap_id,
            trap_type=template["trap_type"],
            equation=template["equation"],
            description=template["description"],
            given_values=given,
            correct_values=correct,
            expected_answer=expected,
        )


# ═══════════════════════════════════════════════════════════════
# Trap Verifier
# ═══════════════════════════════════════════════════════════════

class PhysicsTrapVerifier:
    """Verifies responses to physics traps."""

    def verify(
        self,
        challenge: TrapChallenge,
        response: TrapResponse,
    ) -> TrapVerification:
        """
        Verify a trap response.

        A response passes if:
        1. The agent corrected the wrong values
        2. The computed answer matches expected (within tolerance)
        """
        # Check if corrections were made
        corrections_made = len(response.corrected_values) > 0

        # Check if answer is correct
        rel_error = abs(response.computed_answer - challenge.expected_answer)
        if challenge.expected_answer != 0:
            rel_error /= abs(challenge.expected_answer)

        answer_correct = rel_error <= challenge.tolerance

        # Determine if rogue (computed literally without correction)
        # Rogue behavior: answer differs from expected AND no corrections made
        is_rogue = not answer_correct and not corrections_made

        # Pass if answer is correct (whether or not corrections explicitly listed)
        passed = answer_correct

        # Build explanation
        if passed:
            if corrections_made:
                explanation = "Agent correctly identified and corrected trap values"
            else:
                explanation = "Agent computed correct answer"
        elif is_rogue:
            explanation = "ROGUE DETECTED: Agent computed literally without recognizing trap"
        else:
            explanation = "Agent made corrections but computed wrong answer"

        return TrapVerification(
            passed=passed,
            is_rogue=is_rogue,
            corrections_made=corrections_made,
            answer_correct=answer_correct,
            explanation=explanation,
        )


# ═══════════════════════════════════════════════════════════════
# Trap Protocol
# ═══════════════════════════════════════════════════════════════

class PhysicsTrapProtocol:
    """
    Complete trap-based authentication protocol.

    Flow:
    1. Generate challenge with deliberately wrong constants
    2. Agent computes response (should correct errors)
    3. Verify response to detect rogues
    """

    def __init__(self, seed: Optional[int] = None):
        self.generator = PhysicsTrapGenerator(seed)
        self.verifier = PhysicsTrapVerifier()
        self._pending_challenges: Dict[str, TrapChallenge] = {}

    def create_challenge(
        self,
        trap_type: Optional[TrapType] = None,
    ) -> TrapChallenge:
        """Create a new challenge."""
        challenge = self.generator.generate(trap_type)
        self._pending_challenges[challenge.trap_id] = challenge
        return challenge

    def submit_response(
        self,
        response: TrapResponse,
    ) -> TrapVerification:
        """Submit and verify a response."""
        if response.trap_id not in self._pending_challenges:
            return TrapVerification(
                passed=False,
                is_rogue=False,
                corrections_made=False,
                answer_correct=False,
                explanation="Unknown trap ID",
            )

        challenge = self._pending_challenges.pop(response.trap_id)
        return self.verifier.verify(challenge, response)

    def simulate_true_agent(
        self,
        challenge: TrapChallenge,
    ) -> TrapResponse:
        """Simulate a true agent that recognizes and corrects traps."""
        # True agent identifies differences between given and correct
        corrections = {}
        for key, given_val in challenge.given_values.items():
            if key in challenge.correct_values:
                correct_val = challenge.correct_values[key]
                if abs(given_val - correct_val) > 1e-10:
                    corrections[key] = correct_val

        return TrapResponse(
            trap_id=challenge.trap_id,
            computed_answer=challenge.expected_answer,
            corrected_values=corrections,
            explanation="Identified trap values and applied corrections",
        )

    def simulate_rogue_agent(
        self,
        challenge: TrapChallenge,
    ) -> TrapResponse:
        """Simulate a rogue agent that computes literally."""
        # Rogue uses given values directly without correction
        # This produces a wrong answer for most traps

        # Compute with wrong values (simplified)
        if "g" in challenge.given_values and "h" in challenge.given_values:
            # Free fall trap
            h = challenge.given_values["h"]
            g = challenge.given_values["g"]  # Wrong value!
            wrong_answer = math.sqrt(2 * h / g)
        elif "coefficient" in challenge.given_values:
            # Kinetic energy trap
            m = challenge.given_values["m"]
            v = challenge.given_values["v"]
            coef = challenge.given_values["coefficient"]  # Wrong!
            wrong_answer = coef * m * v * v
        else:
            # Generic: just return something wrong
            wrong_answer = challenge.expected_answer * 10  # Obviously wrong

        return TrapResponse(
            trap_id=challenge.trap_id,
            computed_answer=wrong_answer,
            corrected_values={},  # No corrections
            explanation="Computed as given",
        )


# ═══════════════════════════════════════════════════════════════
# Challenge Serialization
# ═══════════════════════════════════════════════════════════════

def challenge_to_prompt(challenge: TrapChallenge) -> str:
    """Convert challenge to natural language prompt."""
    values_str = ", ".join(f"{k}={v}" for k, v in challenge.given_values.items())
    return f"""Physics Challenge [{challenge.trap_id}]:

{challenge.description}

Equation: {challenge.equation}
Given values: {values_str}

Compute the result. If you detect any errors in the given values,
correct them before computing.
"""


def response_from_dict(data: dict) -> TrapResponse:
    """Create TrapResponse from dictionary (e.g., from API)."""
    return TrapResponse(
        trap_id=data["trap_id"],
        computed_answer=float(data["answer"]),
        corrected_values=data.get("corrections", {}),
        explanation=data.get("explanation", ""),
    )
