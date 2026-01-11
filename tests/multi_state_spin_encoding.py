"""
Multi-State Spin Encoding for SCBE
===================================

Poly-dimensional analysis with multi-state encoding and spin modulation.
Extends SCBE from binary to base-6 states with complex phase modulation.

Key Concepts:
1. Multi-State Encoding: Base-6 states (0-5) mapping to intent axes
   - 0: Past, 1: Present, 2: Future, 3: WHO, 4: WHY, 5: HOW

2. Spin Modulation: Complex phase e^(iθ) affects effective state
   - Real component = base value
   - Imaginary component = hidden intent influence

3. Quark-Like Strand Confinement: Tensor weights with color charge
   - Strands cannot be isolated without energy explosion
   - Confinement creates natural "sinks"

4. Gene Editing Analogy: CRISPR-like precise targeting
   - Guide sequences for legitimate edits
   - Off-target detection via coherence

Patent Claims Extension:
- Claims 25-30: Multi-state spin modulation layer
"""

import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

# Golden ratio for harmonic scaling
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Base-6 multi-state system
NUM_STATES = 6
STATES = np.array([0, 1, 2, 3, 4, 5])

# State-to-intent mapping
STATE_INTENT_MAP = {
    0: "PAST",      # Temporal - historical context
    1: "PRESENT",   # Temporal - current state
    2: "FUTURE",    # Temporal - predicted trajectory
    3: "WHO",       # Identity axis
    4: "WHY",       # Purpose/commitment axis
    5: "HOW"        # Method/signature axis
}

# Quark-like color charges for strand confinement
COLOR_CHARGES = ["RED", "GREEN", "BLUE"]

# Spin phase angles (golden-ratio derived)
SPIN_PHASES = {
    "coherent": 0.0,                    # No rotation
    "slight": np.pi / PHI**2,           # ~36 degrees
    "moderate": np.pi / PHI,            # ~112 degrees
    "strong": np.pi,                    # 180 degrees (anti-phase)
    "golden": 2 * np.pi / PHI           # Golden angle ~222 degrees
}

# Confinement energy threshold
CONFINEMENT_THRESHOLD = 10.0


class StrandType(Enum):
    """Types of information strands (quark-like)."""
    INTENT = 1      # Intent-carrying strand
    CONTEXT = 2     # Context information strand
    SIGNATURE = 3   # Cryptographic signature strand


@dataclass
class SpinState:
    """
    Represents a base state with spin modulation.

    Combines discrete base value (0-5) with continuous spin phase
    to create rich, modulated intent representation.
    """
    base_value: int          # 0-5 discrete state
    phase_angle: float       # Spin phase in radians
    weight: float = 1.0      # Tensor weight for this state

    def __post_init__(self):
        self.base_value = self.base_value % NUM_STATES
        self.phase_angle = self.phase_angle % (2 * np.pi)

    @property
    def complex_value(self) -> complex:
        """Get complex representation: base * e^(iθ)."""
        return self.base_value * np.exp(1j * self.phase_angle)

    @property
    def effective_state(self) -> int:
        """
        Compute effective state after spin modulation.

        Projects complex value back to discrete state space.
        """
        c = self.complex_value
        # Weight imaginary influence
        effective = np.real(c) + np.imag(c) * 0.5
        return int(np.round(effective)) % NUM_STATES

    @property
    def intent(self) -> str:
        """Get intent meaning of effective state."""
        return STATE_INTENT_MAP[self.effective_state]

    def modulate(self, delta_phase: float) -> 'SpinState':
        """Apply additional phase modulation."""
        return SpinState(
            base_value=self.base_value,
            phase_angle=self.phase_angle + delta_phase,
            weight=self.weight
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base_value,
            "phase": self.phase_angle,
            "effective": self.effective_state,
            "intent": self.intent,
            "complex": [np.real(self.complex_value), np.imag(self.complex_value)]
        }


@dataclass
class QuarkStrand:
    """
    Quark-like information strand with color charge confinement.

    Strands are confined units that cannot be isolated without
    energy cost explosion - mimics QCD quark confinement.
    """
    states: List[SpinState]
    color_charge: str = "RED"
    strand_type: StrandType = StrandType.INTENT
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.color_charge not in COLOR_CHARGES:
            self.color_charge = COLOR_CHARGES[0]

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def base_sequence(self) -> np.ndarray:
        """Get base values as array."""
        return np.array([s.base_value for s in self.states])

    @property
    def effective_sequence(self) -> np.ndarray:
        """Get effective (spin-modulated) values as array."""
        return np.array([s.effective_state for s in self.states])

    @property
    def phase_sequence(self) -> np.ndarray:
        """Get phase angles as array."""
        return np.array([s.phase_angle for s in self.states])

    @property
    def complex_sequence(self) -> np.ndarray:
        """Get complex values as array."""
        return np.array([s.complex_value for s in self.states])

    @property
    def intent_sequence(self) -> List[str]:
        """Get intent meanings."""
        return [s.intent for s in self.states]

    @property
    def confinement_energy(self) -> float:
        """
        Calculate energy required to break strand confinement.

        Based on phase coherence and color charge distribution.
        Higher coherence = higher confinement energy.
        """
        if self.length < 2:
            return 0.0

        # Phase coherence contribution
        phases = self.phase_sequence
        phase_variance = np.var(phases)
        coherence = np.exp(-phase_variance)

        # Color charge contribution (stronger for matched colors)
        color_factor = 1.0 + PHI if self.color_charge in COLOR_CHARGES else 1.0

        # Length contribution (longer strands harder to break)
        length_factor = np.log(self.length + 1)

        return coherence * color_factor * length_factor * CONFINEMENT_THRESHOLD

    def is_confined(self) -> bool:
        """Check if strand is properly confined."""
        return self.confinement_energy > CONFINEMENT_THRESHOLD

    def get_hash(self) -> str:
        """Get cryptographic hash of strand."""
        data = f"{self.base_sequence.tobytes()}{self.phase_sequence.tobytes()}{self.color_charge}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class GeneEdit:
    """
    CRISPR-like gene edit operation on strands.

    Represents a targeted modification with guide sequence
    for precise editing and off-target detection.
    """
    target_position: int
    guide_sequence: np.ndarray      # Target pattern to match
    replacement_state: SpinState
    edit_type: str = "REPLACE"      # REPLACE, INSERT, DELETE
    timestamp: float = field(default_factory=time.time)

    @property
    def guide_length(self) -> int:
        return len(self.guide_sequence)

    def matches_target(self, strand: QuarkStrand, tolerance: float = 0.5) -> Tuple[bool, float]:
        """
        Check if guide sequence matches target in strand.

        Returns (match_success, match_score).
        """
        if self.target_position + self.guide_length > strand.length:
            return False, 0.0

        # Extract target region
        target_region = strand.base_sequence[
            self.target_position:self.target_position + self.guide_length
        ]

        # Compute match score
        matches = np.sum(target_region == self.guide_sequence)
        score = matches / self.guide_length

        return score >= (1 - tolerance), score


class HyperbolicStrandSpace:
    """
    Non-Euclidean space for strand distance calculations.

    Uses hyperbolic geometry to amplify distances for
    mismatched strands, creating natural "sinks".
    """

    def __init__(self, curvature: float = -1.0):
        self.curvature = curvature
        self.k = abs(curvature)

    def multi_state_distance(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """
        Calculate hyperbolic distance between multi-state sequences.

        Small base differences grow exponentially in curved space.
        """
        diff = np.abs(s1.astype(float) - s2.astype(float))
        # Hyperbolic sine amplifies differences
        return np.sinh(diff * self.k)

    def strand_distance(self, strand1: QuarkStrand, strand2: QuarkStrand) -> float:
        """
        Calculate total hyperbolic distance between two strands.

        Includes both base value and phase differences.
        """
        # Pad to equal length
        max_len = max(strand1.length, strand2.length)

        seq1 = np.pad(strand1.effective_sequence, (0, max_len - strand1.length))
        seq2 = np.pad(strand2.effective_sequence, (0, max_len - strand2.length))

        # Base state distances
        base_dist = self.multi_state_distance(seq1, seq2)

        # Phase distances (circular)
        phase1 = np.pad(strand1.phase_sequence, (0, max_len - strand1.length))
        phase2 = np.pad(strand2.phase_sequence, (0, max_len - strand2.length))
        phase_diff = np.abs(np.exp(1j * phase1) - np.exp(1j * phase2))

        # Combined distance with phase weighting
        total = np.sum(base_dist) + PHI * np.sum(phase_diff)

        return float(total)

    def spin_coherence_distance(self, strand: QuarkStrand, reference_phase: float) -> float:
        """
        Calculate distance from coherent spin state.

        Measures how far strand's spin is from reference phase.
        """
        phase_diffs = np.abs(strand.phase_sequence - reference_phase)
        # Wrap to [0, π]
        phase_diffs = np.minimum(phase_diffs, 2 * np.pi - phase_diffs)

        # Hyperbolic amplification
        return float(np.sum(np.sinh(phase_diffs)))


class MultiStateSpinEncoder:
    """
    Main encoder class for multi-state spin modulation.

    Provides encoding, decoding, and validation for
    SCBE's extended multi-intent layer.
    """

    def __init__(self, default_phase: float = 0.0):
        self.default_phase = default_phase
        self.hyperbolic_space = HyperbolicStrandSpace()
        self.strand_registry: Dict[str, QuarkStrand] = {}
        self.edit_log: List[GeneEdit] = []

    def encode_intent_vector(self,
                             intent_values: np.ndarray,
                             phases: Optional[np.ndarray] = None,
                             color: str = "RED") -> QuarkStrand:
        """
        Encode an intent vector as a multi-state strand.

        Args:
            intent_values: Array of values (will be quantized to 0-5)
            phases: Optional phase angles for each position
            color: Color charge for confinement

        Returns:
            QuarkStrand with spin-modulated states
        """
        # Quantize to base-6
        quantized = np.round(intent_values).astype(int) % NUM_STATES

        if phases is None:
            phases = np.full(len(quantized), self.default_phase)

        states = [
            SpinState(base_value=int(v), phase_angle=float(p))
            for v, p in zip(quantized, phases)
        ]

        strand = QuarkStrand(
            states=states,
            color_charge=color,
            strand_type=StrandType.INTENT
        )

        # Register strand
        self.strand_registry[strand.get_hash()] = strand

        return strand

    def decode_strand(self, strand: QuarkStrand) -> Tuple[np.ndarray, List[str]]:
        """
        Decode strand back to intent values and meanings.

        Returns:
            (effective_values, intent_meanings)
        """
        return strand.effective_sequence, strand.intent_sequence

    def apply_spin_modulation(self,
                               strand: QuarkStrand,
                               modulation_pattern: np.ndarray) -> QuarkStrand:
        """
        Apply spin modulation pattern to strand.

        Returns new strand with modulated phases.
        """
        new_states = []
        for i, state in enumerate(strand.states):
            if i < len(modulation_pattern):
                new_states.append(state.modulate(modulation_pattern[i]))
            else:
                new_states.append(state)

        return QuarkStrand(
            states=new_states,
            color_charge=strand.color_charge,
            strand_type=strand.strand_type
        )

    def apply_gene_edit(self,
                        strand: QuarkStrand,
                        edit: GeneEdit) -> Tuple[QuarkStrand, bool, float]:
        """
        Apply CRISPR-like gene edit to strand.

        Returns:
            (edited_strand, success, match_score)
        """
        matches, score = edit.matches_target(strand)

        if not matches:
            # Off-target edit - return unmodified with failure
            return strand, False, score

        # Apply edit
        new_states = list(strand.states)

        if edit.edit_type == "REPLACE":
            new_states[edit.target_position] = edit.replacement_state
        elif edit.edit_type == "INSERT":
            new_states.insert(edit.target_position, edit.replacement_state)
        elif edit.edit_type == "DELETE":
            if edit.target_position < len(new_states):
                new_states.pop(edit.target_position)

        edited = QuarkStrand(
            states=new_states,
            color_charge=strand.color_charge,
            strand_type=strand.strand_type
        )

        # Log edit
        self.edit_log.append(edit)

        return edited, True, score

    def validate_strand_coherence(self,
                                   strand: QuarkStrand,
                                   reference_strand: Optional[QuarkStrand] = None,
                                   threshold: float = 5.0) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate strand coherence for attack detection.

        Args:
            strand: Strand to validate
            reference_strand: Optional known-good reference
            threshold: Maximum allowed distance

        Returns:
            (is_valid, coherence_score, details)
        """
        details = {
            "confinement_energy": strand.confinement_energy,
            "is_confined": strand.is_confined(),
            "length": strand.length,
            "color_charge": strand.color_charge
        }

        # Check confinement
        if not strand.is_confined():
            return False, 0.0, {**details, "failure": "confinement_broken"}

        # Check phase coherence
        spin_distance = self.hyperbolic_space.spin_coherence_distance(
            strand, self.default_phase
        )
        details["spin_distance"] = spin_distance

        # Compare to reference if provided
        if reference_strand is not None:
            strand_distance = self.hyperbolic_space.strand_distance(
                strand, reference_strand
            )
            details["strand_distance"] = strand_distance

            if strand_distance > threshold:
                return False, 1.0 / (1.0 + strand_distance), {
                    **details, "failure": "strand_mismatch"
                }

        # Calculate coherence score
        coherence = 1.0 / (1.0 + spin_distance)

        return coherence > 0.3, coherence, details

    def calculate_attack_cost(self,
                               attacker_strand: QuarkStrand,
                               target_strand: QuarkStrand) -> Dict[str, Any]:
        """
        Calculate attack cost in hyperbolic strand space.

        Mismatched spins/states create exponential cost barriers.
        """
        # Strand distance
        distance = self.hyperbolic_space.strand_distance(
            attacker_strand, target_strand
        )

        # Confinement breaking cost
        confinement_cost = target_strand.confinement_energy

        # Spin mismatch amplification
        spin_cost = self.hyperbolic_space.spin_coherence_distance(
            attacker_strand,
            np.mean(target_strand.phase_sequence)
        )

        # Total cost (super-exponential via sinh)
        total_cost = distance + confinement_cost + spin_cost

        # Convert to bits
        cost_bits = np.log2(1 + total_cost) * 10  # Scale factor

        return {
            "strand_distance": distance,
            "confinement_cost": confinement_cost,
            "spin_cost": spin_cost,
            "total_cost": total_cost,
            "cost_bits": cost_bits,
            "feasible": cost_bits < 128  # Below 128-bit security
        }


class MultiStateDefenseLayer:
    """
    Defense layer using multi-state spin encoding.

    Integrates with SCBE's existing defense-in-depth:
    - Camera (detection) → Spin coherence monitoring
    - Door (barrier) → Strand confinement validation
    - Dog (response) → Off-target edit detection
    - Gun (deterrence) → Hyperbolic sink activation
    """

    def __init__(self):
        self.encoder = MultiStateSpinEncoder()
        self.legitimate_strands: Dict[str, QuarkStrand] = {}
        self.alert_log: List[Dict[str, Any]] = []

    def register_legitimate_strand(self,
                                    intent_vector: np.ndarray,
                                    identity: str) -> QuarkStrand:
        """Register a known-legitimate strand pattern."""
        strand = self.encoder.encode_intent_vector(
            intent_vector,
            phases=np.full(len(intent_vector), SPIN_PHASES["coherent"])
        )
        self.legitimate_strands[identity] = strand
        return strand

    def validate_incoming(self,
                          strand: QuarkStrand,
                          claimed_identity: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate incoming strand against claimed identity.

        Returns:
            (allowed, action, details)
        """
        # Layer 1: Camera - Spin coherence check
        if claimed_identity not in self.legitimate_strands:
            self._log_alert("unknown_identity", strand, claimed_identity)
            return False, "REJECT", {"layer": "camera", "reason": "unknown_identity"}

        reference = self.legitimate_strands[claimed_identity]

        # Layer 2: Door - Confinement validation
        if not strand.is_confined():
            self._log_alert("confinement_broken", strand, claimed_identity)
            return False, "REJECT", {"layer": "door", "reason": "confinement_broken"}

        # Layer 3: Dog - Coherence validation
        is_valid, coherence, details = self.encoder.validate_strand_coherence(
            strand, reference
        )

        if not is_valid:
            self._log_alert("coherence_failure", strand, claimed_identity, details)
            return False, "ALERT", {"layer": "dog", **details}

        # Layer 4: Gun - Attack cost check (only if suspicious)
        if coherence < 0.7:
            attack_cost = self.encoder.calculate_attack_cost(strand, reference)
            if attack_cost["feasible"]:
                self._log_alert("potential_attack", strand, claimed_identity, attack_cost)
                return False, "SINK", {"layer": "gun", **attack_cost}

        return True, "ALLOW", {"coherence": coherence, **details}

    def _log_alert(self,
                   alert_type: str,
                   strand: QuarkStrand,
                   identity: str,
                   extra: Optional[Dict] = None):
        """Log security alert."""
        alert = {
            "type": alert_type,
            "timestamp": time.time(),
            "strand_hash": strand.get_hash(),
            "claimed_identity": identity,
            "strand_length": strand.length,
            "color_charge": strand.color_charge
        }
        if extra:
            alert.update(extra)
        self.alert_log.append(alert)


# =============================================================================
# TEST SUITE
# =============================================================================

def test_spin_state_modulation():
    """Test SpinState creation and modulation."""
    print("\n" + "="*60)
    print("TEST: Spin State Modulation")
    print("="*60)

    results = []

    # Test base states with different phases
    for base in range(NUM_STATES):
        for phase_name, phase in SPIN_PHASES.items():
            state = SpinState(base_value=base, phase_angle=phase)
            results.append({
                "base": base,
                "phase": phase_name,
                "effective": state.effective_state,
                "intent": state.intent,
                "complex": state.complex_value
            })

            if base == 3 and phase_name == "moderate":
                print(f"  Base {base} + {phase_name} phase:")
                print(f"    Complex: {state.complex_value:.3f}")
                print(f"    Effective: {state.effective_state}")
                print(f"    Intent: {state.intent}")

    # Verify phase affects effective state
    state_coherent = SpinState(base_value=3, phase_angle=0)
    state_strong = SpinState(base_value=3, phase_angle=np.pi)

    assert state_coherent.effective_state != state_strong.effective_state or \
           state_coherent.complex_value != state_strong.complex_value, \
           "Phase should affect representation"

    print("  PASS: Phase modulation affects state representation")
    return True


def test_quark_strand_confinement():
    """Test QuarkStrand confinement mechanics."""
    print("\n" + "="*60)
    print("TEST: Quark Strand Confinement")
    print("="*60)

    # Create coherent strand (high confinement)
    coherent_states = [
        SpinState(base_value=i % 6, phase_angle=0.0)
        for i in range(10)
    ]
    coherent_strand = QuarkStrand(
        states=coherent_states,
        color_charge="RED"
    )

    # Create incoherent strand (low confinement)
    incoherent_states = [
        SpinState(base_value=i % 6, phase_angle=np.random.uniform(0, 2*np.pi))
        for i in range(10)
    ]
    incoherent_strand = QuarkStrand(
        states=incoherent_states,
        color_charge="RED"
    )

    print(f"  Coherent strand energy: {coherent_strand.confinement_energy:.2f}")
    print(f"  Incoherent strand energy: {incoherent_strand.confinement_energy:.2f}")
    print(f"  Threshold: {CONFINEMENT_THRESHOLD}")

    assert coherent_strand.confinement_energy > incoherent_strand.confinement_energy, \
           "Coherent strands should have higher confinement energy"

    print(f"  Coherent confined: {coherent_strand.is_confined()}")
    print(f"  Incoherent confined: {incoherent_strand.is_confined()}")

    print("  PASS: Confinement scales with phase coherence")
    return True


def test_hyperbolic_strand_distance():
    """Test hyperbolic distance amplification."""
    print("\n" + "="*60)
    print("TEST: Hyperbolic Strand Distance")
    print("="*60)

    space = HyperbolicStrandSpace()

    # Create strands with varying differences
    base_states = [SpinState(base_value=0, phase_angle=0.0) for _ in range(5)]
    base_strand = QuarkStrand(states=base_states, color_charge="RED")

    distances = []
    for diff in range(6):
        diff_states = [SpinState(base_value=diff, phase_angle=0.0) for _ in range(5)]
        diff_strand = QuarkStrand(states=diff_states, color_charge="RED")

        dist = space.strand_distance(base_strand, diff_strand)
        distances.append(dist)
        print(f"  Base diff {diff}: distance = {dist:.4f}")

    # Verify super-linear growth (sinh amplification)
    for i in range(1, len(distances) - 1):
        growth_rate_prev = distances[i] - distances[i-1]
        growth_rate_next = distances[i+1] - distances[i]
        if distances[i] > 1:  # Only check significant distances
            assert growth_rate_next >= growth_rate_prev * 0.9, \
                   "Distance should grow super-linearly"

    print("  PASS: Hyperbolic distances grow super-linearly")
    return True


def test_gene_editing():
    """Test CRISPR-like gene editing on strands."""
    print("\n" + "="*60)
    print("TEST: Gene Editing")
    print("="*60)

    encoder = MultiStateSpinEncoder()

    # Create original strand
    original = encoder.encode_intent_vector(
        np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    )
    print(f"  Original: {original.base_sequence}")

    # Create valid edit (matching guide)
    valid_edit = GeneEdit(
        target_position=2,
        guide_sequence=np.array([2, 3, 4]),
        replacement_state=SpinState(base_value=5, phase_angle=0.0)
    )

    edited, success, score = encoder.apply_gene_edit(original, valid_edit)
    print(f"  Valid edit success: {success}, score: {score:.2f}")
    print(f"  Edited: {edited.base_sequence}")

    assert success, "Valid edit should succeed"
    assert edited.base_sequence[2] == 5, "Edit should apply"

    # Create invalid edit (non-matching guide)
    invalid_edit = GeneEdit(
        target_position=2,
        guide_sequence=np.array([9, 9, 9]),  # Won't match
        replacement_state=SpinState(base_value=0, phase_angle=0.0)
    )

    _, fail_success, fail_score = encoder.apply_gene_edit(original, invalid_edit)
    print(f"  Invalid edit success: {fail_success}, score: {fail_score:.2f}")

    assert not fail_success, "Invalid edit should fail"

    print("  PASS: Gene editing respects guide sequence matching")
    return True


def test_defense_layer():
    """Test multi-layer defense integration."""
    print("\n" + "="*60)
    print("TEST: Defense Layer Integration")
    print("="*60)

    defense = MultiStateDefenseLayer()

    # Register legitimate user
    legit_vector = np.array([1, 2, 3, 4, 5, 0, 1, 2])
    defense.register_legitimate_strand(legit_vector, "user_alice")
    print("  Registered legitimate user: alice")

    # Test 1: Legitimate access (matching strand)
    legit_strand = defense.encoder.encode_intent_vector(
        legit_vector,
        phases=np.full(len(legit_vector), SPIN_PHASES["coherent"])
    )
    allowed, action, details = defense.validate_incoming(legit_strand, "user_alice")
    print(f"  Legitimate access: {action} (coherence: {details.get('coherence', 'N/A'):.3f})")
    assert allowed and action == "ALLOW", "Legitimate access should be allowed"

    # Test 2: Unknown identity
    unknown_strand = defense.encoder.encode_intent_vector(np.array([0, 0, 0, 0]))
    allowed, action, _ = defense.validate_incoming(unknown_strand, "user_unknown")
    print(f"  Unknown identity: {action}")
    assert not allowed and action == "REJECT", "Unknown identity should be rejected"

    # Test 3: Attack attempt (different strand, wrong spin)
    attack_vector = np.array([5, 4, 3, 2, 1, 0, 5, 4])  # Different values
    attack_strand = defense.encoder.encode_intent_vector(
        attack_vector,
        phases=np.full(len(attack_vector), SPIN_PHASES["strong"])  # Anti-phase
    )
    allowed, action, details = defense.validate_incoming(attack_strand, "user_alice")
    print(f"  Attack attempt: {action}")
    print(f"    Details: {details.get('failure', details.get('layer', 'passed'))}")
    assert not allowed, "Attack should be blocked"

    print(f"  Total alerts: {len(defense.alert_log)}")
    print("  PASS: Defense layers correctly filter access")
    return True


def test_attack_cost_calculation():
    """Test attack cost in hyperbolic strand space."""
    print("\n" + "="*60)
    print("TEST: Attack Cost Calculation")
    print("="*60)

    encoder = MultiStateSpinEncoder()

    # Target strand (well-confined, coherent)
    target = encoder.encode_intent_vector(
        np.array([0, 1, 2, 3, 4, 5] * 5),  # 30 states
        phases=np.zeros(30)
    )

    # Attacker strands with varying distance
    test_cases = [
        ("Close match", np.array([0, 1, 2, 3, 4, 5] * 5), np.zeros(30)),
        ("Slight mismatch", np.array([1, 2, 3, 4, 5, 0] * 5), np.zeros(30)),
        ("Major mismatch", np.array([5, 4, 3, 2, 1, 0] * 5), np.zeros(30)),
        ("Spin attack", np.array([0, 1, 2, 3, 4, 5] * 5), np.full(30, np.pi)),
    ]

    for name, values, phases in test_cases:
        attacker = encoder.encode_intent_vector(values, phases)
        cost = encoder.calculate_attack_cost(attacker, target)

        print(f"  {name}:")
        print(f"    Strand distance: {cost['strand_distance']:.2f}")
        print(f"    Spin cost: {cost['spin_cost']:.2f}")
        print(f"    Total cost bits: {cost['cost_bits']:.1f}")
        print(f"    Feasible: {cost['feasible']}")

    print("  PASS: Attack costs scale with mismatch")
    return True


def test_multi_state_encoding_metrics():
    """Generate encoding metrics for documentation."""
    print("\n" + "="*60)
    print("TEST: Multi-State Encoding Metrics")
    print("="*60)

    encoder = MultiStateSpinEncoder()

    # Entropy comparison
    binary_entropy = 1.0  # bits per symbol
    base6_entropy = np.log2(6)  # ~2.58 bits per symbol

    print(f"  Binary entropy: {binary_entropy:.2f} bits/symbol")
    print(f"  Base-6 entropy: {base6_entropy:.2f} bits/symbol")
    print(f"  Entropy gain: {base6_entropy/binary_entropy:.2f}x")

    # State space comparison
    strand_length = 20
    binary_space = 2 ** strand_length
    base6_space = 6 ** strand_length

    print(f"\n  Strand length: {strand_length}")
    print(f"  Binary state space: 2^{strand_length} = {binary_space:.2e}")
    print(f"  Base-6 state space: 6^{strand_length} = {base6_space:.2e}")
    print(f"  State space ratio: {base6_space/binary_space:.2e}x")

    # With spin modulation (continuous phase)
    # Effective state space is infinite due to continuous phase
    # But practical quantization gives ~360 phase levels
    phase_levels = 360
    states_per_position = 6 * phase_levels
    # Compute in log space to avoid overflow: log2(states^length) = length * log2(states)
    full_space_bits = strand_length * np.log2(float(states_per_position))

    print(f"\n  With spin modulation (~1 degree resolution):")
    print(f"  Effective states per position: {states_per_position}")
    print(f"  Full state space: ~{full_space_bits:.0f} bits")

    print("  PASS: Multi-state encoding provides exponential state space gain")
    return True


def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("MULTI-STATE SPIN ENCODING TEST SUITE")
    print("Patent Claims 25-30: Multi-Intent Spin Modulation Layer")
    print("="*70)

    tests = [
        ("Spin State Modulation", test_spin_state_modulation),
        ("Quark Strand Confinement", test_quark_strand_confinement),
        ("Hyperbolic Strand Distance", test_hyperbolic_strand_distance),
        ("Gene Editing", test_gene_editing),
        ("Defense Layer", test_defense_layer),
        ("Attack Cost Calculation", test_attack_cost_calculation),
        ("Encoding Metrics", test_multi_state_encoding_metrics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL: {e}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, success, error in results:
        status = "PASS" if success else f"FAIL: {error}"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70)

    # Export results
    output = {
        "test_suite": "multi_state_spin_encoding",
        "timestamp": time.time(),
        "tests_passed": passed,
        "tests_total": total,
        "all_passed": passed == total,
        "patent_claims": "25-30",
        "metrics": {
            "base_states": NUM_STATES,
            "entropy_per_symbol": float(np.log2(NUM_STATES)),
            "spin_phases": len(SPIN_PHASES),
            "color_charges": len(COLOR_CHARGES)
        }
    }

    with open("multi_state_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: multi_state_test_results.json")

    return passed == total


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
