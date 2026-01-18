"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCBE LAYER MAPPING                                        ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Maps compliance tests to SCBE 14-Layer Pipeline                             ║
║                                                                              ║
║  Layer Architecture:                                                         ║
║  L1  - Axiom Verifier         (Mathematical foundations)                     ║
║  L2  - Phase Verifier         (Phase-breath transforms)                      ║
║  L3  - Hyperbolic Distance    (Geodesic integrity)                           ║
║  L4  - Entropy Flow           (Thermodynamic bounds)                         ║
║  L5  - Quantum Coherence      (PQC state validation)                         ║
║  L6  - Session Key            (Key establishment)                            ║
║  L7  - Trajectory Smooth      (Path continuity)                              ║
║  L8  - Boundary Proximity     (Constraint satisfaction)                      ║
║  L9  - Crypto Integrity       (Cryptographic verification)                   ║
║  L10 - Temporal Consistency   (Time-based validation)                        ║
║  L11 - Manifold Curvature     (Geometric bounds)                             ║
║  L12 - Energy Conservation    (Resource constraints)                         ║
║  L13 - Decision Boundary      (Classification thresholds)                    ║
║  L14 - Governance Decision    (Final policy enforcement)                     ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, Any


class SCBELayer(Enum):
    """SCBE 14-Layer Pipeline layers."""
    L1_AXIOM_VERIFIER = 1
    L2_PHASE_VERIFIER = 2
    L3_HYPERBOLIC_DISTANCE = 3
    L4_ENTROPY_FLOW = 4
    L5_QUANTUM_COHERENCE = 5
    L6_SESSION_KEY = 6
    L7_TRAJECTORY_SMOOTH = 7
    L8_BOUNDARY_PROXIMITY = 8
    L9_CRYPTO_INTEGRITY = 9
    L10_TEMPORAL_CONSISTENCY = 10
    L11_MANIFOLD_CURVATURE = 11
    L12_ENERGY_CONSERVATION = 12
    L13_DECISION_BOUNDARY = 13
    L14_GOVERNANCE_DECISION = 14


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    HIPAA = "hipaa"              # Health Insurance Portability and Accountability Act
    HITECH = "hitech"           # Health Information Technology for Economic and Clinical Health
    NIST_800_53 = "nist_800_53" # NIST Security and Privacy Controls
    FIPS_140_3 = "fips_140_3"   # FIPS Cryptographic Module Validation
    PCI_DSS = "pci_dss"         # Payment Card Industry Data Security Standard
    IEC_62443 = "iec_62443"     # Industrial Communication Networks Security
    ISO_27001 = "iso_27001"     # Information Security Management
    SOC2_TYPE2 = "soc2_type2"   # Service Organization Control 2


# Layer descriptions for documentation
LAYER_DESCRIPTIONS: Dict[SCBELayer, Dict[str, str]] = {
    SCBELayer.L1_AXIOM_VERIFIER: {
        "name": "Axiom Verifier",
        "purpose": "Validates mathematical foundations and invariants",
        "axiom": "4.1 - Harmonic Space Core",
        "formula": "H(d, R) = R^(d²)",
    },
    SCBELayer.L2_PHASE_VERIFIER: {
        "name": "Phase Verifier",
        "purpose": "Validates phase-breath transforms and signal integrity",
        "axiom": "4.3 - HAL-Attention",
        "formula": "phase_modulated_intent(signal, phase)",
    },
    SCBELayer.L3_HYPERBOLIC_DISTANCE: {
        "name": "Hyperbolic Distance",
        "purpose": "Measures geodesic distance on hyperbolic manifold",
        "axiom": "4.1 - Harmonic Space Core",
        "formula": "d_H(u, v) = √(Σ w_i(u_i - v_i)²)",
    },
    SCBELayer.L4_ENTROPY_FLOW: {
        "name": "Entropy Flow",
        "purpose": "Validates thermodynamic bounds and entropy constraints",
        "axiom": "4.5 - Vacuum-Acoustics",
        "formula": "η_min ≤ η(t) ≤ η_max",
    },
    SCBELayer.L5_QUANTUM_COHERENCE: {
        "name": "Quantum Coherence",
        "purpose": "Validates PQC state and quantum-resistant operations",
        "axiom": "5 - PQC Harmonic",
        "formula": "S_bits = B + d² × log₂(R)",
    },
    SCBELayer.L6_SESSION_KEY: {
        "name": "Session Key",
        "purpose": "Establishes and validates session keys",
        "axiom": "5 - PQC Harmonic / 7 - Spiral Seal",
        "formula": "K_session = Kyber.Decaps(ct, sk)",
    },
    SCBELayer.L7_TRAJECTORY_SMOOTH: {
        "name": "Trajectory Smooth",
        "purpose": "Ensures path continuity without discontinuities",
        "axiom": "4.6 - Cymatic Storage",
        "formula": "|∇trajectory| < threshold",
    },
    SCBELayer.L8_BOUNDARY_PROXIMITY: {
        "name": "Boundary Proximity",
        "purpose": "Validates constraint satisfaction at boundaries",
        "axiom": "8 - Quasicrystal Lattice",
        "formula": "dist_to_boundary > ε_safe",
    },
    SCBELayer.L9_CRYPTO_INTEGRITY: {
        "name": "Crypto Integrity",
        "purpose": "Verifies cryptographic operations and MACs",
        "axiom": "7 - Spiral Seal",
        "formula": "HMAC(K, m) = tag",
    },
    SCBELayer.L10_TEMPORAL_CONSISTENCY: {
        "name": "Temporal Consistency",
        "purpose": "Validates time-based constraints and ordering",
        "axiom": "6 - EDE",
        "formula": "τ_coherence > t_current - t_prev",
    },
    SCBELayer.L11_MANIFOLD_CURVATURE: {
        "name": "Manifold Curvature",
        "purpose": "Bounds curvature on the governance manifold",
        "axiom": "4.1 / 4.5",
        "formula": "|κ| < κ_max",
    },
    SCBELayer.L12_ENERGY_CONSERVATION: {
        "name": "Energy Conservation",
        "purpose": "Validates resource constraints and energy bounds",
        "axiom": "4.5 - Vacuum-Acoustics",
        "formula": "E_total = constant (flux redistribution)",
    },
    SCBELayer.L13_DECISION_BOUNDARY: {
        "name": "Decision Boundary",
        "purpose": "Classification thresholds and decision margins",
        "axiom": "4.3 - HAL-Attention",
        "formula": "margin > λ_bound",
    },
    SCBELayer.L14_GOVERNANCE_DECISION: {
        "name": "Governance Decision",
        "purpose": "Final policy enforcement and action authorization",
        "axiom": "All (Full Pipeline)",
        "formula": "decision = f(L1...L13)",
    },
}


# Standard to Layer mappings
STANDARD_LAYER_MAPPING: Dict[ComplianceStandard, Set[SCBELayer]] = {
    ComplianceStandard.HIPAA: {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    ComplianceStandard.HITECH: {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },
    ComplianceStandard.NIST_800_53: {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L2_PHASE_VERIFIER,
        SCBELayer.L3_HYPERBOLIC_DISTANCE,
        SCBELayer.L4_ENTROPY_FLOW,
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    ComplianceStandard.FIPS_140_3: {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    ComplianceStandard.PCI_DSS: {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    ComplianceStandard.IEC_62443: {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L8_BOUNDARY_PROXIMITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    ComplianceStandard.ISO_27001: {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    ComplianceStandard.SOC2_TYPE2: {
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
}


# Test category to layer mappings
TEST_CATEGORY_MAPPING: Dict[str, Set[SCBELayer]] = {
    # Self-Healing Workflow
    "self_healing": {
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L12_ENERGY_CONSERVATION,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "circuit_breaker": {
        SCBELayer.L8_BOUNDARY_PROXIMITY,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },

    # Medical AI-to-AI
    "medical_phi": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    "medical_audit": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },

    # Military-Grade
    "military_classification": {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "military_key_rotation": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },

    # Adversarial Resistance
    "replay_attack": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },
    "bit_flip": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    "timing_attack": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },
    "padding_oracle": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    "chosen_plaintext": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    "key_extraction": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },

    # Zero-Trust
    "zero_trust": {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L2_PHASE_VERIFIER,
        SCBELayer.L3_HYPERBOLIC_DISTANCE,
        SCBELayer.L4_ENTROPY_FLOW,
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L8_BOUNDARY_PROXIMITY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L11_MANIFOLD_CURVATURE,
        SCBELayer.L12_ENERGY_CONSERVATION,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
}


@dataclass
class LayerMapping:
    """Complete mapping of a test to SCBE layers and compliance standards."""
    test_name: str
    test_id: str
    category: str
    layers: Set[SCBELayer]
    standards: Set[ComplianceStandard]
    axioms: Set[str] = field(default_factory=set)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'test_id': self.test_id,
            'category': self.category,
            'layers': [l.name for l in self.layers],
            'layer_numbers': sorted([l.value for l in self.layers]),
            'standards': [s.value for s in self.standards],
            'axioms': list(self.axioms),
            'description': self.description,
        }


def get_layers_for_category(category: str) -> Set[SCBELayer]:
    """Get SCBE layers for a test category."""
    return TEST_CATEGORY_MAPPING.get(category, set())


def get_layers_for_standard(standard: ComplianceStandard) -> Set[SCBELayer]:
    """Get SCBE layers required by a compliance standard."""
    return STANDARD_LAYER_MAPPING.get(standard, set())


def get_standards_for_layers(layers: Set[SCBELayer]) -> Set[ComplianceStandard]:
    """Get compliance standards satisfied by a set of layers."""
    satisfied = set()
    for standard, required_layers in STANDARD_LAYER_MAPPING.items():
        if required_layers.issubset(layers):
            satisfied.add(standard)
    return satisfied


def get_axioms_for_layers(layers: Set[SCBELayer]) -> Set[str]:
    """Get axiom IDs covered by a set of layers."""
    axioms = set()
    axiom_map = {
        SCBELayer.L1_AXIOM_VERIFIER: "4.1",
        SCBELayer.L2_PHASE_VERIFIER: "4.3",
        SCBELayer.L3_HYPERBOLIC_DISTANCE: "4.1",
        SCBELayer.L4_ENTROPY_FLOW: "4.5",
        SCBELayer.L5_QUANTUM_COHERENCE: "5",
        SCBELayer.L6_SESSION_KEY: "5",
        SCBELayer.L7_TRAJECTORY_SMOOTH: "4.6",
        SCBELayer.L8_BOUNDARY_PROXIMITY: "8",
        SCBELayer.L9_CRYPTO_INTEGRITY: "7",
        SCBELayer.L10_TEMPORAL_CONSISTENCY: "6",
        SCBELayer.L11_MANIFOLD_CURVATURE: "4.1",
        SCBELayer.L12_ENERGY_CONSERVATION: "4.5",
        SCBELayer.L13_DECISION_BOUNDARY: "4.3",
        SCBELayer.L14_GOVERNANCE_DECISION: "ALL",
    }
    for layer in layers:
        if layer in axiom_map:
            axioms.add(axiom_map[layer])
    return axioms


def get_layer_for_test(
    test_name: str,
    test_id: str,
    category: str,
    standards: Optional[Set[ComplianceStandard]] = None,
    description: str = "",
) -> LayerMapping:
    """
    Create a complete layer mapping for a test.

    Args:
        test_name: Human-readable test name
        test_id: Unique test identifier (e.g., "test_101")
        category: Test category (e.g., "medical_phi", "self_healing")
        standards: Optional explicit standards (auto-detected if None)
        description: Optional test description

    Returns:
        LayerMapping with all relevant layers, standards, and axioms
    """
    # Get layers from category
    layers = get_layers_for_category(category)

    # If no category match, try to infer from test name
    if not layers:
        name_lower = test_name.lower()
        for cat, cat_layers in TEST_CATEGORY_MAPPING.items():
            if cat.replace('_', ' ') in name_lower or cat in name_lower:
                layers = layers.union(cat_layers)

    # Default to crypto integrity if still empty
    if not layers:
        layers = {SCBELayer.L9_CRYPTO_INTEGRITY}

    # Get axioms for layers
    axioms = get_axioms_for_layers(layers)

    # Get or infer standards
    if standards is None:
        standards = get_standards_for_layers(layers)

    return LayerMapping(
        test_name=test_name,
        test_id=test_id,
        category=category,
        layers=layers,
        standards=standards,
        axioms=axioms,
        description=description,
    )


# Pre-built mappings for the industry-grade test suite
INDUSTRY_TEST_MAPPINGS: Dict[str, LayerMapping] = {}

def _build_industry_mappings():
    """Build mappings for the industry-grade test suite."""
    global INDUSTRY_TEST_MAPPINGS

    # Self-Healing Tests (101-110)
    for i in range(101, 111):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Self-Healing Test {i}",
            test_id=f"test_{i}",
            category="self_healing",
            standards={ComplianceStandard.ISO_27001, ComplianceStandard.SOC2_TYPE2},
        )

    # Medical AI-to-AI Tests (111-125)
    for i in range(111, 126):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Medical AI Test {i}",
            test_id=f"test_{i}",
            category="medical_phi",
            standards={ComplianceStandard.HIPAA, ComplianceStandard.HITECH},
        )

    # Military-Grade Tests (126-140)
    for i in range(126, 141):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Military-Grade Test {i}",
            test_id=f"test_{i}",
            category="military_classification",
            standards={ComplianceStandard.NIST_800_53, ComplianceStandard.FIPS_140_3},
        )

    # Adversarial Tests (141-155)
    categories = [
        "replay_attack", "bit_flip", "timing_attack", "padding_oracle",
        "chosen_plaintext", "key_extraction", "bit_flip", "timing_attack",
        "padding_oracle", "chosen_plaintext", "key_extraction", "replay_attack",
        "bit_flip", "timing_attack", "padding_oracle"
    ]
    for i, cat in enumerate(categories, start=141):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Adversarial Test {i}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.NIST_800_53},
        )

_build_industry_mappings()
