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
    # ==========================================================================
    # 1. Self-Healing Workflow (Tests 101-110)
    # ==========================================================================
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
    "retry_logic": {
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },
    "exponential_backoff": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    "health_metrics": {
        SCBELayer.L4_ENTROPY_FLOW,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },

    # ==========================================================================
    # 2. Medical AI-to-AI Communication (Tests 111-125) - HIPAA
    # ==========================================================================
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
    "patient_id_hashing": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    "multi_ai_chain": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },

    # ==========================================================================
    # 3. Military-Grade Security (Tests 126-140) - NIST 800-53 / FIPS 140-3
    # ==========================================================================
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
    "classification_levels": {
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "nist_controls": {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },

    # ==========================================================================
    # 4. Adversarial Attack Resistance (Tests 141-155)
    # ==========================================================================
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
    "injection_attack": {
        SCBELayer.L8_BOUNDARY_PROXIMITY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },

    # ==========================================================================
    # 5. Quantum Resistant Crypto (Tests 156-170)
    # ==========================================================================
    "kyber_kem": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
    },
    "dilithium_signature": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    "hybrid_key_derivation": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    "pqc_session": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
    },

    # ==========================================================================
    # 6. Chaos Engineering / Fault Injection (Tests 171-185)
    # ==========================================================================
    "chaos_engineering": {
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L8_BOUNDARY_PROXIMITY,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    "random_failure": {
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    "network_partition": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    "memory_pressure": {
        SCBELayer.L4_ENTROPY_FLOW,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },

    # ==========================================================================
    # 7. Performance & Scalability (Tests 186-195)
    # ==========================================================================
    "performance": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    "throughput": {
        SCBELayer.L12_ENERGY_CONSERVATION,
    },
    "latency": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },
    "concurrent_load": {
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L12_ENERGY_CONSERVATION,
    },

    # ==========================================================================
    # 8. Compliance Audit (Tests 196-210)
    # ==========================================================================
    "compliance_audit": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "hipaa_audit": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "pci_dss_audit": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    "sox_audit": {
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "gdpr_audit": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "iso27001_audit": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },

    # ==========================================================================
    # 9. Financial Critical Infrastructure (Tests 211-220) - PCI-DSS
    # ==========================================================================
    "financial_critical": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    "transaction_integrity": {
        SCBELayer.L9_CRYPTO_INTEGRITY,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },
    "hsm_simulation": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },

    # ==========================================================================
    # 10. AI-to-AI Multi-Agent (Tests 221-235)
    # ==========================================================================
    "ai_multi_agent": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L13_DECISION_BOUNDARY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "federated_learning": {
        SCBELayer.L5_QUANTUM_COHERENCE,
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L9_CRYPTO_INTEGRITY,
    },
    "robotic_surgery": {
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    "swarm_coordination": {
        SCBELayer.L6_SESSION_KEY,
        SCBELayer.L7_TRAJECTORY_SMOOTH,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
    },

    # ==========================================================================
    # 11. Zero-Trust Defense-in-Depth (Tests 236-250)
    # ==========================================================================
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
    "microsegmentation": {
        SCBELayer.L8_BOUNDARY_PROXIMITY,
        SCBELayer.L13_DECISION_BOUNDARY,
    },
    "continuous_verification": {
        SCBELayer.L1_AXIOM_VERIFIER,
        SCBELayer.L10_TEMPORAL_CONSISTENCY,
        SCBELayer.L14_GOVERNANCE_DECISION,
    },
    "least_privilege": {
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
    """Build mappings for the industry-grade test suite (150 tests: 101-250)."""
    global INDUSTRY_TEST_MAPPINGS

    # ==========================================================================
    # 1. Self-Healing Workflow (Tests 101-110)
    # ==========================================================================
    self_healing_cats = [
        "self_healing", "circuit_breaker", "retry_logic", "exponential_backoff",
        "health_metrics", "self_healing", "circuit_breaker", "retry_logic",
        "exponential_backoff", "health_metrics"
    ]
    for i, cat in enumerate(self_healing_cats, start=101):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Self-Healing: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.ISO_27001, ComplianceStandard.SOC2_TYPE2},
        )

    # ==========================================================================
    # 2. Medical AI-to-AI Communication (Tests 111-125) - HIPAA
    # ==========================================================================
    medical_cats = [
        "medical_phi", "medical_phi", "medical_phi", "patient_id_hashing",
        "patient_id_hashing", "medical_audit", "medical_audit", "multi_ai_chain",
        "multi_ai_chain", "medical_phi", "medical_phi", "patient_id_hashing",
        "medical_audit", "multi_ai_chain", "medical_phi"
    ]
    for i, cat in enumerate(medical_cats, start=111):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Medical AI: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.HIPAA, ComplianceStandard.HITECH},
        )

    # ==========================================================================
    # 3. Military-Grade Security (Tests 126-140) - NIST 800-53 / FIPS 140-3
    # ==========================================================================
    military_cats = [
        "military_classification", "military_classification", "military_key_rotation",
        "classification_levels", "nist_controls", "military_classification",
        "military_key_rotation", "classification_levels", "nist_controls",
        "military_classification", "military_key_rotation", "classification_levels",
        "nist_controls", "military_classification", "nist_controls"
    ]
    for i, cat in enumerate(military_cats, start=126):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Military-Grade: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.NIST_800_53, ComplianceStandard.FIPS_140_3},
        )

    # ==========================================================================
    # 4. Adversarial Attack Resistance (Tests 141-155)
    # ==========================================================================
    adversarial_cats = [
        "replay_attack", "bit_flip", "timing_attack", "padding_oracle",
        "chosen_plaintext", "key_extraction", "injection_attack", "bit_flip",
        "timing_attack", "padding_oracle", "chosen_plaintext", "key_extraction",
        "replay_attack", "timing_attack", "injection_attack"
    ]
    for i, cat in enumerate(adversarial_cats, start=141):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Adversarial: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.NIST_800_53},
        )

    # ==========================================================================
    # 5. Quantum Resistant Crypto (Tests 156-170)
    # ==========================================================================
    pqc_cats = [
        "kyber_kem", "kyber_kem", "dilithium_signature", "dilithium_signature",
        "hybrid_key_derivation", "hybrid_key_derivation", "pqc_session",
        "pqc_session", "kyber_kem", "dilithium_signature", "hybrid_key_derivation",
        "pqc_session", "kyber_kem", "dilithium_signature", "hybrid_key_derivation"
    ]
    for i, cat in enumerate(pqc_cats, start=156):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"PQC: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.NIST_800_53, ComplianceStandard.FIPS_140_3},
        )

    # ==========================================================================
    # 6. Chaos Engineering / Fault Injection (Tests 171-185)
    # ==========================================================================
    chaos_cats = [
        "chaos_engineering", "random_failure", "network_partition", "memory_pressure",
        "chaos_engineering", "random_failure", "network_partition", "memory_pressure",
        "chaos_engineering", "random_failure", "network_partition", "memory_pressure",
        "chaos_engineering", "random_failure", "network_partition"
    ]
    for i, cat in enumerate(chaos_cats, start=171):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Chaos: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.ISO_27001, ComplianceStandard.SOC2_TYPE2},
        )

    # ==========================================================================
    # 7. Performance & Scalability (Tests 186-195)
    # ==========================================================================
    perf_cats = [
        "performance", "throughput", "latency", "concurrent_load",
        "performance", "throughput", "latency", "concurrent_load",
        "performance", "throughput"
    ]
    for i, cat in enumerate(perf_cats, start=186):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Performance: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.SOC2_TYPE2},
        )

    # ==========================================================================
    # 8. Compliance Audit (Tests 196-210)
    # ==========================================================================
    audit_cats = [
        "hipaa_audit", "hipaa_audit", "pci_dss_audit", "pci_dss_audit",
        "sox_audit", "gdpr_audit", "iso27001_audit", "hipaa_audit",
        "pci_dss_audit", "sox_audit", "gdpr_audit", "iso27001_audit",
        "compliance_audit", "compliance_audit", "compliance_audit"
    ]
    audit_standards = {
        "hipaa_audit": {ComplianceStandard.HIPAA, ComplianceStandard.HITECH},
        "pci_dss_audit": {ComplianceStandard.PCI_DSS},
        "sox_audit": {ComplianceStandard.SOC2_TYPE2},
        "gdpr_audit": {ComplianceStandard.ISO_27001},
        "iso27001_audit": {ComplianceStandard.ISO_27001},
        "compliance_audit": {ComplianceStandard.ISO_27001, ComplianceStandard.SOC2_TYPE2},
    }
    for i, cat in enumerate(audit_cats, start=196):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Audit: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards=audit_standards.get(cat, {ComplianceStandard.ISO_27001}),
        )

    # ==========================================================================
    # 9. Financial Critical Infrastructure (Tests 211-220) - PCI-DSS
    # ==========================================================================
    financial_cats = [
        "financial_critical", "transaction_integrity", "hsm_simulation",
        "financial_critical", "transaction_integrity", "hsm_simulation",
        "financial_critical", "transaction_integrity", "hsm_simulation",
        "financial_critical"
    ]
    for i, cat in enumerate(financial_cats, start=211):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Financial: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.PCI_DSS, ComplianceStandard.SOC2_TYPE2},
        )

    # ==========================================================================
    # 10. AI-to-AI Multi-Agent (Tests 221-235)
    # ==========================================================================
    ai_cats = [
        "ai_multi_agent", "federated_learning", "robotic_surgery", "swarm_coordination",
        "ai_multi_agent", "federated_learning", "robotic_surgery", "swarm_coordination",
        "ai_multi_agent", "federated_learning", "robotic_surgery", "swarm_coordination",
        "ai_multi_agent", "federated_learning", "robotic_surgery"
    ]
    for i, cat in enumerate(ai_cats, start=221):
        stds = {ComplianceStandard.ISO_27001}
        if cat == "robotic_surgery":
            stds.add(ComplianceStandard.HIPAA)
        if cat == "federated_learning":
            stds.add(ComplianceStandard.NIST_800_53)
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Multi-Agent: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards=stds,
        )

    # ==========================================================================
    # 11. Zero-Trust Defense-in-Depth (Tests 236-250)
    # ==========================================================================
    zt_cats = [
        "zero_trust", "microsegmentation", "continuous_verification", "least_privilege",
        "zero_trust", "microsegmentation", "continuous_verification", "least_privilege",
        "zero_trust", "microsegmentation", "continuous_verification", "least_privilege",
        "zero_trust", "microsegmentation", "zero_trust"
    ]
    for i, cat in enumerate(zt_cats, start=236):
        INDUSTRY_TEST_MAPPINGS[f"test_{i}"] = get_layer_for_test(
            test_name=f"Zero-Trust: {cat.replace('_', ' ').title()}",
            test_id=f"test_{i}",
            category=cat,
            standards={ComplianceStandard.NIST_800_53, ComplianceStandard.ISO_27001},
        )

_build_industry_mappings()
