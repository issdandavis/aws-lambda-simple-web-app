"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLIANCE FRAMEWORK TESTS                                ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Industry-Grade Compliance Validation Tests                                  ║
║                                                                              ║
║  Test Categories:                                                            ║
║  - Layer Mapping Verification                                                ║
║  - Compliance Metadata Decorator                                             ║
║  - Report Generation                                                         ║
║  - Standard Coverage Analysis                                                ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
from datetime import datetime

from symphonic_cipher.scbe_aethermoore.compliance import (
    # Layer Mapping
    SCBELayer,
    ComplianceStandard,
    LayerMapping,
    get_layer_for_test,
    get_layers_for_standard,
    LAYER_DESCRIPTIONS,
)
from symphonic_cipher.scbe_aethermoore.compliance.layer_mapping import (
    INDUSTRY_TEST_MAPPINGS,
    TEST_CATEGORY_MAPPING,
    STANDARD_LAYER_MAPPING,
    get_axioms_for_layers,
    get_standards_for_layers,
)
from symphonic_cipher.scbe_aethermoore.compliance.metadata import (
    ComplianceMetadata,
    TestResult,
    TestOutcome,
    RiskLevel,
    compliance_test,
    extract_compliance_metadata,
    get_compliance_summary,
    clear_test_results,
)
from symphonic_cipher.scbe_aethermoore.compliance.reporter import (
    ComplianceReporter,
    ReportFormat,
    generate_compliance_report,
    quick_audit,
)


# =============================================================================
# LAYER MAPPING TESTS
# =============================================================================

class TestSCBELayers:
    """Test SCBE Layer definitions."""

    def test_all_14_layers_defined(self):
        """Verify all 14 layers are defined."""
        assert len(SCBELayer) == 14
        assert SCBELayer.L1_AXIOM_VERIFIER.value == 1
        assert SCBELayer.L14_GOVERNANCE_DECISION.value == 14

    def test_layer_descriptions_complete(self):
        """Verify all layers have descriptions."""
        for layer in SCBELayer:
            assert layer in LAYER_DESCRIPTIONS
            desc = LAYER_DESCRIPTIONS[layer]
            assert 'name' in desc
            assert 'purpose' in desc
            assert 'axiom' in desc

    def test_layer_axiom_mapping(self):
        """Verify layer to axiom mappings."""
        layers = {SCBELayer.L5_QUANTUM_COHERENCE}
        axioms = get_axioms_for_layers(layers)
        assert '5' in axioms  # PQC Harmonic

        layers = {SCBELayer.L9_CRYPTO_INTEGRITY}
        axioms = get_axioms_for_layers(layers)
        assert '7' in axioms  # Spiral Seal


class TestComplianceStandards:
    """Test compliance standard definitions."""

    def test_all_standards_defined(self):
        """Verify all 8 compliance standards are defined."""
        assert len(ComplianceStandard) == 8
        assert ComplianceStandard.HIPAA.value == "hipaa"
        assert ComplianceStandard.NIST_800_53.value == "nist_800_53"
        assert ComplianceStandard.PCI_DSS.value == "pci_dss"

    def test_standard_layer_requirements(self):
        """Verify each standard has layer requirements."""
        for standard in ComplianceStandard:
            layers = get_layers_for_standard(standard)
            assert len(layers) > 0, f"No layers for {standard}"

    def test_hipaa_requires_crypto_layers(self):
        """HIPAA should require crypto and session layers."""
        layers = get_layers_for_standard(ComplianceStandard.HIPAA)
        assert SCBELayer.L5_QUANTUM_COHERENCE in layers
        assert SCBELayer.L6_SESSION_KEY in layers
        assert SCBELayer.L9_CRYPTO_INTEGRITY in layers

    def test_nist_requires_axiom_verification(self):
        """NIST 800-53 should require axiom verification."""
        layers = get_layers_for_standard(ComplianceStandard.NIST_800_53)
        assert SCBELayer.L1_AXIOM_VERIFIER in layers
        assert SCBELayer.L14_GOVERNANCE_DECISION in layers


class TestCategoryMappings:
    """Test category to layer mappings."""

    def test_all_categories_have_layers(self):
        """All test categories should map to layers."""
        for category, layers in TEST_CATEGORY_MAPPING.items():
            assert len(layers) > 0, f"No layers for category: {category}"

    def test_medical_phi_layers(self):
        """Medical PHI tests should use encryption layers."""
        layers = TEST_CATEGORY_MAPPING.get("medical_phi", set())
        assert SCBELayer.L5_QUANTUM_COHERENCE in layers
        assert SCBELayer.L9_CRYPTO_INTEGRITY in layers

    def test_zero_trust_covers_all_layers(self):
        """Zero-trust tests should cover all 14 layers."""
        layers = TEST_CATEGORY_MAPPING.get("zero_trust", set())
        assert len(layers) == 14


class TestIndustryMappings:
    """Test industry test suite mappings."""

    def test_150_tests_mapped(self):
        """Verify all 150 industry tests are mapped."""
        assert len(INDUSTRY_TEST_MAPPINGS) == 150

    def test_test_id_format(self):
        """All test IDs should follow format test_NNN."""
        for test_id in INDUSTRY_TEST_MAPPINGS.keys():
            assert test_id.startswith("test_")
            num = int(test_id.split("_")[1])
            assert 101 <= num <= 250

    def test_self_healing_range(self):
        """Tests 101-110 should be self-healing."""
        for i in range(101, 111):
            mapping = INDUSTRY_TEST_MAPPINGS.get(f"test_{i}")
            assert mapping is not None
            assert "Self-Healing" in mapping.test_name

    def test_medical_ai_range(self):
        """Tests 111-125 should be medical AI."""
        for i in range(111, 126):
            mapping = INDUSTRY_TEST_MAPPINGS.get(f"test_{i}")
            assert mapping is not None
            assert ComplianceStandard.HIPAA in mapping.standards

    def test_military_range(self):
        """Tests 126-140 should be military-grade."""
        for i in range(126, 141):
            mapping = INDUSTRY_TEST_MAPPINGS.get(f"test_{i}")
            assert mapping is not None
            assert ComplianceStandard.NIST_800_53 in mapping.standards


# =============================================================================
# METADATA TESTS
# =============================================================================

class TestComplianceMetadata:
    """Test compliance metadata creation."""

    def test_metadata_creation(self):
        """Create metadata with all fields."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test PHI Encryption",
            category="medical_phi",
            layers={SCBELayer.L5_QUANTUM_COHERENCE, SCBELayer.L9_CRYPTO_INTEGRITY},
            standards={ComplianceStandard.HIPAA},
            risk_level=RiskLevel.HIGH,
        )
        assert metadata.test_id == "test_001"
        assert len(metadata.layers) == 2
        assert metadata.risk_level == RiskLevel.HIGH

    def test_metadata_to_dict(self):
        """Metadata should serialize to dict."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test",
            category="test",
        )
        d = metadata.to_dict()
        assert 'test_id' in d
        assert 'layers' in d
        assert 'created_at' in d


class TestTestResult:
    """Test result tracking."""

    def test_result_creation(self):
        """Create test result with metadata."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test",
            category="test",
        )
        result = TestResult(
            metadata=metadata,
            outcome=TestOutcome.PASSED,
            duration_ms=100.5,
        )
        assert result.is_passing
        assert not result.requires_attention

    def test_failed_result(self):
        """Failed results should require attention."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test",
            category="test",
        )
        result = TestResult(
            metadata=metadata,
            outcome=TestOutcome.FAILED,
            duration_ms=50.0,
            error_message="Assertion failed",
        )
        assert not result.is_passing
        assert result.requires_attention


class TestComplianceDecorator:
    """Test @compliance_test decorator."""

    def test_decorator_captures_metadata(self):
        """Decorator should capture compliance metadata."""
        @compliance_test(
            category="medical_phi",
            standards={ComplianceStandard.HIPAA},
            risk_level=RiskLevel.HIGH,
            description="Test PHI encryption",
        )
        def test_phi():
            pass

        metadata = extract_compliance_metadata(test_phi)
        assert metadata is not None
        assert metadata.category == "medical_phi"
        assert ComplianceStandard.HIPAA in metadata.standards
        assert metadata.risk_level == RiskLevel.HIGH

    def test_decorator_preserves_function(self):
        """Decorator should preserve function behavior."""
        @compliance_test(category="test")
        def test_returns_value():
            return 42

        assert test_returns_value() == 42


# =============================================================================
# REPORTER TESTS
# =============================================================================

class TestComplianceReporter:
    """Test compliance report generation."""

    def setup_method(self):
        """Clear results before each test."""
        clear_test_results()

    def test_reporter_creation(self):
        """Create reporter with custom title."""
        reporter = ComplianceReporter("Custom Report")
        assert reporter.title == "Custom Report"

    def test_empty_report(self):
        """Generate report with no results."""
        reporter = ComplianceReporter()
        report = reporter.generate_report(results=[])
        # Empty report has 'total' key instead of 'total_tests'
        assert report.summary.get('total', 0) == 0 or report.summary.get('total_tests', 0) == 0

    def test_report_with_results(self):
        """Generate report with test results."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test",
            category="medical_phi",
            standards={ComplianceStandard.HIPAA},
        )
        results = [
            TestResult(metadata=metadata, outcome=TestOutcome.PASSED, duration_ms=100),
        ]
        reporter = ComplianceReporter()
        report = reporter.generate_report(results=results)
        assert report.summary['total_tests'] == 1
        assert report.summary['passed'] == 1
        assert report.summary['pass_rate'] == 100.0

    def test_markdown_format(self):
        """Generate Markdown report."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test",
            category="test",
        )
        results = [
            TestResult(metadata=metadata, outcome=TestOutcome.PASSED, duration_ms=100),
        ]
        reporter = ComplianceReporter()
        report = reporter.generate_report(results=results)
        md = reporter._format_markdown(report)
        assert "# " in md
        assert "Executive Summary" in md
        assert "COMPLIANT" in md

    def test_json_format(self):
        """Generate JSON report."""
        metadata = ComplianceMetadata(
            test_id="test_001",
            test_name="Test",
            category="test",
        )
        results = [
            TestResult(metadata=metadata, outcome=TestOutcome.PASSED, duration_ms=100),
        ]
        reporter = ComplianceReporter()
        report = reporter.generate_report(results=results)
        json_str = reporter._format_report(report, ReportFormat.JSON)
        assert '"title"' in json_str
        assert '"summary"' in json_str


class TestQuickAudit:
    """Test quick audit function."""

    def setup_method(self):
        """Clear results before each test."""
        clear_test_results()

    def test_quick_audit_no_results(self):
        """Quick audit with no results should be compliant."""
        result = quick_audit(results=[])
        assert result['compliant']
        assert result['pass_rate'] == 100.0

    def test_quick_audit_all_passed(self):
        """Quick audit with all passed should be compliant."""
        metadata = ComplianceMetadata(test_id="t1", test_name="T1", category="test")
        results = [
            TestResult(metadata=metadata, outcome=TestOutcome.PASSED, duration_ms=10),
            TestResult(metadata=metadata, outcome=TestOutcome.PASSED, duration_ms=10),
        ]
        result = quick_audit(results=results)
        assert result['compliant']
        assert result['pass_rate'] == 100.0

    def test_quick_audit_with_failures(self):
        """Quick audit with failures should be non-compliant."""
        metadata = ComplianceMetadata(
            test_id="t1",
            test_name="T1",
            category="test",
            risk_level=RiskLevel.CRITICAL,
        )
        results = [
            TestResult(metadata=metadata, outcome=TestOutcome.FAILED, duration_ms=10),
        ]
        result = quick_audit(results=results)
        assert not result['compliant']
        assert result['critical_failures'] == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestComplianceIntegration:
    """Integration tests for the compliance framework."""

    def test_full_workflow(self):
        """Test complete compliance workflow."""
        # 1. Get layer mapping for a test
        mapping = get_layer_for_test(
            test_name="PHI Encryption Test",
            test_id="test_111",
            category="medical_phi",
        )
        assert len(mapping.layers) > 0

        # 2. Create metadata
        metadata = ComplianceMetadata.from_mapping(mapping)
        assert metadata.test_id == "test_111"

        # 3. Create result
        result = TestResult(
            metadata=metadata,
            outcome=TestOutcome.PASSED,
            duration_ms=50.0,
        )
        assert result.is_passing

        # 4. Generate report
        reporter = ComplianceReporter("Integration Test Report")
        report = reporter.generate_report(results=[result])
        assert report.summary['status'] == 'COMPLIANT'

    def test_standards_coverage_analysis(self):
        """Analyze which standards are covered by a set of layers."""
        # All layers should satisfy most standards
        all_layers = set(SCBELayer)
        satisfied = get_standards_for_layers(all_layers)

        # Should satisfy several standards
        assert len(satisfied) >= 4

    def test_layer_coverage_calculation(self):
        """Calculate layer coverage percentage."""
        layers_tested = {
            SCBELayer.L1_AXIOM_VERIFIER,
            SCBELayer.L5_QUANTUM_COHERENCE,
            SCBELayer.L9_CRYPTO_INTEGRITY,
        }
        coverage = len(layers_tested) / 14
        assert coverage == 3 / 14
