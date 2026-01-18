"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCBE COMPLIANCE FRAMEWORK                                 ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Industry-Grade Compliance Validation & Reporting                            ║
║                                                                              ║
║  Standards Supported:                                                        ║
║  - HIPAA/HITECH (Medical AI Communication)                                   ║
║  - NIST 800-53 / FIPS 140-3 (Military/Government)                            ║
║  - PCI-DSS (Financial)                                                       ║
║  - IEC 62443 (Industrial Control Systems)                                    ║
║  - ISO 27001 / SOC 2 Type II (Enterprise Security)                           ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Author: Isaac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .layer_mapping import (
    SCBELayer,
    ComplianceStandard,
    LayerMapping,
    get_layer_for_test,
    get_layers_for_standard,
    LAYER_DESCRIPTIONS,
)

from .metadata import (
    ComplianceMetadata,
    TestResult,
    compliance_test,
    extract_compliance_metadata,
)

from .reporter import (
    ComplianceReporter,
    ReportFormat,
    generate_compliance_report,
    quick_audit,
)

__all__ = [
    # Layer Mapping
    'SCBELayer',
    'ComplianceStandard',
    'LayerMapping',
    'get_layer_for_test',
    'get_layers_for_standard',
    'LAYER_DESCRIPTIONS',
    # Metadata
    'ComplianceMetadata',
    'TestResult',
    'compliance_test',
    'extract_compliance_metadata',
    # Reporter
    'ComplianceReporter',
    'ReportFormat',
    'generate_compliance_report',
    'quick_audit',
]
