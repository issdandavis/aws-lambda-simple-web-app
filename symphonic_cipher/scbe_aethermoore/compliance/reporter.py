"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLIANCE REPORT GENERATOR                               ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Automatic compliance report generation for audits                           ║
║                                                                              ║
║  Report Formats:                                                             ║
║  - JSON: Machine-readable for automation                                     ║
║  - Markdown: Human-readable documentation                                    ║
║  - HTML: Rich formatted reports                                              ║
║  - CSV: Spreadsheet-compatible data export                                   ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .layer_mapping import (
    SCBELayer,
    ComplianceStandard,
    LAYER_DESCRIPTIONS,
    STANDARD_LAYER_MAPPING,
)
from .metadata import (
    ComplianceMetadata,
    TestResult,
    TestOutcome,
    RiskLevel,
    get_test_results,
    get_compliance_summary,
    get_results_by_standard,
    get_results_by_layer,
)


class ReportFormat(Enum):
    """Available report formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


@dataclass
class ComplianceReport:
    """Complete compliance report."""
    title: str
    generated_at: datetime
    summary: Dict[str, Any]
    results: List[TestResult]
    standards_coverage: Dict[str, Dict[str, Any]]
    layer_coverage: Dict[str, Dict[str, Any]]
    failures: List[Dict[str, Any]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'generated_at': self.generated_at.isoformat(),
            'summary': self.summary,
            'results': [r.to_dict() for r in self.results],
            'standards_coverage': self.standards_coverage,
            'layer_coverage': self.layer_coverage,
            'failures': self.failures,
            'recommendations': self.recommendations,
        }


class ComplianceReporter:
    """
    Generate compliance reports from test results.

    Usage:
        reporter = ComplianceReporter()
        report = reporter.generate_report()
        reporter.export(report, "compliance_report.md", ReportFormat.MARKDOWN)
    """

    def __init__(self, title: str = "SCBE Compliance Report"):
        self.title = title

    def generate_report(
        self,
        results: Optional[List[TestResult]] = None,
        standards: Optional[Set[ComplianceStandard]] = None,
    ) -> ComplianceReport:
        """
        Generate a comprehensive compliance report.

        Args:
            results: Test results (uses global registry if None)
            standards: Filter to specific standards (all if None)

        Returns:
            Complete ComplianceReport
        """
        if results is None:
            results = get_test_results()

        if standards:
            results = [
                r for r in results
                if r.metadata.standards & standards
            ]

        summary = self._generate_summary(results)
        standards_coverage = self._analyze_standards_coverage(results)
        layer_coverage = self._analyze_layer_coverage(results)
        failures = self._extract_failures(results)
        recommendations = self._generate_recommendations(results, failures)

        return ComplianceReport(
            title=self.title,
            generated_at=datetime.utcnow(),
            summary=summary,
            results=results,
            standards_coverage=standards_coverage,
            layer_coverage=layer_coverage,
            failures=failures,
            recommendations=recommendations,
        )

    def _generate_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate executive summary."""
        total = len(results)
        if total == 0:
            return {'total': 0, 'status': 'NO_DATA'}

        passed = sum(1 for r in results if r.outcome == TestOutcome.PASSED)
        failed = sum(1 for r in results if r.outcome == TestOutcome.FAILED)
        errors = sum(1 for r in results if r.outcome == TestOutcome.ERROR)
        skipped = sum(1 for r in results if r.outcome == TestOutcome.SKIPPED)

        pass_rate = passed / total
        status = "COMPLIANT" if pass_rate >= 0.95 else "NON_COMPLIANT" if pass_rate < 0.80 else "NEEDS_REVIEW"

        # Calculate risk exposure
        high_risk_failures = sum(
            1 for r in results
            if r.outcome == TestOutcome.FAILED and r.metadata.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
        )

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'pass_rate': round(pass_rate * 100, 2),
            'status': status,
            'high_risk_failures': high_risk_failures,
            'total_duration_ms': sum(r.duration_ms for r in results),
        }

    def _analyze_standards_coverage(self, results: List[TestResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze coverage per compliance standard."""
        coverage = {}

        for standard in ComplianceStandard:
            std_results = [r for r in results if standard in r.metadata.standards]
            if not std_results:
                continue

            passed = sum(1 for r in std_results if r.is_passing)
            total = len(std_results)

            coverage[standard.value] = {
                'standard': standard.value,
                'standard_name': standard.name,
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'pass_rate': round(passed / total * 100, 2) if total > 0 else 0,
                'status': 'PASS' if passed == total else 'FAIL',
                'required_layers': [l.name for l in STANDARD_LAYER_MAPPING.get(standard, set())],
            }

        return coverage

    def _analyze_layer_coverage(self, results: List[TestResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze coverage per SCBE layer."""
        coverage = {}

        for layer in SCBELayer:
            layer_results = [r for r in results if layer in r.metadata.layers]
            if not layer_results:
                continue

            passed = sum(1 for r in layer_results if r.is_passing)
            total = len(layer_results)
            layer_desc = LAYER_DESCRIPTIONS.get(layer, {})

            coverage[layer.name] = {
                'layer': layer.name,
                'layer_number': layer.value,
                'layer_name': layer_desc.get('name', layer.name),
                'purpose': layer_desc.get('purpose', ''),
                'axiom': layer_desc.get('axiom', ''),
                'formula': layer_desc.get('formula', ''),
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'pass_rate': round(passed / total * 100, 2) if total > 0 else 0,
            }

        return coverage

    def _extract_failures(self, results: List[TestResult]) -> List[Dict[str, Any]]:
        """Extract failure details."""
        failures = []

        for result in results:
            if result.requires_attention:
                failures.append({
                    'test_id': result.metadata.test_id,
                    'test_name': result.metadata.test_name,
                    'outcome': result.outcome.value,
                    'category': result.metadata.category,
                    'risk_level': result.metadata.risk_level.value,
                    'standards': [s.value for s in result.metadata.standards],
                    'layers': [l.name for l in result.metadata.layers],
                    'error_message': result.error_message,
                    'duration_ms': result.duration_ms,
                })

        # Sort by risk level
        risk_order = {RiskLevel.CRITICAL.value: 0, RiskLevel.HIGH.value: 1, RiskLevel.MEDIUM.value: 2, RiskLevel.LOW.value: 3}
        failures.sort(key=lambda f: risk_order.get(f['risk_level'], 99))

        return failures

    def _generate_recommendations(self, results: List[TestResult], failures: List[Dict]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if not failures:
            recommendations.append("All compliance tests passed. Maintain current security posture.")
            return recommendations

        # Analyze failure patterns
        failed_layers = set()
        failed_standards = set()
        critical_failures = 0

        for f in failures:
            failed_layers.update(f['layers'])
            failed_standards.update(f['standards'])
            if f['risk_level'] in ['critical', 'high']:
                critical_failures += 1

        if critical_failures > 0:
            recommendations.append(
                f"URGENT: Address {critical_failures} critical/high-risk failures immediately."
            )

        # Layer-specific recommendations
        if 'L9_CRYPTO_INTEGRITY' in failed_layers:
            recommendations.append(
                "Review cryptographic implementations. Ensure HMAC/MAC verification is constant-time."
            )

        if 'L5_QUANTUM_COHERENCE' in failed_layers:
            recommendations.append(
                "PQC validation failures detected. Verify Kyber/Dilithium implementation."
            )

        if 'L6_SESSION_KEY' in failed_layers:
            recommendations.append(
                "Session key establishment issues. Review key derivation and exchange protocols."
            )

        if 'L10_TEMPORAL_CONSISTENCY' in failed_layers:
            recommendations.append(
                "Temporal validation failures. Check timestamp handling and replay protection."
            )

        # Standard-specific recommendations
        if 'hipaa' in failed_standards or 'hitech' in failed_standards:
            recommendations.append(
                "HIPAA/HITECH failures require immediate remediation for PHI protection."
            )

        if 'nist_800_53' in failed_standards or 'fips_140_3' in failed_standards:
            recommendations.append(
                "NIST/FIPS failures may affect federal compliance. Review cryptographic module."
            )

        if 'pci_dss' in failed_standards:
            recommendations.append(
                "PCI-DSS failures affect payment data security. Review encryption of CHD."
            )

        return recommendations

    def export(
        self,
        report: ComplianceReport,
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """
        Export report to file.

        Args:
            report: The compliance report
            output_path: Output file path
            format: Output format

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        content = self._format_report(report, format)

        with open(output_path, 'w') as f:
            f.write(content)

        return str(output_path)

    def _format_report(self, report: ComplianceReport, format: ReportFormat) -> str:
        """Format report for output."""
        if format == ReportFormat.JSON:
            return json.dumps(report.to_dict(), indent=2)
        elif format == ReportFormat.MARKDOWN:
            return self._format_markdown(report)
        elif format == ReportFormat.HTML:
            return self._format_html(report)
        elif format == ReportFormat.CSV:
            return self._format_csv(report)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _format_markdown(self, report: ComplianceReport) -> str:
        """Format as Markdown."""
        lines = []

        # Header
        lines.append(f"# {report.title}")
        lines.append(f"\n**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        s = report.summary
        status_emoji = {"COMPLIANT": "✅", "NEEDS_REVIEW": "⚠️", "NON_COMPLIANT": "❌"}.get(s.get('status', ''), '❓')
        lines.append(f"**Status:** {status_emoji} {s.get('status', 'UNKNOWN')}")
        lines.append(f"**Pass Rate:** {s.get('pass_rate', 0)}%")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Tests | {s.get('total_tests', 0)} |")
        lines.append(f"| Passed | {s.get('passed', 0)} |")
        lines.append(f"| Failed | {s.get('failed', 0)} |")
        lines.append(f"| Errors | {s.get('errors', 0)} |")
        lines.append(f"| High-Risk Failures | {s.get('high_risk_failures', 0)} |")
        lines.append(f"| Total Duration | {s.get('total_duration_ms', 0):.2f}ms |")
        lines.append("")

        # Standards Coverage
        lines.append("## Standards Coverage")
        lines.append("")
        lines.append("| Standard | Tests | Passed | Pass Rate | Status |")
        lines.append("|----------|-------|--------|-----------|--------|")
        for std_name, std_data in report.standards_coverage.items():
            status_icon = "✅" if std_data['status'] == 'PASS' else "❌"
            lines.append(
                f"| {std_data['standard_name']} | {std_data['total_tests']} | "
                f"{std_data['passed']} | {std_data['pass_rate']}% | {status_icon} |"
            )
        lines.append("")

        # Layer Coverage
        lines.append("## SCBE Layer Coverage")
        lines.append("")
        lines.append("| Layer | Name | Tests | Passed | Axiom |")
        lines.append("|-------|------|-------|--------|-------|")
        for layer_name, layer_data in sorted(report.layer_coverage.items(), key=lambda x: x[1]['layer_number']):
            lines.append(
                f"| L{layer_data['layer_number']} | {layer_data['layer_name']} | "
                f"{layer_data['total_tests']} | {layer_data['passed']} | {layer_data['axiom']} |"
            )
        lines.append("")

        # Failures
        if report.failures:
            lines.append("## Failures")
            lines.append("")
            for f in report.failures:
                risk_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(f['risk_level'], "⚪")
                lines.append(f"### {risk_icon} {f['test_name']}")
                lines.append(f"- **Test ID:** `{f['test_id']}`")
                lines.append(f"- **Risk Level:** {f['risk_level'].upper()}")
                lines.append(f"- **Standards:** {', '.join(f['standards'])}")
                lines.append(f"- **Layers:** {', '.join(f['layers'])}")
                if f['error_message']:
                    lines.append(f"- **Error:** `{f['error_message'][:200]}`")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by SCBE Compliance Framework*")

        return "\n".join(lines)

    def _format_html(self, report: ComplianceReport) -> str:
        """Format as HTML."""
        md_content = self._format_markdown(report)
        # Simple HTML wrapper (could use a proper template)
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: green; }} .fail {{ color: red; }}
        h1 {{ color: #333; }} h2 {{ color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
<pre style="white-space: pre-wrap;">{md_content}</pre>
</body>
</html>"""

    def _format_csv(self, report: ComplianceReport) -> str:
        """Format as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'test_id', 'test_name', 'outcome', 'category', 'risk_level',
            'standards', 'layers', 'duration_ms', 'error_message'
        ])

        # Data
        for result in report.results:
            writer.writerow([
                result.metadata.test_id,
                result.metadata.test_name,
                result.outcome.value,
                result.metadata.category,
                result.metadata.risk_level.value,
                ';'.join(s.value for s in result.metadata.standards),
                ';'.join(l.name for l in result.metadata.layers),
                result.duration_ms,
                result.error_message,
            ])

        return output.getvalue()


def generate_compliance_report(
    title: str = "SCBE Compliance Report",
    results: Optional[List[TestResult]] = None,
    standards: Optional[Set[ComplianceStandard]] = None,
    output_path: Optional[Union[str, Path]] = None,
    format: ReportFormat = ReportFormat.MARKDOWN,
) -> Union[ComplianceReport, str]:
    """
    Quick function to generate a compliance report.

    Args:
        title: Report title
        results: Test results (uses global registry if None)
        standards: Filter to specific standards
        output_path: If provided, exports to file and returns path
        format: Output format (if exporting)

    Returns:
        ComplianceReport if no output_path, otherwise path to exported file
    """
    reporter = ComplianceReporter(title)
    report = reporter.generate_report(results, standards)

    if output_path:
        return reporter.export(report, output_path, format)

    return report


def quick_audit(
    results: Optional[List[TestResult]] = None,
) -> Dict[str, Any]:
    """
    Quick audit check returning pass/fail status.

    Returns:
        Dict with 'compliant' bool and summary
    """
    if results is None:
        results = get_test_results()

    if not results:
        return {
            'compliant': True,
            'message': 'No tests to audit',
            'pass_rate': 100.0,
        }

    passed = sum(1 for r in results if r.is_passing)
    total = len(results)
    pass_rate = (passed / total) * 100

    critical_failures = sum(
        1 for r in results
        if r.requires_attention and r.metadata.risk_level in {RiskLevel.CRITICAL, RiskLevel.HIGH}
    )

    return {
        'compliant': pass_rate >= 95.0 and critical_failures == 0,
        'pass_rate': round(pass_rate, 2),
        'total_tests': total,
        'passed': passed,
        'failed': total - passed,
        'critical_failures': critical_failures,
        'message': 'COMPLIANT' if pass_rate >= 95.0 and critical_failures == 0 else 'NON-COMPLIANT',
    }
