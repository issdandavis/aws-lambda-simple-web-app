#!/usr/bin/env python3
"""
SCBE-AETHERMOORE Enterprise Test Runner

Run enterprise-grade tests for compliance and certification.

Usage:
    python run_enterprise_tests.py              # Run all tests
    python run_enterprise_tests.py --quick      # Run quick tests only
    python run_enterprise_tests.py --quantum    # Run quantum tests
    python run_enterprise_tests.py --compliance # Run compliance tests
    python run_enterprise_tests.py --report     # Generate HTML report
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime


def run_tests(args):
    """Run enterprise tests with specified options."""

    # Base pytest command
    cmd = ["python", "-m", "pytest", "tests/enterprise/", "-v"]

    # Add markers based on arguments
    if args.quantum:
        cmd.extend(["-m", "quantum"])
    elif args.ai_safety:
        cmd.extend(["-m", "ai_safety"])
    elif args.compliance:
        cmd.extend(["-m", "compliance"])
    elif args.stress:
        cmd.extend(["-m", "stress"])
    elif args.agentic:
        cmd.extend(["-m", "agentic"])

    # Quick mode - skip slow tests
    if args.quick:
        cmd.extend(["--ignore=tests/enterprise/stress/"])

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=symphonic_cipher", "--cov-report=html"])

    # Add HTML report if requested
    if args.report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd.extend([f"--html=reports/enterprise_test_report_{timestamp}.html"])

    # Verbose output
    if args.verbose:
        cmd.append("-vv")

    # Show test output
    cmd.append("-s")

    # Run the tests
    print(f"\n{'='*60}")
    print("SCBE-AETHERMOORE Enterprise Test Suite")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Install with: pip install pytest")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run SCBE-AETHERMOORE enterprise tests"
    )

    # Test category selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quantum", action="store_true",
                       help="Run quantum attack simulation tests")
    group.add_argument("--ai-safety", dest="ai_safety", action="store_true",
                       help="Run AI safety tests")
    group.add_argument("--compliance", action="store_true",
                       help="Run compliance tests (SOC2, ISO, FIPS)")
    group.add_argument("--stress", action="store_true",
                       help="Run stress and performance tests")
    group.add_argument("--agentic", action="store_true",
                       help="Run agentic coding tests")

    # Options
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true",
                        help="Generate coverage report")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML test report")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Ensure reports directory exists
    if args.report:
        os.makedirs("reports", exist_ok=True)

    # Run tests
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
