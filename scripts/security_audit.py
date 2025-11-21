#!/usr/bin/env python3
"""Security audit script for dependency vulnerability scanning.

This script performs security checks including:
1. Dependency vulnerability scanning using pip-audit
2. Security configuration validation
3. Security best practices checks

Usage:
    python scripts/security_audit.py [--fix] [--report]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def check_pip_audit_installed() -> bool:
    """Check if pip-audit is installed."""
    try:
        subprocess.run(
            ["pip-audit", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_pip_audit() -> bool:
    """Install pip-audit if not present."""
    print("Installing pip-audit...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pip-audit"],
            capture_output=True,
            check=True
        )
        print("✓ pip-audit installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install pip-audit: {e}")
        return False


def run_pip_audit(fix: bool = False) -> tuple[bool, dict]:
    """Run pip-audit to check for vulnerabilities.
    
    Args:
        fix: If True, attempt to fix vulnerabilities by upgrading packages
        
    Returns:
        Tuple of (success, results_dict)
    """
    print("\n" + "=" * 60)
    print("Running dependency vulnerability scan with pip-audit...")
    print("=" * 60)

    cmd = ["pip-audit", "--format", "json"]
    if fix:
        cmd.append("--fix")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # Parse JSON output
        if result.stdout:
            try:
                audit_results = json.loads(result.stdout)
            except json.JSONDecodeError:
                audit_results = {"error": "Failed to parse pip-audit output"}
        else:
            audit_results = {}

        if result.returncode == 0:
            print("✓ No vulnerabilities found!")
            return True, audit_results
        else:
            print(f"✗ Found vulnerabilities (exit code: {result.returncode})")
            if audit_results.get("dependencies"):
                for dep in audit_results["dependencies"]:
                    print(f"\n  Package: {dep.get('name', 'unknown')}")
                    print(f"  Version: {dep.get('version', 'unknown')}")
                    for vuln in dep.get("vulns", []):
                        print(f"    - {vuln.get('id', 'N/A')}: {vuln.get('description', 'No description')}")
                        print(f"      Fix: {vuln.get('fix_versions', 'No fix available')}")
            return False, audit_results

    except FileNotFoundError:
        print("✗ pip-audit not found")
        return False, {"error": "pip-audit not installed"}
    except Exception as e:
        print(f"✗ Error running pip-audit: {e}")
        return False, {"error": str(e)}


def check_security_configs() -> bool:
    """Check security configuration files are present and valid.
    
    Returns:
        True if all checks pass
    """
    print("\n" + "=" * 60)
    print("Checking security configuration files...")
    print("=" * 60)

    required_files = [
        "SECURITY_POLICY.md",
        "THREAT_MODEL.md"
    ]

    all_present = True
    for file in required_files:
        filepath = Path(file)
        if filepath.exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_present = False

    return all_present


def check_security_implementations() -> tuple[bool, list[str]]:
    """Check that security features are implemented.
    
    Returns:
        Tuple of (all_present, list_of_findings)
    """
    print("\n" + "=" * 60)
    print("Checking security feature implementations...")
    print("=" * 60)

    findings = []

    # Check for rate limiter
    rate_limiter_file = Path("src/utils/rate_limiter.py")
    if rate_limiter_file.exists():
        print("✓ Rate limiter implemented")
    else:
        print("✗ Rate limiter not found")
        findings.append("Missing: Rate limiter implementation")

    # Check for input validator
    validator_file = Path("src/utils/input_validator.py")
    if validator_file.exists():
        print("✓ Input validator implemented")
    else:
        print("✗ Input validator not found")
        findings.append("Missing: Input validator implementation")

    # Check for security logger
    logger_file = Path("src/utils/security_logger.py")
    if logger_file.exists():
        print("✓ Security logger implemented")
    else:
        print("✗ Security logger not found")
        findings.append("Missing: Security logger implementation")

    # Check for security tests
    test_file = Path("src/tests/unit/test_security.py")
    if test_file.exists():
        print("✓ Security tests present")
    else:
        print("✗ Security tests not found")
        findings.append("Missing: Security tests")

    return len(findings) == 0, findings


def generate_security_report(
    audit_results: dict,
    config_check: bool,
    impl_check: bool,
    impl_findings: list[str]
) -> str:
    """Generate security audit report.
    
    Args:
        audit_results: Results from pip-audit
        config_check: Whether config files check passed
        impl_check: Whether implementation check passed
        impl_findings: List of implementation findings
        
    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("SECURITY AUDIT REPORT")
    report.append("=" * 60)
    report.append("")

    # Summary
    report.append("SUMMARY")
    report.append("-" * 60)

    vuln_count = 0
    if audit_results.get("dependencies"):
        for dep in audit_results["dependencies"]:
            vuln_count += len(dep.get("vulns", []))

    report.append(f"Dependencies scanned: {len(audit_results.get('dependencies', []))}")
    report.append(f"Vulnerabilities found: {vuln_count}")
    report.append(f"Configuration files: {'✓ Present' if config_check else '✗ Missing'}")
    report.append(f"Security implementations: {'✓ Complete' if impl_check else '✗ Incomplete'}")
    report.append("")

    # Details
    if vuln_count > 0:
        report.append("VULNERABILITIES")
        report.append("-" * 60)
        for dep in audit_results.get("dependencies", []):
            for vuln in dep.get("vulns", []):
                report.append(f"• {dep['name']} {dep['version']}")
                report.append(f"  {vuln.get('id', 'N/A')}: {vuln.get('description', 'No description')}")
                report.append(f"  Fix: {vuln.get('fix_versions', 'No fix available')}")
                report.append("")

    if impl_findings:
        report.append("IMPLEMENTATION FINDINGS")
        report.append("-" * 60)
        for finding in impl_findings:
            report.append(f"• {finding}")
        report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 60)
    if vuln_count > 0:
        report.append("• Run 'python scripts/security_audit.py --fix' to attempt automatic fixes")
    if not config_check:
        report.append("• Create missing security configuration files")
    if impl_findings:
        report.append("• Implement missing security features")
    if vuln_count == 0 and config_check and impl_check:
        report.append("• All security checks passed! ✓")

    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Security audit for MLSDM Cognitive Memory")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix vulnerabilities by upgrading packages"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Save report to file"
    )
    args = parser.parse_args()

    print("MLSDM Governed Cognitive Memory - Security Audit")
    print("=" * 60)

    # Ensure pip-audit is installed
    if not check_pip_audit_installed():
        print("pip-audit not found. Attempting to install...")
        if not install_pip_audit():
            print("\nERROR: Could not install pip-audit")
            print("Please install manually: pip install pip-audit")
            return 1

    # Run checks
    audit_success, audit_results = run_pip_audit(fix=args.fix)
    config_check = check_security_configs()
    impl_check, impl_findings = check_security_implementations()

    # Generate report
    report = generate_security_report(
        audit_results,
        config_check,
        impl_check,
        impl_findings
    )

    print("\n" + report)

    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to {args.report}")

    # Exit with appropriate code
    all_passed = audit_success and config_check and impl_check
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
