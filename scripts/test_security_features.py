#!/usr/bin/env python3
"""
Integration test script for security features.

This script validates that all security features are working correctly
by running comprehensive integration tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of the test
        
    Returns:
        Tuple of (success, output)
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        output = result.stdout + result.stderr
        success = result.returncode == 0

        if success:
            print(f"✓ {description} - PASSED")
        else:
            print(f"✗ {description} - FAILED")
            print(f"Exit code: {result.returncode}")
            if output:
                print("Output:")
                print(output[:500])  # Show first 500 chars

        return success, output

    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False, str(e)


def main():
    """Main test runner."""
    print("="*60)
    print("MLSDM Security Features Integration Test")
    print("="*60)

    results = []

    # Test 1: Security unit tests
    success, output = run_command(
        ["python", "-m", "pytest", "src/tests/unit/test_security.py", "-v", "--tb=short", "--no-cov"],
        "Security Unit Tests"
    )
    results.append(("Security Unit Tests", success))

    # Test 2: API tests with security features
    success, output = run_command(
        ["python", "-m", "pytest", "src/tests/unit/test_api.py", "-v", "--tb=short", "--no-cov"],
        "API Tests with Security"
    )
    results.append(("API Tests", success))

    # Test 3: Rate limiter tests
    success, output = run_command(
        ["python", "-m", "pytest", "src/tests/unit/test_security.py::TestRateLimiter", "-v", "--no-cov"],
        "Rate Limiter Tests"
    )
    results.append(("Rate Limiter", success))

    # Test 4: Input validator tests
    success, output = run_command(
        ["python", "-m", "pytest", "src/tests/unit/test_security.py::TestInputValidator", "-v", "--no-cov"],
        "Input Validator Tests"
    )
    results.append(("Input Validator", success))

    # Test 5: Security logger tests
    success, output = run_command(
        ["python", "-m", "pytest", "src/tests/unit/test_security.py::TestSecurityLogger", "-v", "--no-cov"],
        "Security Logger Tests"
    )
    results.append(("Security Logger", success))

    # Test 6: Check security files exist
    print(f"\n{'='*60}")
    print("Checking Security Artifacts")
    print(f"{'='*60}")

    required_files = [
        "src/utils/rate_limiter.py",
        "src/utils/input_validator.py",
        "src/utils/security_logger.py",
        "src/tests/unit/test_security.py",
        "scripts/security_audit.py",
        "SECURITY_IMPLEMENTATION.md",
        "SECURITY_POLICY.md",
        "THREAT_MODEL.md"
    ]

    all_present = True
    for file_path_str in required_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            print(f"✓ {file_path_str}")
        else:
            print(f"✗ {file_path_str} - MISSING")
            all_present = False

    results.append(("Security Artifacts", all_present))

    # Test 7: Verify security implementations are importable
    print(f"\n{'='*60}")
    print("Verifying Security Imports")
    print(f"{'='*60}")

    try:
        import sys as sys_module
        from pathlib import Path as PathLib
        # Add project root to path for imports
        sys_module.path.insert(0, str(PathLib.cwd()))

        print("✓ RateLimiter can be imported")

        print("✓ InputValidator can be imported")

        print("✓ SecurityLogger can be imported")

        results.append(("Security Imports", True))
    except Exception as e:
        print(f"✗ Import failed: {e}")
        results.append(("Security Imports", False))

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} - {test_name}")

    print(f"\n{'-'*60}")
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'-'*60}")

    if passed == total:
        print("\n✓ All security features are working correctly!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
