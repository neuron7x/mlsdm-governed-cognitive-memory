#!/usr/bin/env python3
"""
MLSDM Policy Configuration Validator
=====================================
Validates that policy YAML files are consistent with actual CI workflows,
code structure, and test locations.

Usage:
    python scripts/validate_policy_config.py
    python scripts/validate_policy_config.py --policy-dir policy/

Exit codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)


class PolicyValidator:
    """Validates policy configuration files against repository reality."""

    def __init__(self, repo_root: Path, policy_dir: Path):
        self.repo_root = repo_root
        self.policy_dir = policy_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("=" * 70)
        print("MLSDM Policy Configuration Validation")
        print("=" * 70)
        print()

        # Load policy files
        security_policy = self._load_yaml(self.policy_dir / "security-baseline.yaml")
        slo_policy = self._load_yaml(self.policy_dir / "observability-slo.yaml")

        if not security_policy or not slo_policy:
            print("\n❌ FAILED: Could not load policy files")
            return False

        # Run validation checks
        self._validate_security_workflows(security_policy)
        self._validate_security_modules(security_policy)
        self._validate_slo_tests(slo_policy)
        self._validate_documentation(security_policy, slo_policy)

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _load_yaml(self, path: Path) -> Dict | None:
        """Load and parse YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.errors.append(f"Policy file not found: {path}")
            return None
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error in {path}: {e}")
            return None

    def _validate_security_workflows(self, policy: Dict) -> None:
        """Validate that required CI workflows exist."""
        print("CHECK: Security Workflow Files")
        print("-" * 70)

        required_checks = policy.get("required_checks", [])
        workflows_dir = self.repo_root / ".github" / "workflows"

        for check in required_checks:
            check_name = check.get("name")
            workflow_file = check.get("workflow_file")

            if workflow_file:
                workflow_path = self.repo_root / workflow_file
                if workflow_path.exists():
                    print(f"✓ {check_name}: {workflow_file} exists")
                else:
                    self.errors.append(
                        f"{check_name}: Workflow file not found: {workflow_file}"
                    )
                    print(f"✗ {check_name}: {workflow_file} NOT FOUND")
            elif check.get("command"):
                # Command-based check, verify the command is valid
                command = check.get("command")
                print(f"✓ {check_name}: Command-based check '{command}'")
            elif check.get("script"):
                # Script-based check
                script = check.get("script")
                script_path = self.repo_root / script.lstrip("./")
                if script_path.exists():
                    print(f"✓ {check_name}: {script} exists")
                else:
                    self.errors.append(f"{check_name}: Script not found: {script}")
                    print(f"✗ {check_name}: {script} NOT FOUND")

        print()

    def _validate_security_modules(self, policy: Dict) -> None:
        """Validate that referenced security modules exist."""
        print("CHECK: Security Module References")
        print("-" * 70)

        security_reqs = policy.get("security_requirements", {})
        
        # Check input validation modules
        input_val = security_reqs.get("input_validation", {})
        llm_safety_module = input_val.get("llm_safety_module")
        scrubber_module = input_val.get("payload_scrubber_module")

        modules_to_check = [
            ("LLM Safety Gateway", llm_safety_module),
            ("Payload Scrubber", scrubber_module),
        ]

        for name, module_path in modules_to_check:
            if module_path:
                # Convert module path to file path
                # e.g., mlsdm.security.llm_safety -> src/mlsdm/security/llm_safety.py
                parts = module_path.split(".")
                if parts[0] == "mlsdm":
                    file_path = self.repo_root / "src" / "/".join(parts)
                    py_path = file_path.parent / f"{file_path.name}.py"
                    init_path = file_path / "__init__.py"

                    if py_path.exists() or init_path.exists():
                        print(f"✓ {name}: {module_path} exists")
                    else:
                        self.warnings.append(
                            f"{name}: Module {module_path} not found (checked {py_path} and {init_path})"
                        )
                        print(f"⚠ {name}: {module_path} NOT FOUND (warning)")

        print()

    def _validate_slo_tests(self, policy: Dict) -> None:
        """Validate that SLO test locations exist."""
        print("CHECK: SLO Test Locations")
        print("-" * 70)

        slos = policy.get("slos", {})
        
        test_locations = []
        
        # Collect test locations from API endpoints
        for endpoint in slos.get("api_endpoints", []):
            if "test_location" in endpoint:
                test_locations.append((endpoint["name"], endpoint["test_location"]))
        
        # Collect test locations from system resources
        for resource in slos.get("system_resources", []):
            if "test_location" in resource:
                test_locations.append((resource["name"], resource["test_location"]))
        
        # Collect test locations from cognitive engine
        for component in slos.get("cognitive_engine", []):
            if "test_location" in component:
                test_locations.append((component["name"], component["test_location"]))

        for name, test_loc in test_locations:
            # Extract file path (before ::)
            if "::" in test_loc:
                file_path, test_name = test_loc.split("::", 1)
            else:
                file_path = test_loc
                test_name = None

            full_path = self.repo_root / file_path
            
            if full_path.exists():
                print(f"✓ {name}: {file_path} exists")
                # Could further validate that the test name exists in the file
            else:
                self.errors.append(f"{name}: Test file not found: {file_path}")
                print(f"✗ {name}: {file_path} NOT FOUND")

        print()

    def _validate_documentation(self, security_policy: Dict, slo_policy: Dict) -> None:
        """Validate that referenced documentation exists."""
        print("CHECK: Documentation Files")
        print("-" * 70)

        # Documentation from SLO policy
        docs = slo_policy.get("documentation", {})
        doc_files = [
            ("SLO Spec", docs.get("slo_spec")),
            ("Validation Protocol", docs.get("validation_protocol")),
            ("Runbook", docs.get("runbook")),
            ("Observability Guide", docs.get("observability_guide")),
        ]

        for name, doc_file in doc_files:
            if doc_file:
                doc_path = self.repo_root / doc_file
                if doc_path.exists():
                    print(f"✓ {name}: {doc_file} exists")
                else:
                    self.warnings.append(f"{name}: Documentation not found: {doc_file}")
                    print(f"⚠ {name}: {doc_file} NOT FOUND (warning)")

        print()

    def _print_results(self) -> None:
        """Print validation results summary."""
        print("=" * 70)
        print("Validation Summary")
        print("=" * 70)
        print(f"Errors:   {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  ❌ {error}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
            print()

        if len(self.errors) == 0:
            print("✓ All critical validations passed!")
        else:
            print(f"✗ Validation failed with {len(self.errors)} error(s)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate MLSDM policy configuration files"
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("policy"),
        help="Path to policy directory (default: policy/)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory (default: current directory)",
    )

    args = parser.parse_args()

    # Resolve paths
    repo_root = args.repo_root.resolve()
    policy_dir = (repo_root / args.policy_dir).resolve()

    if not policy_dir.exists():
        print(f"ERROR: Policy directory not found: {policy_dir}")
        return 1

    # Run validation
    validator = PolicyValidator(repo_root, policy_dir)
    success = validator.validate_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
