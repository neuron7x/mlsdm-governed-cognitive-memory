#!/usr/bin/env python3
"""Generate evidence inputs JSON with null handling for missing files."""
import json
import sys
from pathlib import Path


def main() -> int:
    """Generate evidence inputs JSON for capture_evidence.py.

    Returns:
        0 on success, 1 on error
    """
    try:
        inputs = {
            "coverage_xml": "coverage.xml" if Path("coverage.xml").exists() else None,
            "coverage_log": "coverage-gate.log" if Path("coverage-gate.log").exists() else None,
            "junit_xml": "reports/junit.xml",
        }

        output_path = Path("/tmp/evidence-inputs.json")
        output_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")

        # Log to stderr so it doesn't interfere with JSON output
        print(f"✓ Generated {output_path}", file=sys.stderr)
        for key, value in inputs.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value or 'missing'}", file=sys.stderr)

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
