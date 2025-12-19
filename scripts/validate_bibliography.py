#!/usr/bin/env python3
"""
Offline bibliography validator for MLSDM repository.

Validates:
- CITATION.cff exists
- docs/bibliography/REFERENCES.bib parses successfully
- BibTeX keys are unique
- Each entry has title + year + at least one of (doi, url, eprint)

No network requests are made.
Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find repository root by looking for CITATION.cff or pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        if (current / "CITATION.cff").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: assume script is in scripts/ directory
    return Path(__file__).resolve().parent.parent


def check_citation_cff(repo_root: Path) -> list[str]:
    """Check that CITATION.cff exists and has required fields."""
    errors: list[str] = []
    cff_path = repo_root / "CITATION.cff"

    if not cff_path.exists():
        errors.append(f"CITATION.cff not found at {cff_path}")
        return errors

    content = cff_path.read_text(encoding="utf-8")

    # Basic checks for required fields
    required_fields = ["cff-version", "title", "version", "authors"]
    for field in required_fields:
        if field not in content:
            errors.append(f"CITATION.cff missing required field: {field}")

    return errors


def parse_bibtex_simple(content: str) -> tuple[list[dict], list[str]]:
    """
    Simple BibTeX parser that extracts entries.
    Returns (entries, errors) where entries is a list of dicts with key/fields.
    """
    entries: list[dict] = []
    errors: list[str] = []

    # Find all entry blocks: @type{key, ... }
    # Pattern to match BibTeX entries
    entry_pattern = re.compile(
        r"@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]*?)(?=\n@|\Z)", re.DOTALL | re.MULTILINE
    )

    for match in entry_pattern.finditer(content):
        entry_type = match.group(1).lower()
        entry_key = match.group(2).strip()
        fields_str = match.group(3)

        entry = {
            "type": entry_type,
            "key": entry_key,
            "fields": {},
        }

        # Parse fields: field = {value} or field = "value"
        field_pattern = re.compile(r"(\w+)\s*=\s*[\{\"](.*?)[\}\"]", re.DOTALL)
        for field_match in field_pattern.finditer(fields_str):
            field_name = field_match.group(1).lower()
            field_value = field_match.group(2).strip()
            entry["fields"][field_name] = field_value

        entries.append(entry)

    return entries, errors


def check_bibtex(repo_root: Path) -> list[str]:
    """Check that REFERENCES.bib is valid and entries meet requirements."""
    errors: list[str] = []
    bib_path = repo_root / "docs" / "bibliography" / "REFERENCES.bib"

    if not bib_path.exists():
        errors.append(f"REFERENCES.bib not found at {bib_path}")
        return errors

    content = bib_path.read_text(encoding="utf-8")

    # Try to use bibtexparser if available, otherwise use simple parser
    try:
        import bibtexparser

        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_database = bibtexparser.loads(content, parser=parser)
        entries = []
        for entry in bib_database.entries:
            entries.append(
                {
                    "type": entry.get("ENTRYTYPE", "unknown"),
                    "key": entry.get("ID", "unknown"),
                    "fields": {k.lower(): v for k, v in entry.items()},
                }
            )
    except ImportError:
        # Fallback to simple parser
        entries, parse_errors = parse_bibtex_simple(content)
        errors.extend(parse_errors)

    if not entries:
        errors.append("No BibTeX entries found in REFERENCES.bib")
        return errors

    # Check for unique keys
    keys = [e["key"] for e in entries]
    seen_keys: set[str] = set()
    for key in keys:
        if key in seen_keys:
            errors.append(f"Duplicate BibTeX key: {key}")
        seen_keys.add(key)

    # Check each entry has required fields
    for entry in entries:
        key = entry["key"]
        fields = entry["fields"]

        # Must have title
        if "title" not in fields or not fields["title"]:
            errors.append(f"Entry '{key}' missing required field: title")

        # Must have year
        if "year" not in fields or not fields["year"]:
            errors.append(f"Entry '{key}' missing required field: year")

        # Must have at least one of: doi, url, eprint
        has_identifier = any(
            fields.get(f) for f in ["doi", "url", "eprint"]
        )
        if not has_identifier:
            errors.append(
                f"Entry '{key}' must have at least one of: doi, url, eprint"
            )

    print(f"Validated {len(entries)} BibTeX entries")
    return errors


def main() -> int:
    """Run all validation checks."""
    repo_root = find_repo_root()
    print(f"Repository root: {repo_root}")

    all_errors: list[str] = []

    # Check CITATION.cff
    print("\n[1/2] Checking CITATION.cff...")
    cff_errors = check_citation_cff(repo_root)
    all_errors.extend(cff_errors)
    if cff_errors:
        for err in cff_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: CITATION.cff is valid")

    # Check REFERENCES.bib
    print("\n[2/2] Checking REFERENCES.bib...")
    bib_errors = check_bibtex(repo_root)
    all_errors.extend(bib_errors)
    if bib_errors:
        for err in bib_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: REFERENCES.bib is valid")

    # Summary
    print("\n" + "=" * 50)
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s) found")
        return 1
    else:
        print("PASSED: All bibliography checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
