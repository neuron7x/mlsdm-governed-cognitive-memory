#!/usr/bin/env python3
"""
Production-grade offline bibliography validator for MLSDM repository.

Validates:
- CITATION.cff exists and has required fields
- docs/bibliography/REFERENCES.bib parses successfully
- BibTeX keys are unique
- Each entry has title + year + author + at least one of (doi, url, eprint)
- Year is 4 digits (1900-2099)
- DOI format is valid (basic regex)
- URLs use HTTPS protocol
- No forbidden content (TODO, example.com, placeholder text)
- BibTeX and APA files have 1:1 key mapping

No network requests are made.
Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Forbidden patterns (applied to CITATION.cff and REFERENCES.bib only, not APA)
FORBIDDEN_PATTERNS_STRICT = [
    r"\bTODO\b",
    r"\bTBD\b",
    r"\bFIXME\b",
    r"example\.com",
    r"\bplaceholder\b",
]

# Patterns that apply to CITATION.cff only (not bibliography files)
FORBIDDEN_PATTERNS_CFF = FORBIDDEN_PATTERNS_STRICT + [
    r"\.\.\.",  # Ellipsis placeholders (but allowed in APA author lists)
]

# DOI format regex (basic validation - must have prefix and suffix)
DOI_PATTERN = re.compile(r"^10\.\d{4,}/.+")

# Year range for validation
MIN_YEAR = 1850
MAX_YEAR = 2026  # current_year + 1

# Year regex (4 digits)
YEAR_PATTERN = re.compile(r"^\d{4}$")


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
    required_fields = ["cff-version", "title", "version", "authors", "license"]
    for field in required_fields:
        if field + ":" not in content:
            errors.append(f"CITATION.cff missing required field: {field}")

    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS_CFF:
        if re.search(pattern, content, re.IGNORECASE):
            errors.append(f"CITATION.cff contains forbidden pattern: {pattern}")

    return errors


def parse_bibtex_simple(content: str) -> tuple[list[dict], list[str]]:
    """
    Simple BibTeX parser that extracts entries.
    Returns (entries, errors) where entries is a list of dicts with key/fields.
    """
    entries: list[dict] = []
    errors: list[str] = []

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


def validate_doi(doi: str) -> bool:
    """Validate DOI format (basic check)."""
    return bool(DOI_PATTERN.match(doi))


def validate_year(year: str) -> tuple[bool, str]:
    """Validate year is 4 digits in reasonable range [1850..2026].
    
    Returns (is_valid, error_message).
    """
    # Handle n.d. for "no date" entries
    if year.lower() == "n.d.":
        return True, ""
    if not YEAR_PATTERN.match(year):
        return False, f"year '{year}' is not a 4-digit integer"
    year_int = int(year)
    if year_int < MIN_YEAR or year_int > MAX_YEAR:
        return False, f"year {year_int} outside valid range [{MIN_YEAR}..{MAX_YEAR}]"
    return True, ""


def validate_url(url: str) -> bool:
    """Validate URL uses HTTPS and is not example.com."""
    if not url.startswith("https://"):
        return False
    # Check for placeholder domains (intentional substring check for validation, not sanitization)
    url_lower = url.lower()
    if "example.com" in url_lower or "example.org" in url_lower:  # noqa: S105
        return False
    return True


def check_forbidden_content(content: str, context: str, patterns: list[str] | None = None) -> list[str]:
    """Check for forbidden patterns in content."""
    if patterns is None:
        patterns = FORBIDDEN_PATTERNS_STRICT
    errors = []
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            errors.append(f"{context} contains forbidden pattern: {pattern}")
    return errors


def check_bibtex(repo_root: Path) -> tuple[list[str], set[str]]:
    """Check that REFERENCES.bib is valid and entries meet requirements.
    
    Returns (errors, bib_keys) where bib_keys is the set of all BibTeX keys.
    """
    errors: list[str] = []
    bib_keys: set[str] = set()
    bib_path = repo_root / "docs" / "bibliography" / "REFERENCES.bib"

    if not bib_path.exists():
        errors.append(f"REFERENCES.bib not found at {bib_path}")
        return errors, bib_keys

    content = bib_path.read_text(encoding="utf-8")

    # Check for forbidden patterns
    errors.extend(check_forbidden_content(content, "REFERENCES.bib"))

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
        return errors, bib_keys

    # Check for unique keys
    seen_keys: set[str] = set()
    for entry in entries:
        key = entry["key"]
        if key in seen_keys:
            errors.append(f"Duplicate BibTeX key: {key}")
        seen_keys.add(key)
    bib_keys = seen_keys.copy()

    # Check each entry has required fields
    for entry in entries:
        key = entry["key"]
        fields = entry["fields"]

        # Must have title
        if "title" not in fields or not fields["title"]:
            errors.append(f"Entry '{key}' missing required field: title")

        # Must have author
        if "author" not in fields or not fields["author"]:
            errors.append(f"Entry '{key}' missing required field: author")

        # Must have year
        year = fields.get("year", "")
        if not year:
            errors.append(f"Entry '{key}' missing required field: year")
        else:
            year_valid, year_error = validate_year(year)
            if not year_valid:
                errors.append(f"Entry '{key}' has invalid year: {year_error}")

        # Must have at least one of: doi, url, eprint
        has_identifier = any(fields.get(f) for f in ["doi", "url", "eprint"])
        if not has_identifier:
            errors.append(f"Entry '{key}' must have at least one of: doi, url, eprint")

        # Validate DOI format if present
        doi = fields.get("doi", "")
        if doi and not validate_doi(doi):
            errors.append(f"Entry '{key}' has invalid DOI format: {doi}")

        # Validate URL if present
        url = fields.get("url", "")
        if url and not validate_url(url):
            errors.append(f"Entry '{key}' has invalid URL (must be HTTPS, not example.com): {url}")

    print(f"Validated {len(entries)} BibTeX entries")
    return errors, bib_keys


def extract_apa_keys(repo_root: Path) -> tuple[list[str], set[str]]:
    """Extract BibTeX key comments from APA file.
    
    Returns (errors, apa_keys) where apa_keys is the set of all keys found.
    """
    errors: list[str] = []
    apa_keys: set[str] = set()
    apa_path = repo_root / "docs" / "bibliography" / "REFERENCES_APA7.md"

    if not apa_path.exists():
        errors.append(f"REFERENCES_APA7.md not found at {apa_path}")
        return errors, apa_keys

    content = apa_path.read_text(encoding="utf-8")

    # Check for forbidden patterns
    errors.extend(check_forbidden_content(content, "REFERENCES_APA7.md"))

    # Extract keys from HTML comments: <!-- key: bibkey -->
    key_pattern = re.compile(r"<!--\s*key:\s*(\S+)\s*-->")
    for match in key_pattern.finditer(content):
        key = match.group(1)
        if key in apa_keys:
            errors.append(f"Duplicate APA key comment: {key}")
        apa_keys.add(key)

    print(f"Found {len(apa_keys)} key comments in APA file")
    return errors, apa_keys


def check_bib_apa_consistency(bib_keys: set[str], apa_keys: set[str]) -> list[str]:
    """Check 1:1 mapping between BibTeX and APA keys."""
    errors: list[str] = []

    # Keys in BibTeX but not in APA
    missing_in_apa = bib_keys - apa_keys
    for key in sorted(missing_in_apa):
        errors.append(f"BibTeX key '{key}' has no corresponding APA entry (add <!-- key: {key} --> comment)")

    # Keys in APA but not in BibTeX
    missing_in_bib = apa_keys - bib_keys
    for key in sorted(missing_in_bib):
        errors.append(f"APA key '{key}' has no corresponding BibTeX entry")

    return errors


def main() -> int:
    """Run all validation checks."""
    repo_root = find_repo_root()
    print(f"Repository root: {repo_root}")

    all_errors: list[str] = []

    # Check CITATION.cff
    print("\n[1/4] Checking CITATION.cff...")
    cff_errors = check_citation_cff(repo_root)
    all_errors.extend(cff_errors)
    if cff_errors:
        for err in cff_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: CITATION.cff is valid")

    # Check REFERENCES.bib
    print("\n[2/4] Checking REFERENCES.bib...")
    bib_errors, bib_keys = check_bibtex(repo_root)
    all_errors.extend(bib_errors)
    if bib_errors:
        for err in bib_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: REFERENCES.bib is valid")

    # Check REFERENCES_APA7.md
    print("\n[3/4] Checking REFERENCES_APA7.md...")
    apa_errors, apa_keys = extract_apa_keys(repo_root)
    all_errors.extend(apa_errors)
    if apa_errors:
        for err in apa_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: REFERENCES_APA7.md is valid")

    # Check BibTeX-APA consistency
    print("\n[4/4] Checking BibTeX-APA consistency...")
    consistency_errors = check_bib_apa_consistency(bib_keys, apa_keys)
    all_errors.extend(consistency_errors)
    if consistency_errors:
        for err in consistency_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: BibTeX and APA files are consistent")

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
