#!/usr/bin/env python3
"""
Production-grade offline bibliography validator for MLSDM repository.

Validates:
- CITATION.cff exists and has required fields
- docs/bibliography/REFERENCES.bib parses successfully
- BibTeX keys are unique
- Each entry has title + year + author + at least one of (doi, url, eprint)
- Year is 4 digits within range [1850..2026]
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

ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ALLOWED_EVIDENCE_TYPES = {"peer_reviewed", "preprint", "standard"}


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


def _has_odd_trailing_backslashes(buffer: list[str]) -> bool:
    """Return True if buffer ends with an odd number of backslashes."""
    count = 0
    idx = len(buffer) - 1
    while idx >= 0 and buffer[idx] == "\\":
        count += 1
        idx -= 1
    return count % 2 == 1


def _split_fields(fields_str: str) -> list[str]:
    """Split a BibTeX fields block into individual field strings."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_quotes = False

    for ch in fields_str:
        if ch == '"' and not _has_odd_trailing_backslashes(current):
            in_quotes = not in_quotes
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
        if ch == "," and depth == 0 and not in_quotes:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)

    if current and "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def parse_bibtex_entries(content: str) -> tuple[list[dict], list[str]]:
    """
    Robust, dependency-free BibTeX parser that handles:
    - nested braces (e.g., author={{OpenAI}})
    - commas inside braced values
    - last field without trailing comma
    """
    entries: list[dict] = []
    errors: list[str] = []
    idx = 0

    while True:
        at_idx = content.find("@", idx)
        if at_idx == -1:
            break

        match = re.match(r"@(\w+)\s*\{", content[at_idx:])
        if not match:
            idx = at_idx + 1
            continue

        entry_type = match.group(1).lower()
        cursor = at_idx + match.end()

        while cursor < len(content) and content[cursor].isspace():
            cursor += 1

        key_start = cursor
        while cursor < len(content) and content[cursor] not in {",", "\n", " "}:
            cursor += 1
        entry_key = content[key_start:cursor].strip()

        # Move to start of fields (after first comma)
        while cursor < len(content) and content[cursor] != ",":
            cursor += 1
        if cursor >= len(content):
            errors.append(f"Malformed BibTeX entry near index {at_idx}: missing field list")
            break
        cursor += 1  # skip comma
        body_start = cursor
        brace_depth = 1
        while cursor < len(content) and brace_depth > 0:
            ch = content[cursor]
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
            cursor += 1

        if brace_depth != 0:
            errors.append(f"Unbalanced braces while parsing entry '{entry_key or 'unknown'}'")
            break

        fields_block = content[body_start : cursor - 1]
        fields: dict[str, str] = {}
        for field_str in _split_fields(fields_block):
            if "=" not in field_str:
                errors.append(f"Entry '{entry_key}': could not parse field '{field_str}'")
                continue
            name_raw, value_raw = field_str.split("=", 1)
            field_name = name_raw.strip().lower()
            value = value_raw.strip().rstrip(",")
            if (value.startswith("{") and value.endswith("}")) or (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]
            fields[field_name] = value.strip()

        entries.append({"type": entry_type, "key": entry_key, "fields": fields})
        idx = cursor

    return entries, errors


def run_parser_self_checks() -> list[str]:
    """Internal self-checks to ensure parser robustness."""
    errors: list[str] = []
    sample = """
@misc{selfcheck_one,
  author={{OpenAI}},
  title={ISO/IEC 42001:2023 — Artificial intelligence management system},
  url={https://example.com/path?with=comma,inside}
}

@article{selfcheck_two,
  author = {Example, Author},
  title = {Nested {brace} sample},
  year = {2024},
  note = {Trailing field without comma}
}
"""
    entries, parse_errors = parse_bibtex_entries(sample)
    errors.extend(parse_errors)
    if len(entries) != 2:
        errors.append(f"Self-check expected 2 entries, found {len(entries)}")
        return errors

    first_fields = entries[0]["fields"]
    if first_fields.get("author") != "{OpenAI}":
        errors.append("Self-check failed: nested braces not preserved in author field")
    if first_fields.get("url") != "https://example.com/path?with=comma,inside":
        errors.append("Self-check failed: URL with comma not parsed correctly")

    second_fields = entries[1]["fields"]
    if second_fields.get("note") != "Trailing field without comma":
        errors.append("Self-check failed: last field without trailing comma was not captured")
    return errors


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

    entries, parse_errors = parse_bibtex_entries(content)
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


def parse_verification_table(repo_root: Path) -> tuple[list[str], list[dict]]:
    """Parse verification table rows from VERIFICATION.md."""
    errors: list[str] = []
    rows: list[dict] = []
    verification_path = repo_root / "docs" / "bibliography" / "VERIFICATION.md"

    if not verification_path.exists():
        return [f"VERIFICATION.md not found at {verification_path}"], rows

    content = verification_path.read_text(encoding="utf-8")
    table_started = False
    for line in content.splitlines():
        if line.strip().startswith("| key"):
            table_started = True
            continue
        if not table_started:
            continue
        if line.strip().startswith("|---"):
            continue
        if not line.strip().startswith("|"):
            # stop when table section ends
            if table_started:
                break
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        row = {
            "key": cells[0],
            "category": cells[1],
            "evidence_type": cells[2],
            "canonical_id": cells[3],
            "canonical_url": cells[4],
            "verification_method": cells[5],
            "verified_on": cells[6],
            "notes": cells[7],
        }
        rows.append(row)

    if not rows:
        errors.append("No rows parsed from VERIFICATION.md table")
    return errors, rows


def check_verification_table(repo_root: Path, bib_keys: set[str]) -> list[str]:
    """Ensure verification table covers all BibTeX keys exactly once and is well-formed."""
    errors: list[str] = []
    parse_errors, rows = parse_verification_table(repo_root)
    errors.extend(parse_errors)
    if parse_errors:
        return errors

    table_keys: set[str] = set()
    for row in rows:
        key = row["key"]
        if key in table_keys:
            errors.append(f"Duplicate key in VERIFICATION.md: {key}")
        table_keys.add(key)

        if row["evidence_type"] not in ALLOWED_EVIDENCE_TYPES:
            errors.append(
                f"Row '{key}' has invalid evidence_type '{row['evidence_type']}' "
                f"(allowed: {sorted(ALLOWED_EVIDENCE_TYPES)})"
            )
        if not row["canonical_url"].startswith("https://"):
            errors.append(f"Row '{key}' canonical_url must use https: {row['canonical_url']}")
        if not row["canonical_id"]:
            errors.append(f"Row '{key}' missing canonical_id")
        if not ISO_DATE_PATTERN.match(row["verified_on"]):
            errors.append(f"Row '{key}' verified_on must be ISO date (YYYY-MM-DD)")

    missing = bib_keys - table_keys
    extra = table_keys - bib_keys
    if missing:
        for key in sorted(missing):
            errors.append(f"BibTeX key '{key}' missing from VERIFICATION.md table")
    if extra:
        for key in sorted(extra):
            errors.append(f"VERIFICATION.md contains key not present in BibTeX: {key}")

    return errors


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

    print("\n[1/6] Running parser self-checks...")
    parser_errors = run_parser_self_checks()
    all_errors.extend(parser_errors)
    if parser_errors:
        for err in parser_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: parser self-checks passed")

    # Check CITATION.cff
    print("\n[2/6] Checking CITATION.cff...")
    cff_errors = check_citation_cff(repo_root)
    all_errors.extend(cff_errors)
    if cff_errors:
        for err in cff_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: CITATION.cff is valid")

    # Check REFERENCES.bib
    print("\n[3/6] Checking REFERENCES.bib...")
    bib_errors, bib_keys = check_bibtex(repo_root)
    all_errors.extend(bib_errors)
    if bib_errors:
        for err in bib_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: REFERENCES.bib is valid")

    # Check REFERENCES_APA7.md
    print("\n[4/6] Checking REFERENCES_APA7.md...")
    apa_errors, apa_keys = extract_apa_keys(repo_root)
    all_errors.extend(apa_errors)
    if apa_errors:
        for err in apa_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: REFERENCES_APA7.md is valid")

    # Check BibTeX-APA consistency
    print("\n[5/6] Checking BibTeX-APA consistency...")
    consistency_errors = check_bib_apa_consistency(bib_keys, apa_keys)
    all_errors.extend(consistency_errors)
    if consistency_errors:
        for err in consistency_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: BibTeX and APA files are consistent")

    print("\n[6/6] Checking VERIFICATION.md coverage...")
    verification_errors = check_verification_table(repo_root, bib_keys)
    all_errors.extend(verification_errors)
    if verification_errors:
        for err in verification_errors:
            print(f"  ERROR: {err}")
    else:
        print("  OK: VERIFICATION.md covers all BibTeX keys")

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
