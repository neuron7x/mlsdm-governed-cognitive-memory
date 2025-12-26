"""Deterministic readiness change log generator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from . import change_analyzer as ca

ROOT = Path(__file__).resolve().parents[2]


def _ensure_no_bidi(text: str, label: str) -> None:
    ca._ensure_no_bidi(text, label)  # type: ignore[attr-defined]


def _collect_changed_files(base_ref: str, root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{base_ref}..HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unable to compute git diff")
    paths = [ca.normalize_path(line) for line in result.stdout.splitlines() if ca.normalize_path(line)]
    return sorted(dict.fromkeys(paths))


def _build_entry(title: str, date_str: str, base_ref: str, analysis: dict[str, object], paths: Sequence[str]) -> str:
    counts = analysis.get("counts", {})  # type: ignore[var-annotated]
    categories = json.dumps(counts.get("categories", {}), sort_keys=True, separators=(", ", ": "))
    risks = json.dumps(counts.get("risks", {}), sort_keys=True, separators=(", ", ": "))
    primary_category = analysis.get("primary_category", "functional_core")
    max_risk = analysis.get("max_risk", "info")
    files_fmt = ", ".join(f"`{p}`" for p in paths)
    return (
        f"- {date_str} — **{title}** — Base: {base_ref}\n"
        f"  - Changed files ({len(paths)}): {files_fmt}\n"
        f"  - Primary category: {primary_category}; Max risk: {max_risk}\n"
        f"  - Category counts: {categories}\n"
        f"  - Risk counts: {risks}"
    )


def _update_last_updated(content: str, date_str: str) -> str:
    lines = content.splitlines()
    updated = []
    replaced = False
    for line in lines:
        if line.startswith("Last updated:"):
            updated.append(f"Last updated: {date_str}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        raise FileNotFoundError("Last updated line not found in readiness document")
    return "\n".join(updated) + "\n"


def _insert_entry(content: str, entry: str) -> str:
    lines = content.splitlines()
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip() == "## Change Log")
    except StopIteration:
        raise FileNotFoundError("## Change Log section not found in readiness document")

    entry_lines = entry.splitlines()
    if entry_lines and lines[idx + 1 : idx + 1 + len(entry_lines)] == entry_lines:
        return "\n".join(lines) + "\n"

    new_lines = lines[: idx + 1] + entry_lines + lines[idx + 1 :]
    return "\n".join(new_lines) + "\n"


def generate_update(
    title: str,
    base_ref: str,
    root: Path = ROOT,
    *,
    diff_provider: Callable[[str, Path], Sequence[str]] | None = None,
    analyzer: Callable[[Sequence[str], str, Path], dict[str, object]] | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> tuple[Path, str]:
    _ensure_no_bidi(title, "title")
    now = now_provider() if now_provider else datetime.now(timezone.utc)
    date_str = now.date().isoformat()
    readiness_path = root / "docs" / "status" / "READINESS.md"
    if not readiness_path.exists():
        raise FileNotFoundError(f"Missing readiness file at {readiness_path}")

    provided_paths = diff_provider(base_ref, root) if diff_provider else _collect_changed_files(base_ref, root)
    for path in provided_paths:
        _ensure_no_bidi(path, "diff path")
    paths = sorted({ca.normalize_path(p) for p in provided_paths if ca.normalize_path(p)})
    analysis = (analyzer or ca.analyze_paths)(paths, base_ref=base_ref, root=root)

    entry = _build_entry(title, date_str, base_ref, analysis, paths)
    content = readiness_path.read_text(encoding="utf-8")
    _ensure_no_bidi(content, "docs/status/READINESS.md")
    updated = _update_last_updated(content, date_str)
    updated = _insert_entry(updated.rstrip("\n"), entry)
    return readiness_path, updated


def _write_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate readiness change log entry")
    parser.add_argument("--title", required=True, help="Change log entry title")
    parser.add_argument("--base-ref", default="origin/main", help="Git base reference (default: origin/main)")
    parser.add_argument(
        "--mode",
        choices=("preview", "apply"),
        default="preview",
        help="preview: print updated document; apply: write to docs/status/READINESS.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path, updated = generate_update(args.title, args.base_ref)
        if args.mode == "preview":
            print(updated, end="")
            return 0
        _write_atomic(path, updated)
        print(f"Wrote readiness update to {path}")
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
