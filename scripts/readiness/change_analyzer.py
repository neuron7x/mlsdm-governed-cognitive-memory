#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parent.parent.parent

_BIDI_PATTERN = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_SECURITY_MARKERS = (
    "moral_filter",
    "/security/",
    "auth",
    "crypto",
    "encryption",
    "permission",
)
_OBSERVABILITY_MARKERS = ("observability", "metrics", "logging", "tracing")

CATEGORY_PRIORITY: list[str] = [
    "security_critical",
    "test_coverage",
    "documentation",
    "infrastructure",
    "observability",
    "functional_core",
]

RISK_MAP: dict[str, str] = {
    "security_critical": "critical",
    "functional_core": "high",
    "infrastructure": "medium",
    "observability": "low",
    "test_coverage": "informational",
    "documentation": "informational",
}

RISK_ORDER: dict[str, int] = {
    "informational": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def normalize_path(path: str) -> str:
    """Normalize a file path to use forward slashes and drop redundant prefixes."""
    normalized = path.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip()


def _ensure_no_bidi(text: str, context: str) -> None:
    if _BIDI_PATTERN.search(text):
        raise ValueError(f"Bidirectional control character detected in {context}")


def classify_category(path: str) -> str:
    """Classify a path into a readiness category following priority rules."""
    normalized = normalize_path(path)
    lower = normalized.lower()
    name = Path(normalized).name

    if any(marker in lower for marker in _SECURITY_MARKERS):
        return "security_critical"
    if normalized.startswith("src/security/") or (normalized.startswith("src/") and "/security/" in normalized[4:]):
        return "security_critical"
    if normalized.startswith("tests/") or name.startswith("test_"):
        return "test_coverage"
    if normalized.startswith("docs/") or name.endswith((".md", ".rst", ".txt")):
        return "documentation"
    if normalized.startswith(".github/workflows/") or normalized.startswith("deploy/") or normalized.startswith(
        "config/"
    ):
        return "infrastructure"
    if any(marker in lower for marker in _OBSERVABILITY_MARKERS):
        return "observability"
    return "functional_core"


def risk_for_category(category: str) -> str:
    return RISK_MAP.get(category, "high")


def module_name(path: str) -> str:
    normalized = normalize_path(path)
    if normalized.startswith("src/"):
        normalized = normalized[4:]
    elif normalized.startswith("tests/"):
        normalized = normalized[6:]
    without_suffix = Path(normalized).with_suffix("")
    return ".".join(without_suffix.parts)


def _format_args(args: ast.arguments) -> str:
    parts: list[str] = []
    for arg in args.posonlyargs:
        parts.append(arg.arg)
    if args.posonlyargs:
        parts.append("/")
    for arg in args.args:
        parts.append(arg.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg in args.kwonlyargs:
        parts.append(arg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ",".join(parts)


def _format_return(node: ast.AST) -> str:
    returns = getattr(node, "returns", None)
    if returns is None:
        return "None"
    try:
        return ast.unparse(returns)
    except (TypeError, ValueError):
        return "None"


def _decorator_suffix(node: ast.AST) -> str:
    names: list[str] = []
    for decorator in getattr(node, "decorator_list", []):
        try:
            names.append(ast.unparse(decorator).strip())
        except (TypeError, ValueError):
            continue
    return f"|decorators={','.join(sorted(names))}" if names else ""


def _function_sig(
    node: ast.FunctionDef | ast.AsyncFunctionDef, module: str, class_name: str | None = None
) -> str:
    qualname = f"{class_name}.{node.name}" if class_name else node.name
    args = _format_args(node.args)
    ret = _format_return(node)
    return f"{module}:{qualname}({args})->{ret}{_decorator_suffix(node)}"


def _class_sig(node: ast.ClassDef, module: str) -> str:
    bases: list[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base).strip())
        except (TypeError, ValueError):
            continue
    suffix = f"[bases={','.join(sorted(bases))}]" if bases else ""
    return f"{module}:{node.name}{suffix}{_decorator_suffix(node)}"


def parse_python_signatures(source: str, module: str) -> dict[str, str]:
    """Extract canonical signatures for top-level functions/classes and class methods."""
    try:
        _ensure_no_bidi(source, module)
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    signatures: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = _function_sig(node, module)
        elif isinstance(node, ast.ClassDef):
            signatures[node.name] = _class_sig(node, module)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{node.name}.{child.name}"
                    signatures[key] = _function_sig(child, module, class_name=node.name)
    return signatures


def semantic_diff(before: str | None, after: str | None, module: str) -> dict[str, object]:
    before_sigs = parse_python_signatures(before, module) if before else {}
    after_sigs = parse_python_signatures(after, module) if after else {}
    before_keys = set(before_sigs)
    after_keys = set(after_sigs)

    added = sorted(after_sigs[k] for k in after_keys - before_keys)
    removed = sorted(before_sigs[k] for k in before_keys - after_keys)
    modified = sorted(
        f"{before_sigs[k]} -> {after_sigs[k]}" for k in before_keys & after_keys if before_sigs[k] != after_sigs[k]
    )

    return {
        "added_functions": added,
        "removed_functions": removed,
        "modified_functions": modified,
        "summary": {"added": len(added), "removed": len(removed), "modified": len(modified)},
    }


def _read_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        _ensure_no_bidi(content, str(path))
        return content
    except OSError:
        return None


def analyze_paths(paths: Sequence[str], base_ref: str, root: Path = ROOT) -> dict[str, object]:
    normalized_paths = sorted({normalize_path(p) for p in paths if normalize_path(p)})
    categories: list[str] = []
    risks: list[str] = []
    files: list[dict[str, object]] = []

    for path in normalized_paths:
        category = classify_category(path)
        risk = risk_for_category(category)
        categories.append(category)
        risks.append(risk)

        module = module_name(path) if path.endswith(".py") else ""
        after_content = _read_file(root / path) if path.endswith(".py") else None
        before_content = None  # base-ref integration deferred; present for future compatibility
        semantic = (
            semantic_diff(before_content, after_content, module)
            if path.endswith(".py")
            else {
                "added_functions": [],
                "removed_functions": [],
                "modified_functions": [],
                "summary": {"added": 0, "removed": 0, "modified": 0},
            }
        )

        files.append(
            {
                "path": path,
                "category": category,
                "risk": risk,
                "details": {
                    "module": module,
                    "semantic": semantic,
                },
            }
        )

    cat_counts = dict.fromkeys(CATEGORY_PRIORITY, 0)  # type: ignore[arg-type]
    risk_counts = dict.fromkeys(RISK_ORDER, 0)  # type: ignore[arg-type]
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for risk in risks:
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    primary = _select_primary_category(cat_counts)
    max_risk = _max_risk(risks)

    return {
        "base_ref": base_ref,
        "primary_category": primary,
        "max_risk": max_risk,
        "counts": {"categories": cat_counts, "risks": risk_counts},
        "files": files,
    }


def _select_primary_category(counts: dict[str, int]) -> str:
    max_count = max(counts.values()) if counts else 0
    candidates = [cat for cat, count in counts.items() if count == max_count]
    for cat in CATEGORY_PRIORITY:
        if cat in candidates:
            return cat
    return CATEGORY_PRIORITY[-1]


def _max_risk(risks: list[str]) -> str:
    max_rank = -1
    max_name = "informational"
    for name in risks:
        rank = RISK_ORDER.get(name, -1)
        if rank > max_rank:
            max_rank = rank
            max_name = name
    return max_name


def _read_paths_file(path_arg: str) -> list[str]:
    content = sys.stdin.read() if path_arg == "-" else Path(path_arg).read_text(encoding="utf-8")
    _ensure_no_bidi(content, "--files")
    return [line.strip() for line in content.splitlines() if line.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic change analyzer")
    parser.add_argument("--files", required=True, help="Path to file containing list of changed files")
    parser.add_argument("--base-ref", default="origin/main", help="Git base reference (metadata only)")
    parser.add_argument("--output", default="-", help="Output path or '-' for stdout")
    parser.add_argument("--format", default="json", choices=["json"], help="Output format (json only)")
    args = parser.parse_args(argv)
    files_path = Path(args.files) if args.files != "-" else None
    if files_path is not None:
        if not files_path.exists():
            parser.error(f"--files path does not exist: {files_path}")
        if not files_path.is_file():
            parser.error(f"--files path is not a file: {files_path}")
    return args


def _write_output(payload: dict[str, object], output: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output == "-":
        print(text)
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        paths = _read_paths_file(args.files)
        result = analyze_paths(paths, args.base_ref, ROOT)
        _write_output(result, args.output)
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
