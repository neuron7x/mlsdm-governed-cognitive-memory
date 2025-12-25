#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent

SRC_PREFIX = "src/"
TESTS_PREFIX = "tests/"
SECURITY_KEYWORD = "moral_filter"
INFRA_PREFIXES = (".github/workflows/", "deploy/", "config/")
OBSERVABILITY_KEYWORDS = ("observability", "metrics", "logging", "tracing")

CATEGORIES: list[str] = [
    "security_critical",
    "functional_core",
    "infrastructure",
    "observability",
    "test_coverage",
    "documentation",
]

CATEGORY_PRIORITY: list[str] = CATEGORIES.copy()

RISK_MAP: Mapping[str, str] = {
    "security_critical": "critical",
    "functional_core": "high",
    "infrastructure": "medium",
    "observability": "low",
    "test_coverage": "informational",
    "documentation": "informational",
}

RISK_ORDER: Mapping[str, int] = {
    "informational": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

BIDI_PATTERN = re.compile(r"[\u202A-\u202E\u2066-\u2069]")


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def ensure_no_bidi(text: str, context: str) -> None:
    if BIDI_PATTERN.search(text):
        raise ValueError(f"Bidirectional control character detected in {context}")


def classify_category(path: str) -> str:
    normalized = normalize_path(path)
    lower = normalized.lower()
    if SECURITY_KEYWORD in lower:
        return "security_critical"
    if normalized.startswith(f"{SRC_PREFIX}security/") or (
        normalized.startswith(SRC_PREFIX) and "/security/" in normalized[len(SRC_PREFIX) :]
    ):
        return "security_critical"
    if normalized.startswith(SRC_PREFIX):
        return "functional_core"
    if normalized.startswith(INFRA_PREFIXES):
        return "infrastructure"
    if any(keyword in lower for keyword in OBSERVABILITY_KEYWORDS):
        return "observability"
    name = Path(normalized).name
    if normalized.startswith(TESTS_PREFIX) or (name.startswith("test_") and name.endswith(".py")):
        return "test_coverage"
    if normalized.startswith("docs/") or normalized.endswith(".md"):
        return "documentation"
    return "functional_core"


def risk_for_category(category: str) -> str:
    return RISK_MAP.get(category, "high")


def module_name(path: str) -> str:
    normalized = normalize_path(path)
    if normalized.startswith(SRC_PREFIX):
        normalized = normalized[len(SRC_PREFIX) :]
    elif normalized.startswith(TESTS_PREFIX):
        normalized = normalized[len(TESTS_PREFIX) :]
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


def _format_returns(node: ast.AST) -> str:
    returns = getattr(node, "returns", None)
    if returns is None:
        return "None"
    try:
        return ast.unparse(returns)
    except (TypeError, ValueError):
        return "None"


def _decorator_suffix(node: ast.AST) -> str:
    decorators: list[str] = []
    for decorator in getattr(node, "decorator_list", []):
        try:
            decorators.append(ast.unparse(decorator).strip())
        except (TypeError, ValueError):
            continue
    return f"|decorators={','.join(sorted(decorators))}" if decorators else ""


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, module: str, class_name: str | None = None) -> str:
    qualname = f"{class_name}.{node.name}" if class_name else node.name
    args = _format_args(node.args)
    ret = _format_returns(node)
    return f"{module}:{qualname}({args})->{ret}{_decorator_suffix(node)}"


def _class_signature(node: ast.ClassDef, module: str) -> str:
    bases: list[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base).strip())
        except (TypeError, ValueError):
            continue
    bases_part = f"[bases={','.join(sorted(bases))}]" if bases else ""
    return f"{module}:{node.name}{bases_part}{_decorator_suffix(node)}"


def parse_python_signatures(source: str, module: str) -> dict[str, str]:
    try:
        ensure_no_bidi(source, f"module {module}")
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    signatures: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = _function_signature(node, module)
        elif isinstance(node, ast.ClassDef):
            signatures[node.name] = _class_signature(node, module)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{node.name}.{child.name}"
                    signatures[qual] = _function_signature(child, module, class_name=node.name)
    return signatures


def semantic_diff(before: str | None, after: str | None, module: str) -> dict[str, list[str] | dict[str, int]]:
    before_signatures = parse_python_signatures(before, module) if before else {}
    after_signatures = parse_python_signatures(after, module) if after else {}
    before_keys = set(before_signatures.keys())
    after_keys = set(after_signatures.keys())

    added = sorted(after_signatures[k] for k in after_keys - before_keys)
    removed = sorted(before_signatures[k] for k in before_keys - after_keys)
    modified = sorted(
        f"{before_signatures[k]} -> {after_signatures[k]}"
        for k in before_keys & after_keys
        if before_signatures[k] != after_signatures[k]
    )
    return {
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
        },
        "added_functions": added,
        "removed_functions": removed,
        "modified_functions": modified,
    }


def get_file_at_ref(path: str, ref: str, root: Path) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _read_working_tree(path: str, root: Path) -> str | None:
    target = root / path
    if not target.exists() or not target.is_file():
        return None
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        ensure_no_bidi(content, f"working tree file {path}")
        return content
    except OSError:
        return None


def _empty_semantic() -> dict[str, list[str] | dict[str, int]]:
    return {
        "summary": {"added": 0, "removed": 0, "modified": 0},
        "added_functions": [],
        "removed_functions": [],
        "modified_functions": [],
    }


def _primary_category(counts: Mapping[str, int]) -> str:
    if not counts:
        return CATEGORY_PRIORITY[0]
    max_count = max(counts.values())
    candidates = [cat for cat, count in counts.items() if count == max_count]
    for cat in CATEGORY_PRIORITY:
        if cat in candidates:
            return cat
    return CATEGORY_PRIORITY[0]


def analyze_paths(paths: Sequence[str], base_ref: str, root: Path = ROOT) -> dict[str, object]:
    categories: dict[str, int] = cast("dict[str, int]", dict.fromkeys(CATEGORIES, 0))
    files: dict[str, object] = {}
    max_risk_rank = 0

    for raw_path in paths:
        path = normalize_path(raw_path)
        category = classify_category(path)
        risk = risk_for_category(category)
        categories[category] += 1
        max_risk_rank = max(max_risk_rank, RISK_ORDER.get(risk, 0))

        module = module_name(path)
        before_src = get_file_at_ref(path, base_ref, root)
        after_src = _read_working_tree(path, root)
        semantic = (
            semantic_diff(before_src, after_src, module) if path.endswith(".py") else _empty_semantic()
        )

        files[path] = {
            "path": path,
            "category": category,
            "risk": risk,
            "module": module,
            "semantic_diff": semantic["summary"],
            "functions_added": semantic["added_functions"],
            "functions_removed": semantic["removed_functions"],
            "functions_modified": semantic["modified_functions"],
        }

    summary = {"files_analyzed": len(paths), "categories": dict(sorted(categories.items()))}
    max_risk = next((name for name, rank in RISK_ORDER.items() if rank == max_risk_rank), "informational")

    return {
        "primary_category": _primary_category(categories),
        "max_risk": max_risk,
        "summary": summary,
        "files": files,
    }


def _read_paths_file(paths_file: Path) -> list[str]:
    content = paths_file.read_text(encoding="utf-8")
    ensure_no_bidi(content, "--files")
    return [line.strip() for line in content.splitlines() if line.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic change analyzer")
    parser.add_argument("--files", required=True, type=Path, help="Path to file containing list of changed files")
    parser.add_argument("--base-ref", default="origin/main", help="Git base reference")
    parser.add_argument("--output", required=True, type=Path, help="Path to write JSON output")
    args = parser.parse_args(argv)
    if not args.files.exists():
        parser.error(f"--files path does not exist: {args.files}")
    if not args.files.is_file():
        parser.error(f"--files path is not a file: {args.files}")
    return args


def _write_output(result: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        paths = _read_paths_file(args.files)
        result = analyze_paths(paths, args.base_ref, ROOT)
        _write_output(result, args.output)
        return 0
    except SystemExit:
        # argparse already handled exit code 2
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
