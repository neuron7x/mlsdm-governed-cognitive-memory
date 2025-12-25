#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent

SRC_PREFIX = "src/"
TESTS_PREFIX = "tests/"
SECURITY_KEYWORD = "moral_filter"  # Security-sensitive subsystem marker
INFRA_PREFIXES = (".github/workflows/", "deploy/", "config/")

CATEGORIES: List[str] = [
    "security_critical",
    "functional_core",
    "infrastructure",
    "observability",
    "test_coverage",
    "documentation",
]

CATEGORY_PRIORITY: List[str] = CATEGORIES.copy()

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


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def classify_category(path: str) -> str:
    normalized = normalize_path(path)
    lower = normalized.lower()
    if SECURITY_KEYWORD in lower:
        return "security_critical"
    if normalized.startswith(f"{SRC_PREFIX}security/"):
        return "security_critical"
    if normalized.startswith(SRC_PREFIX) and "/security/" in normalized[len(SRC_PREFIX) :]:
        return "security_critical"
    if normalized.startswith(SRC_PREFIX):
        return "functional_core"
    if normalized.startswith(INFRA_PREFIXES):
        return "infrastructure"
    if any(key in lower for key in ("observability", "metrics", "logging", "tracing")):
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


def decorator_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    for decorator in getattr(node, "decorator_list", []):
        try:
            names.append(ast.unparse(decorator).strip())
        except (TypeError, ValueError):
            continue
    return sorted(names)


def format_args(args: ast.arguments) -> str:
    parts: List[str] = []
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


def format_return_annotation(node: ast.AST) -> str:
    returns = getattr(node, "returns", None)
    if returns is not None:
        try:
            return ast.unparse(returns)
        except (TypeError, ValueError):
            return "None"
    return "None"


def canonical_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, module: str, class_name: str | None = None
) -> str:
    """Build canonical signature for a top-level function or class method."""
    qualname = f"{class_name}.{node.name}" if class_name else node.name
    args = format_args(node.args)
    ret = format_return_annotation(node)
    decorators = decorator_names(node)
    decorator_suffix = f"|decorators={','.join(decorators)}" if decorators else ""
    return f"{module}:{qualname}({args})->{ret}{decorator_suffix}"


def canonical_class_signature(node: ast.ClassDef, module: str) -> str:
    bases: List[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base).strip())
        except (TypeError, ValueError):
            continue
    bases_part = f"[bases={','.join(sorted(bases))}]" if bases else ""
    decorators = decorator_names(node)
    decorator_suffix = f"|decorators={','.join(decorators)}" if decorators else ""
    return f"{module}:{node.name}{bases_part}{decorator_suffix}"


def extract_signatures(source: str | None, path: str) -> Dict[str, str]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    module = module_name(path)
    signatures: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = canonical_function_signature(node, module)
        elif isinstance(node, ast.ClassDef):
            signatures[node.name] = canonical_class_signature(node, module)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{node.name}.{child.name}"
                    signatures[qualname] = canonical_function_signature(child, module, class_name=node.name)
    return signatures


def git_show(path: str, base_ref: str, root: Path) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def read_after_content(path: str, root: Path) -> str | None:
    target = root / path
    if not target.exists():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def diff_signatures(before: Mapping[str, str], after: Mapping[str, str]) -> Dict[str, List[str]]:
    before_keys = set(before.keys())
    after_keys = set(after.keys())
    added = sorted(after[k] for k in after_keys - before_keys)
    removed = sorted(before[k] for k in before_keys - after_keys)
    modified = sorted(
        f"{before[k]} -> {after[k]}" for k in before_keys & after_keys if before[k] != after[k]
    )
    return {
        "functions_added": added,
        "functions_removed": removed,
        "functions_modified": modified,
    }


def semantic_for_python(path: str, base_ref: str, root: Path) -> Dict[str, List[str]]:
    before_content = git_show(path, base_ref, root)
    after_content = read_after_content(path, root)
    before_signatures = extract_signatures(before_content, path)
    after_signatures = extract_signatures(after_content, path)
    return diff_signatures(before_signatures, after_signatures)


def empty_semantic() -> Dict[str, List[str]]:
    return {
        "functions_added": [],
        "functions_removed": [],
        "functions_modified": [],
    }


def analyze_file(path: str, base_ref: str, root: Path) -> Dict[str, object]:
    category = classify_category(path)
    risk = risk_for_category(category)
    semantic = (
        semantic_for_python(path, base_ref, root)
        if normalize_path(path).endswith(".py")
        else empty_semantic()
    )
    return {
        "category": category,
        "risk": risk,
        "semantic": semantic,
    }


def primary_category(counts: Mapping[str, int]) -> str:
    max_count = max(counts.values()) if counts else 0
    candidates = [cat for cat, count in counts.items() if count == max_count]
    for cat in CATEGORY_PRIORITY:
        if cat in candidates:
            return cat
    return CATEGORY_PRIORITY[0]


def analyze_paths(paths: Sequence[str], base_ref: str, root: Path = ROOT) -> Dict[str, object]:
    categories: MutableMapping[str, int] = {category: 0 for category in CATEGORIES}
    files_block: Dict[str, object] = {}
    files_analyzed = len(paths)
    max_risk_rank = 0

    for path in paths:
        file_result = analyze_file(path, base_ref, root)
        category = file_result["category"]  # type: ignore[assignment]
        risk = file_result["risk"]  # type: ignore[assignment]
        categories[category] += 1
        max_risk_rank = max(max_risk_rank, RISK_ORDER.get(risk, 0))
        files_block[path] = file_result

    summary = {
        "files_analyzed": files_analyzed,
        "categories": dict(sorted(categories.items())),
    }

    result = {
        "primary_category": primary_category(categories),
        "max_risk": next(
            (name for name, rank in RISK_ORDER.items() if rank == max_risk_rank), "informational"
        ),
        "summary": summary,
        "files": files_block,
    }
    return result


def read_file_list(file_path: Path) -> List[str]:
    content = file_path.read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def write_output(result: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic change analyzer")
    parser.add_argument("--files", required=True, type=Path, help="Path to file containing list of changed files")
    parser.add_argument("--base-ref", required=False, default="origin/main", help="Git base reference")
    parser.add_argument("--output", required=True, type=Path, help="Path to write JSON output")
    args = parser.parse_args(argv)
    if not args.files.exists():
        parser.error(f"--files path does not exist: {args.files}")
    if not args.files.is_file():
        parser.error(f"--files path is not a file: {args.files}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = read_file_list(args.files)
    result = analyze_paths(paths, args.base_ref, ROOT)
    write_output(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
