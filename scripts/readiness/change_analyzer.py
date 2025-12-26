"""Semantic change analyzer CLI and library."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import yaml

BIDI_PATTERN = re.compile(r"[\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069]")
ROOT = Path(__file__).resolve().parents[2]

SECURITY_MARKERS: tuple[str, ...] = (
    "security",
    "auth",
    "oidc",
    "rbac",
    "mtls",
    "guardrails",
    "signing",
    "crypto",
    "encryption",
    "permission",
    "scrubber",
    "policy",
    "secrets",
    "moral_filter",
    "memory/phase",
)
OBSERVABILITY_MARKERS: tuple[str, ...] = ("observability", "metrics", "logging", "tracing")

CATEGORY_PRIORITY: tuple[str, ...] = (
    "security_critical",
    "test_coverage",
    "documentation",
    "infrastructure",
    "observability",
    "functional_core",
    "mixed",
)

RISK_MAP: dict[str, str] = {
    "security_critical": "critical",
    "functional_core": "high",
    "infrastructure": "medium",
    "observability": "low",
    "test_coverage": "informational",
    "documentation": "informational",
}

RISK_LEVELS: tuple[str, ...] = ("informational", "low", "medium", "high", "critical")

RISK_ORDER: dict[str, int] = {name: idx for idx, name in enumerate(RISK_LEVELS)}


def _ensure_no_bidi(text: str, label: str) -> None:
    if BIDI_PATTERN.search(text):
        raise ValueError(f"Bidirectional control characters detected in {label}")


def normalize_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned


def classify_category(path: str) -> str:
    normalized = normalize_path(path).lower()
    if any(marker in normalized for marker in SECURITY_MARKERS) or "/security/" in normalized:
        return "security_critical"
    name = Path(normalized).name
    if normalized.startswith("tests/") or name.startswith("test_"):
        return "test_coverage"
    if normalized.startswith("docs/") or normalized.endswith((".md", ".rst", ".txt")):
        return "documentation"
    if normalized.startswith(".github/workflows/") or normalized.startswith("deploy/") or normalized.startswith(
        "config/"
    ):
        return "infrastructure"
    if any(marker in normalized for marker in OBSERVABILITY_MARKERS):
        return "observability"
    return "functional_core"


def risk_for_category(category: str) -> str:
    return RISK_MAP.get(category, "informational")


def module_name(path: str) -> str:
    normalized = normalize_path(path)
    for prefix in ("src/", "tests/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    without_ext = normalized.rsplit(".", 1)[0]
    return without_ext.replace("/", ".")


def _arg_list(args: ast.arguments) -> list[str]:
    parts: list[str] = []
    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")
    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwonlyargs and not args.vararg:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return parts


def _return_annotation(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
        return ast.unparse(node.returns)
    return "None"


def _function_sig(node: ast.AST, module: str, class_name: str | None = None) -> str:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    name = f"{class_name}.{node.name}" if class_name else node.name
    args = ",".join(_arg_list(node.args))
    ret = _return_annotation(node)
    return f"{module}:{name}({args})->{ret}"


def _class_sig(node: ast.ClassDef, module: str) -> str:
    bases = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            bases.append("...")
    bases_str = f"[{','.join(bases)}]" if bases else ""
    return f"{module}:{node.name}{bases_str}"


def parse_python_signatures(source: str, module: str) -> tuple[dict[str, str], bool]:
    try:
        _ensure_no_bidi(source, module)
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}, True

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
    return signatures, False


def semantic_diff(before: str | None, after: str | None, module: str) -> dict[str, object]:
    before_sigs, before_err = parse_python_signatures(before, module) if before else ({}, False)
    after_sigs, after_err = parse_python_signatures(after, module) if after else ({}, False)
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
        "parse_error": before_err or after_err,
    }


def _yaml_key_diff(before: str | None, after: str | None) -> dict[str, object]:
    parse_error = False

    def _load(text: str | None) -> dict:
        nonlocal parse_error
        if text is None:
            return {}
        try:
            _ensure_no_bidi(text, "yaml")
        except ValueError:
            parse_error = True
            return {}
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            parse_error = True
            return {}
        return data if isinstance(data, dict) else {}

    before_obj = _load(before)
    after_obj = _load(after)
    before_keys = set(before_obj)
    after_keys = set(after_obj)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in before_keys & after_keys if before_obj.get(k) != after_obj.get(k))
    return {"added_keys": added, "removed_keys": removed, "changed_keys": changed, "parse_error": parse_error}


def _read_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        _ensure_no_bidi(content, str(path))
        return content
    except OSError:
        return None


def get_file_at_ref(path: str, ref: str, root: Path = ROOT) -> str | None:
    cmd = ["git", "-C", str(root), "show", f"{ref}:{path}"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        return None
    try:
        _ensure_no_bidi(result.stdout, f"{ref}:{path}")
    except ValueError:
        return None
    return result.stdout


def _max_risk(risks: Sequence[str]) -> str:
    max_rank = -1
    max_name = "informational"
    for name in risks:
        rank = RISK_ORDER.get(name, -1)
        if rank > max_rank:
            max_rank = rank
            max_name = name
    return max_name


def _select_primary_category(categories: Sequence[str]) -> str:
    unique = set(categories)
    if len(unique) > 1:
        return "mixed"
    if not unique:
        return "functional_core"
    return next(iter(unique))


def analyze_paths(paths: Sequence[str], base_ref: str, root: Path = ROOT) -> dict[str, object]:
    """Analyze a list of file paths and return categorized, risk-rated results."""
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
        is_python = path.endswith(".py")
        is_yaml = path.endswith((".yaml", ".yml"))

        after_path = root / path
        after_content = _read_file(after_path) if (is_python or is_yaml) else None
        before_content = get_file_at_ref(path, base_ref, root) if (is_python or is_yaml) else None

        metadata = {
            "missing": after_content is None and (is_python or is_yaml),
            "new_file": before_content is None and after_content is not None,
            "parse_error": False,
        }

        semantic = {
            "added_functions": [],
            "removed_functions": [],
            "modified_functions": [],
            "summary": {"added": 0, "removed": 0, "modified": 0},
            "parse_error": False,
        }
        yaml_diff: dict[str, object] | None = None

        if is_yaml:
            yaml_diff = _yaml_key_diff(before_content, after_content)
            metadata["parse_error"] = bool(yaml_diff["parse_error"])
        elif is_python:
            semantic = semantic_diff(before_content, after_content, module)
            metadata["parse_error"] = bool(semantic["parse_error"])

        files.append(
            {
                "path": path,
                "category": category,
                "risk": risk,
                "metadata": metadata,
                "semantic_diff": semantic if is_python else None,
                "functions_added": semantic["added_functions"] if is_python else [],
                "functions_removed": semantic["removed_functions"] if is_python else [],
                "yaml_diff": yaml_diff,
            }
        )

    cat_counts = {name: 0 for name in CATEGORY_PRIORITY}
    for cat in categories:
        if cat not in cat_counts:
            raise ValueError(f"Unknown category: {cat}")
        cat_counts[cat] += 1
    risk_counts = {name: 0 for name in RISK_LEVELS}
    for risk in risks:
        if risk not in risk_counts:
            raise ValueError(f"Unknown risk: {risk}")
        risk_counts[risk] += 1
    primary = _select_primary_category(categories)
    max_risk = _max_risk(risks)

    return {
        "base_ref": base_ref,
        "primary_category": primary,
        "max_risk": max_risk,
        "counts": {"categories": cat_counts, "risks": risk_counts},
        "files": files,
    }


def _read_paths_file(path_arg: str) -> list[str]:
    content = sys.stdin.read() if path_arg == "-" else Path(path_arg).read_text(encoding="utf-8")
    _ensure_no_bidi(content, "--files")
    return [line.strip() for line in content.splitlines() if line.strip()]


def _write_output(payload: dict[str, object], output: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output == "-":
        print(text)
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(out_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic change analyzer")
    parser.add_argument("--files", required=True, help="Path to file containing list of changed files")
    parser.add_argument("--base-ref", default="origin/main", help="Git base reference")
    parser.add_argument("--output", required=True, help="Output path or '-' for stdout")
    parser.add_argument("--format", default="json", choices=["json"], help="Output format (json only)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        paths = _read_paths_file(args.files)
        result = analyze_paths(paths, base_ref=args.base_ref, root=ROOT)
        _write_output(result, args.output)
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
