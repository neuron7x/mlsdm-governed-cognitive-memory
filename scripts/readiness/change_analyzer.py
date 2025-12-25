#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

ROOT = Path(__file__).resolve().parents[2]

MORAL_FILTER_TOKEN = "moral_filter"
OBSERVABILITY_KEYWORDS = ("observability", "metrics", "logging", "tracing")
CATEGORY_PRIORITY = {
    "security_critical": 0,
    "functional_core": 1,
    "infrastructure": 2,
    "observability": 3,
    "test_coverage": 4,
    "documentation": 5,
}

RISK_ORDER = ["informational", "low", "medium", "high", "critical"]
RISK_LEVEL = {name: index for index, name in enumerate(RISK_ORDER)}

CATEGORY_RISK = {
    "security_critical": "critical",
    "functional_core": "high",
    "infrastructure": "medium",
    "observability": "low",
    "test_coverage": "informational",
    "documentation": "informational",
}


@dataclass(frozen=True)
class Signature:
    qualname: str
    args: Tuple[str, ...]
    return_annotation: str
    decorators: Tuple[str, ...]
    kind: str

    def as_string(self, module: str) -> str:
        args_repr = ",".join(self.args)
        return f"{module}:{self.qualname}({args_repr})->{self.return_annotation}"


def categorize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    parts = tuple(Path(normalized).parts)

    if normalized.startswith("src/") and "/security/" in normalized:
        return "security_critical"
    if any(
        part.lower() == MORAL_FILTER_TOKEN or Path(part).stem.lower() == MORAL_FILTER_TOKEN
        for part in parts
    ):
        return "security_critical"
    if normalized.startswith("tests/") or Path(normalized).name.startswith("test_"):
        return "test_coverage"
    if normalized.endswith(".md") or normalized.startswith("docs/"):
        return "documentation"
    if normalized.startswith(".github/workflows/") or normalized.startswith("deploy/") or normalized.startswith("config/"):
        return "infrastructure"
    if any(segment in lower for segment in OBSERVABILITY_KEYWORDS):
        return "observability"
    if normalized.startswith("src/"):
        return "functional_core"
    return "infrastructure"


def risk_for_category(category: str) -> str:
    return CATEGORY_RISK.get(category, "informational")


def risk_level(risk: str) -> int:
    return RISK_LEVEL[risk]


def category_priority(category: str) -> int:
    return CATEGORY_PRIORITY.get(category, len(CATEGORY_PRIORITY))


def base_ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def load_base_content(path: str, base_ref: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{base_ref}:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def load_current_content(path: str) -> Optional[str]:
    file_path = ROOT / path
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def annotation_to_str(annotation: Optional[ast.AST]) -> str:
    if annotation is None:
        return "None"
    return ast.unparse(annotation).strip()


def _format_arg(arg: ast.arg, default: Optional[ast.expr]) -> str:
    name = arg.arg
    if arg.annotation:
        name = f"{name}:{ast.unparse(arg.annotation).strip()}"
    if default is not None:
        name = f"{name}={ast.unparse(default).strip()}"
    return name


def _args_from_ast(args: ast.arguments) -> Tuple[str, ...]:
    parts: List[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults_padding = [None] * (len(positional) - len(args.defaults))
    positional_defaults = defaults_padding + list(args.defaults)

    for index, (arg, default) in enumerate(zip(positional, positional_defaults)):
        parts.append(_format_arg(arg, default))
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg:
        parts.append(f"*{_format_arg(args.vararg, None)}")
    elif args.kwonlyargs:
        parts.append("*")

    kw_defaults = list(args.kw_defaults or [])
    kw_defaults.extend([None] * (len(args.kwonlyargs) - len(kw_defaults)))
    for arg, default in zip(args.kwonlyargs, kw_defaults):
        parts.append(_format_arg(arg, default))

    if args.kwarg:
        parts.append(f"**{_format_arg(args.kwarg, None)}")
    return tuple(parts)


def _decorators_from_ast(decorators: Sequence[ast.expr]) -> Tuple[str, ...]:
    return tuple(ast.unparse(dec).strip() for dec in decorators)


def module_name_from_path(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    if normalized.endswith("/__init__"):
        normalized = normalized[: -len("/__init__")]
    return normalized.replace("/", ".")


def extract_signatures(content: str, module: str) -> Dict[str, Signature]:
    tree = ast.parse(content)
    signatures: Dict[str, Signature] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = Signature(
                qualname=node.name,
                args=_args_from_ast(node.args),
                return_annotation=annotation_to_str(node.returns),
                decorators=_decorators_from_ast(node.decorator_list),
                kind="function",
            )
        elif isinstance(node, ast.ClassDef):
            signatures[node.name] = Signature(
                qualname=node.name,
                args=tuple(),
                return_annotation="None",
                decorators=_decorators_from_ast(node.decorator_list),
                kind="class",
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{node.name}.{child.name}"
                    signatures[qualname] = Signature(
                        qualname=qualname,
                        args=_args_from_ast(child.args),
                        return_annotation=annotation_to_str(child.returns),
                        decorators=_decorators_from_ast(child.decorator_list),
                        kind="method",
                    )
    return signatures


def diff_signatures(
    base: Mapping[str, Signature], current: Mapping[str, Signature], module: str
) -> Dict[str, List[str]]:
    added: List[str] = []
    removed: List[str] = []
    modified: List[str] = []

    for qualname, signature in current.items():
        base_sig = base.get(qualname)
        if base_sig is None:
            added.append(signature.as_string(module))
            continue
        if (
            signature.args != base_sig.args
            or signature.return_annotation != base_sig.return_annotation
            or signature.decorators != base_sig.decorators
            or signature.kind != base_sig.kind
        ):
            modified.append(signature.as_string(module))

    for qualname, signature in base.items():
        if qualname not in current:
            removed.append(signature.as_string(module))

    return {
        "functions_added": sorted(added),
        "functions_removed": sorted(removed),
        "functions_modified": sorted(modified),
    }


def normalize_paths(paths: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw in paths:
        value = raw.strip()
        if not value:
            continue
        if value.startswith("/"):
            absolute = Path(value)
            try:
                relative = absolute.relative_to(ROOT)
            except ValueError:
                raise ValueError(f"Path outside repository root: {value}")
            normalized_value = relative.as_posix()
        else:
            normalized_value = Path(value).as_posix()
        if normalized_value not in seen:
            seen.add(normalized_value)
            normalized.append(normalized_value)
    return normalized


def analyze_files(
    file_paths: Sequence[str],
    base_ref: str,
    *,
    base_loader=load_base_content,
    current_loader=load_current_content,
) -> Dict[str, object]:
    files_result: Dict[str, Dict[str, object]] = {}

    for path in sorted(normalize_paths(file_paths)):
        category = categorize_path(path)
        risk = risk_for_category(category)
        semantic = {"functions_added": [], "functions_removed": [], "functions_modified": []}
        if path.endswith(".py"):
            module_name = module_name_from_path(path)
            base_content = base_loader(path, base_ref)
            current_content = current_loader(path)
            try:
                base_signatures = extract_signatures(base_content, module_name) if base_content else {}
                current_signatures = extract_signatures(current_content, module_name) if current_content else {}
            except SyntaxError as exc:
                location = ":".join(
                    str(part)
                    for part in (path, exc.lineno or "", exc.offset or "")
                    if str(part) != ""
                )
                message = f"{location} {exc.msg}".strip()
                raise SyntaxError(message) from exc
            semantic = diff_signatures(base_signatures, current_signatures, module_name)
        files_result[path] = {"category": category, "risk": risk, "semantic": semantic}

    summary_categories: Dict[str, int] = {}
    for item in files_result.values():
        category = item["category"]
        summary_categories[category] = summary_categories.get(category, 0) + 1

    primary_category, max_risk = determine_overall(files_result)

    return {
        "primary_category": primary_category,
        "max_risk": max_risk,
        "summary": {
            "files_analyzed": len(files_result),
            "categories": summary_categories,
        },
        "files": files_result,
    }


def determine_overall(files_result: Mapping[str, Mapping[str, object]]) -> Tuple[str, str]:
    if not files_result:
        return "", "informational"

    max_level = -1
    chosen_category = ""
    chosen_risk = "informational"

    for path in sorted(files_result):
        entry = files_result[path]
        risk = cast(str, entry["risk"])
        level = risk_level(risk)
        category = cast(str, entry["category"])
        if level > max_level:
            max_level = level
            chosen_category = category
            chosen_risk = risk
        elif level == max_level and category_priority(category) < category_priority(chosen_category):
            chosen_category = category
            chosen_risk = risk

    return chosen_category, chosen_risk


def read_files_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic change analyzer for readiness.")
    parser.add_argument("--files", required=True, help="Text file with one changed path per line.")
    parser.add_argument("--base-ref", default="origin/main", help="Git base ref for comparison.")
    parser.add_argument("--output", required=True, help="Path to write JSON analysis result.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    files_path = Path(args.files)
    output_path = Path(args.output)
    base_ref = args.base_ref

    if not files_path.exists():
        parser.error(f"--files path does not exist: {files_path}")
    if not base_ref_exists(base_ref):
        parser.error(f"--base-ref is not a valid git ref: {base_ref}")

    changed_files = normalize_paths(read_files_list(files_path))

    try:
        analysis = analyze_files(changed_files, base_ref)
    except (SyntaxError, ValueError) as exc:
        print(f"Failed to analyze changes: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
