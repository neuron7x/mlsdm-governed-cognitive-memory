"""Contract tests for detecting architecture dependency cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlsdm.config.architecture_manifest import (
    ARCHITECTURE_MANIFEST,
    PACKAGE_ROOT,
    build_manifest_dependency_graph,
)
from tests.contracts.architecture_imports import build_module_dependency_graph

if TYPE_CHECKING:
    from collections.abc import Iterable


def _sorted_nodes(graph: dict[str, set[str]]) -> list[str]:
    return sorted(graph.keys())


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Detect cycles in a directed graph with deterministic ordering."""
    nodes = _sorted_nodes(graph)
    index = {node: idx for idx, node in enumerate(nodes)}
    cycles: list[list[str]] = []

    def dfs(start: str, current: str, stack: list[str]) -> None:
        for neighbor in sorted(graph.get(current, set())):
            if neighbor == start:
                cycles.append([*stack, start])
                continue
            if index.get(neighbor, -1) < index[start]:
                continue
            if neighbor in stack:
                continue
            dfs(start, neighbor, [*stack, neighbor])

    for node in nodes:
        dfs(node, node, [node])

    cycles.sort(key=lambda cycle: tuple(cycle))
    return cycles


def _format_cycle(cycle: Iterable[str]) -> str:
    return " -> ".join(cycle)


def test_manifest_dependency_graph_is_acyclic() -> None:
    """Manifest allowed dependencies must not create cycles."""
    graph = build_manifest_dependency_graph(ARCHITECTURE_MANIFEST)
    cycles = find_cycles(graph)
    if cycles:
        formatted = "\n".join(f"manifest cycle: {_format_cycle(cycle)}" for cycle in cycles)
        raise AssertionError("Manifest dependency cycles detected:\n" + formatted)


def test_code_dependency_graph_is_acyclic() -> None:
    """Actual module imports must not create cycles."""
    graph, evidence = build_module_dependency_graph()
    cycles = find_cycles(graph)
    if cycles:
        lines: list[str] = []
        for cycle in cycles:
            lines.append(f"code cycle: {_format_cycle(cycle)}")
            for source, target in zip(cycle, cycle[1:], strict=False):
                edge = (source, target)
                samples = evidence.get(edge, [])
                for path, statement in samples:
                    lines.append(
                        f"  evidence: {source} -> {target}: "
                        f"{path.relative_to(PACKAGE_ROOT)} ({statement})"
                    )
        raise AssertionError("Code dependency cycles detected:\n" + "\n".join(lines))
