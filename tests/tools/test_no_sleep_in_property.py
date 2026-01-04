from __future__ import annotations

from pathlib import Path


def test_no_time_sleep_in_property_tests() -> None:
    property_dir = Path(__file__).resolve().parent.parent / "property"
    offenders: list[str] = []

    for path in property_dir.rglob("*.py"):
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if "time.sleep(" in line and "# allow-sleep:" not in line:
                    relative_path = path.relative_to(property_dir.parent)
                    offenders.append(f"{relative_path}:{lineno} -> {line.strip()}")

    offenders.sort()
    assert not offenders, "time.sleep found in property tests without waiver:\n" + "\n".join(
        offenders
    )
