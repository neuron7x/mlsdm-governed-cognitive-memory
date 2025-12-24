"""Thin wrapper to preserve historical import path."""

from __future__ import annotations

from mlsdm.cli import main as _main


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
