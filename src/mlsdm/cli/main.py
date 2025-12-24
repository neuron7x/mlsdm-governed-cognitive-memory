"""Thin wrapper to the canonical MLSDM CLI."""

from __future__ import annotations

import sys

from mlsdm.cli import main as _canonical_main


def main() -> int:
    """Delegate to the canonical CLI entrypoint."""
    return _canonical_main()


if __name__ == "__main__":
    sys.exit(main())
