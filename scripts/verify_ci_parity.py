"""
Compatibility wrapper to verify CI workflow/Makefile parity.

Delegates to verify_workflow_alignment to satisfy CI parity checks.
"""

from __future__ import annotations

from scripts.verify_workflow_alignment import main


if __name__ == "__main__":
    main()
