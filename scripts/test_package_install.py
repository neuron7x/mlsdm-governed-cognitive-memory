#!/usr/bin/env python3
"""
Package installation smoke test.

This script is used by 'make test-package' to verify that the installed
package works correctly.
"""


def main():
    """Run package smoke test."""
    from mlsdm import __version__, create_llm_wrapper

    print(f"MLSDM v{__version__} installed OK")

    wrapper = create_llm_wrapper()
    result = wrapper.generate("test", moral_value=0.8)

    print(f"Smoke test: accepted={result['accepted']}")

    if not result["accepted"]:
        print("ERROR: Smoke test failed - request should have been accepted")
        return 1

    print("✓ Package smoke test passed")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
