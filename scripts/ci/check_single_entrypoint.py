from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
MAKEFILE = ROOT / "Makefile"

ALLOWED_MAIN = {
    "src/mlsdm/cli/__init__.py",
    "src/mlsdm/entrypoints/serve.py",
    "src/mlsdm/service/neuro_engine_service.py",
    "src/mlsdm/entrypoints/health.py",
}


def fail(msg):
    print(f"::error::{msg}")
    sys.exit(1)


def main():
    mains = []
    for p in SRC.rglob("*.py"):
        if re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', p.read_text(errors="ignore")):
            rel = str(p.relative_to(ROOT))
            mains.append(rel)
            if rel not in ALLOWED_MAIN:
                fail(f"Non-canonical __main__ detected: {rel}")

    if MAKEFILE.exists():
        if re.search(r"python\s+-m\s+mlsdm\.entrypoints\.", MAKEFILE.read_text()):
            fail("Makefile uses forbidden python -m mlsdm.entrypoints.*")

    cli_dup = ROOT / "src/mlsdm/cli/main.py"
    if cli_dup.exists():
        fail("Duplicate CLI detected: src/mlsdm/cli/main.py")

    print("OK: single canonical entrypoint enforced")


if __name__ == "__main__":
    main()
