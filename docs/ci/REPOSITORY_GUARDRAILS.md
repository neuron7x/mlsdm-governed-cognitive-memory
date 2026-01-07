# Repository Guardrails

## Never commit
- Coverage and test byproducts: `.coverage*`, `coverage.xml/json`, `htmlcov/`, `junit*.xml`, `reports/effectiveness_*`, `reports/ablation/*.json`, `evals/*_report.json`, `results/`
- Local caches and builds: `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.hypothesis/`, `dist/`, `build/`, `*.egg-info/`, virtualenvs (`.venv/`, `venv/`)
- Secrets and local config: `.env`, `mlsdm_config.sh`
- Temporary artifacts: `artifacts/tmp/`, downloaded tools like `conftest`

## CI enforces
- **Diff artifact gate:** `python scripts/ci/no_generated_artifacts.py` blocks generated/cache paths appearing in the git diff (only changed files are inspected).
- **Unicode safety:** `python scripts/check_bidi.py` fails the build if bidirectional/hidden Unicode controls appear in changed files.
- **Dependency determinism:** `python scripts/ci/export_requirements.py` is run in CI; any drift between `pyproject.toml` and the generated `requirements.txt` causes a failure.

## Verify locally before pushing
1. `python scripts/ci/no_generated_artifacts.py`
2. `python scripts/check_bidi.py`
3. `python scripts/ci/export_requirements.py --check`

## Why
- Generated artifacts and caches create noisy diffs and break reproducibility.
- Bidirectional control characters are a Trojan Source risk.
- Deterministic dependency exports keep installs repeatable and prevent silent drift.
