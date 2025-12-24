# MLSDM Entrypoints (Source of Truth)

| Command / Invocation | File path | Status | Notes |
| --- | --- | --- | --- |
| `mlsdm serve` | `src/mlsdm/cli/__init__.py` (script: `mlsdm`) | **Canonical** | Primary user entrypoint; delegates to `mlsdm.entrypoints.serve` after applying CLI env (CONFIG_PATH, LLM_BACKEND, etc.). |
| `python -m mlsdm.cli serve` | `src/mlsdm/cli/__init__.py` | Wrapper (canonical alias) | Same behavior as `mlsdm serve`; kept for module execution. |
| `mlsdm.api.app:app` | `src/mlsdm/api/app.py` | **Canonical import string** | Used by uvicorn multi-worker; backed by module-level singleton created via `create_app()`. |
| `python -m mlsdm.entrypoints.cloud` | `src/mlsdm/entrypoints/cloud_entry.py` | Ops wrapper | For Docker/k8s; sets `MLSDM_RUNTIME_MODE=cloud-prod`, applies runtime config, then delegates to canonical serve. |
| `python -m mlsdm.entrypoints.dev` | `src/mlsdm/entrypoints/dev_entry.py` | Dev wrapper | Convenience for local development (reload/logging toggles) that forwards to canonical serve. |
| `python -m mlsdm.entrypoints.agent` | `src/mlsdm/entrypoints/agent_entry.py` | Ops wrapper | Agent/API mode for platform integrations; delegates to canonical serve. |
| Docker `CMD` / `ENTRYPOINT` | `Dockerfile.neuro-engine-service` → `python -m mlsdm.entrypoints.cloud` | Ops wrapper | Container start command; uses cloud runtime defaults and canonical serve. |
| `examples/run_neuro_service.py` | `examples/run_neuro_service.py` | Example wrapper | Demonstrates invoking canonical `mlsdm serve` via subprocess; not a separate runtime. |
| Legacy `mlsdm.service.neuro_engine_service` | `src/mlsdm/service/neuro_engine_service.py` | Deprecated wrapper | Thin shim preserved for backward compatibility; delegates to canonical cloud entrypoint. |

> Canonical public startup for humans: **`mlsdm serve`**. All other entrypoints are wrappers or deprecated shims and must not diverge in behavior.
