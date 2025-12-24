| command/way | file(s) | status (canonical / wrapper / deprecated / remove) | notes |
| --- | --- | --- | --- |
| `mlsdm serve` (console script) | `src/mlsdm/cli/__init__.py` → `mlsdm.entrypoints.serve` | canonical | Primary user entrypoint. Respects CLI/env CONFIG_PATH before importing the app. |
| `python -m mlsdm.cli serve` | `src/mlsdm/cli/__init__.py` | canonical | Module form of the CLI; same contract as `mlsdm serve`. |
| `python -m mlsdm.entrypoints.dev` | `src/mlsdm/entrypoints/dev_entry.py` | wrapper | Dev profile (hot reload, debug); sets `MLSDM_RUNTIME_MODE=dev` before delegating to canonical serve. |
| `python -m mlsdm.entrypoints.cloud` | `src/mlsdm/entrypoints/cloud_entry.py` | wrapper | Cloud/Docker profile; sets `MLSDM_RUNTIME_MODE=cloud-prod`, used by container CMD. |
| `python -m mlsdm.entrypoints.agent` | `src/mlsdm/entrypoints/agent_entry.py` | wrapper | Agent/API profile; sets `MLSDM_RUNTIME_MODE=agent-api` with secure defaults. |
| `python -m mlsdm.entrypoints.health` | `src/mlsdm/entrypoints/health.py` | wrapper | Standalone health probe for diagnostics (no server start). |
| `python -m mlsdm.service.neuro_engine_service` | `src/mlsdm/service/neuro_engine_service.py` | deprecated | Legacy shim kept for backward compatibility; replaced by `mlsdm.entrypoints.cloud`. |
| `python examples/run_neuro_service.py` | `examples/run_neuro_service.py` | wrapper | Example-only launcher delegating to canonical serve; not for production. |
| `docker/Dockerfile` CMD | `docker/Dockerfile` → `python -m mlsdm.entrypoints.cloud` | wrapper | Official container entrypoint (cloud profile, health endpoints exposed). |
| `Dockerfile.neuro-engine-service` CMD | `Dockerfile.neuro-engine-service` → `python -m mlsdm.entrypoints.cloud` | wrapper | Neuro-engine image uses cloud wrapper; bundles `config/` for runtime. |
| `docker/docker-compose.yaml` service `neuro-engine` | `docker/docker-compose.yaml` | wrapper | Compose stack builds `Dockerfile.neuro-engine-service` and hits `/health` probes. |
