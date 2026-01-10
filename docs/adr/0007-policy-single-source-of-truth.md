# ADR-0007: Single source of truth for policy artifacts

**Status**: Accepted
**Date**: 2026-02-15
**Deciders**: MLSDM Architecture Working Group
**Categories**: Architecture | Security | Infrastructure

## Context

Policy-as-code assets have been split across multiple locations, which increases the
risk of drift between documentation, validation scripts, and CI enforcement. Teams
need a predictable canonical path for policy artifacts so that governance checks,
SLO validation, and security baselines stay aligned as the repository evolves.

## Decision

We will consolidate policy artifacts under a single canonical structure rooted at
`policies/`. Human- and tool-consumed YAML policy sources live in `policies/yaml/`,
and OPA/Rego enforcement rules live in `policies/opa/`. Documentation, validation
scripts, and CI checks must reference these canonical paths.

## Consequences

### Positive

- A single source of truth reduces drift between policy files, docs, and tests.
- Automation can reliably discover policy artifacts without bespoke path logic.
- Future policy additions have a clear, consistent home.

### Negative

- Existing references must be updated to the new canonical paths.
- External automation that relied on legacy paths must be adjusted.

### Neutral

- Policy semantics remain unchanged; only file organization is updated.

## Alternatives Considered

### Alternative 1: Keep multiple policy roots

- **Description**: Leave YAML policies in `policy/` while keeping OPA rules in `policies/`.
- **Pros**: Minimal change to existing references.
- **Cons**: Continues the split source of truth and increases drift risk.
- **Reason for rejection**: Fails to provide a single canonical policy location.

### Alternative 2: Store all policies at repository root

- **Description**: Move YAML and Rego files to the repository root with naming conventions.
- **Pros**: Simplifies path depth.
- **Cons**: Clutters the repo root and weakens separation of concerns.
- **Reason for rejection**: Reduces clarity and violates existing repo organization patterns.

## Implementation

- Move YAML policies to `policies/yaml/`.
- Move OPA/Rego policies to `policies/opa/`.
- Update documentation, validation scripts, and tests to reference canonical paths.

### Affected Components

- `policies/yaml/`
- `policies/opa/`
- `scripts/validate_policy_config.py`
- `docs/SECURITY_POLICY.md`
- `docs/SLO_VALIDATION_PROTOCOL.md`

### Related Documents

- `docs/SECURITY_POLICY.md`
- `docs/SLO_VALIDATION_PROTOCOL.md`
- `docs/REPO_ARCHITECTURE_MAP.md`

## References

- N/A

---

*Template based on [Michael Nygard's ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)*
