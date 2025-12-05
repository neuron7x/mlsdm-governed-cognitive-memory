# MLSDM Policy-as-Code

This directory contains policy rules that enforce governance, security, and configuration
standards across the repository. These policies are checked in CI to ensure consistent
compliance.

## Framework

- **OPA/Rego**: Open Policy Agent policies for declarative policy enforcement
- **Conftest**: CLI tool to run OPA policies against configuration files

## Policy Categories

### CI Workflow Policies (`ci/`)
- Workflow permission restrictions (no `write-all`)
- Required security checks
- Action version pinning

### Security Policies (`security/`)
- No hardcoded secrets or credentials
- Required authentication for external services
- Secure defaults enforcement

## Usage

### Running Policy Checks Locally

```bash
# Install conftest (if not installed)
brew install conftest  # macOS
# or
curl -L https://github.com/open-policy-agent/conftest/releases/download/v0.55.0/conftest_0.55.0_Linux_x86_64.tar.gz | tar xzf -

# Check CI workflows
conftest test .github/workflows/*.yml -p policies/ci/

# Check all policies
conftest test -p policies/ .github/workflows/*.yml config/*.yaml
```

### In CI

Policy checks are integrated into the CI workflow and run automatically on PRs.

## Adding New Policies

1. Create a new `.rego` file in the appropriate subdirectory
2. Add tests for the policy in a `*_test.rego` file
3. Update this README with a description of the new policy
4. Run `conftest verify` to test the policies

## Policy Reference

### ci/workflows.rego

| Rule | Description | STRIDE Category |
|------|-------------|-----------------|
| `deny_write_all_permissions` | Blocks workflows with overly permissive `write-all` | Elevation of Privilege |
| `deny_unpinned_actions` | Requires pinned action versions | Tampering, Elevation of Privilege |
| `deny_shell_injection` | Blocks potential shell injection patterns | Tampering |
| `warn_missing_timeout` | Warns if job has no timeout | Denial of Service |

### security/secrets.rego

| Rule | Description | STRIDE Category |
|------|-------------|-----------------|
| `deny_hardcoded_secrets` | Blocks patterns matching secrets | Information Disclosure |
| `deny_unencrypted_env` | Blocks plain text secrets in env vars | Information Disclosure |

## Maintenance

Policies should be reviewed and updated:
- When new CI workflows are added
- When security requirements change
- After security incidents or audits
- Quarterly as part of threat model review
