## Description

<!-- Provide a clear and concise description of the changes in this PR -->

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Security fix
- [ ] Dependencies update

## Scope of Changes

<!-- Describe which components/modules are affected -->

**Affected Components:**
- [ ] Core Engine (`src/mlsdm/core/`, `src/mlsdm/engine/`)
- [ ] Cognition/Moral Filter (`src/mlsdm/cognition/`)
- [ ] Memory System (`src/mlsdm/memory/`)
- [ ] API/Endpoints (`src/mlsdm/api/`)
- [ ] Security (`src/mlsdm/security/`, `src/mlsdm/utils/security_*`)
- [ ] Configuration (`src/mlsdm/utils/config_*`)
- [ ] Observability/Metrics (`src/mlsdm/utils/metrics.py`, `src/mlsdm/observability/`)
- [ ] Speech/NeuroLang (`src/mlsdm/speech/`, `src/mlsdm/extensions/`)
- [ ] Documentation
- [ ] Tests
- [ ] CI/CD

## Security Impact Assessment

<!-- REQUIRED: Evaluate potential security implications -->

**Security Risk Level:** <!-- Choose one: Low / Medium / High -->

**Security Considerations:**
- [ ] No security-sensitive changes
- [ ] Changes to authentication/authorization
- [ ] Changes to data validation/sanitization
- [ ] Changes to LLM integration or prompts
- [ ] Changes to configuration or secrets handling
- [ ] Changes to API endpoints or middleware
- [ ] Introduces new external dependencies

**Security Checklist:**
- [ ] Input validation implemented where needed
- [ ] No secrets or credentials in code
- [ ] No SQL injection or XSS vulnerabilities introduced
- [ ] Rate limiting considered for new endpoints
- [ ] Security tests added/updated
- [ ] Follows principle of least privilege
- [ ] Logging doesn't expose sensitive data

## Testing

<!-- REQUIRED: Describe testing performed -->

**Tests Run:**
```bash
# List commands you ran locally
pytest tests/unit/test_*.py
pytest tests/integration/test_*.py
pytest tests/security/test_*.py
pytest tests/property/test_*.py
ruff check src/ tests/
mypy src/mlsdm
```

**Test Coverage:**
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Property-based tests added/updated (if applicable)
- [ ] Security tests added/updated (if applicable)
- [ ] Manual testing performed

**Test Results:**
<!-- Paste relevant test output or summarize results -->
```
<paste test results here>
```

## Regression Risk

**Risk Level:** <!-- Choose one: Low / Medium / High -->

**Justification:**
<!-- Explain why you assessed this risk level -->
- **Low**: Documentation, tests, or isolated changes with no impact on runtime behavior
- **Medium**: Changes to existing functionality with good test coverage
- **High**: Changes to core components, security, or breaking changes

**Mitigation:**
<!-- If Medium/High risk, describe how you've mitigated potential regressions -->

## Behavioral Changes

<!-- List any changes in system behavior, API responses, or user-facing functionality -->

- 

## Performance Impact

<!-- Describe any performance implications, positive or negative -->

- [ ] No performance impact expected
- [ ] Performance improvement (describe)
- [ ] Potential performance degradation (describe and justify)

## Documentation

- [ ] Documentation updated (README, API docs, architecture docs)
- [ ] Code comments added/updated where needed
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Migration guide provided (for breaking changes)

## Checklist

<!-- Ensure all items are checked before requesting review -->

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings or errors
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] I have checked for security vulnerabilities in new dependencies
- [ ] No secrets or sensitive data are committed

## Related Issues

<!-- Link related issues -->

Closes #
Related to #

## Additional Context

<!-- Add any other context, screenshots, or information about the PR here -->

---

## For Reviewers

**Review Focus Areas:**
<!-- Suggest specific areas reviewers should focus on -->

- 

**Questions for Reviewers:**
<!-- Any specific questions or concerns for the review team -->

- 
