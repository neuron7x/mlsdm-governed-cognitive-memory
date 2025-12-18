# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

To report a vulnerability, please use the GitHub Security Advisory feature:

1. Navigate to [Security Advisories](https://github.com/neuron7x/mlsdm/security/advisories/new)
2. Click "Report a vulnerability"
3. Provide a detailed description of the vulnerability

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested remediation (if known)

### Response Timeline

| Severity | Initial Response | Triage      | Fix Timeline |
| -------- | ---------------- | ----------- | ------------ |
| Critical | 24 hours         | 48 hours    | 7 days       |
| High     | 48 hours         | 7 days      | 14 days      |
| Medium   | 7 days           | 14 days     | 30 days      |
| Low      | 14 days          | 30 days     | 90 days      |

## Scope

### In Scope

- API endpoints and authentication mechanisms
- Memory subsystem and data processing
- Security controls (rate limiting, input validation, payload scrubbing)
- Dependencies and supply chain vulnerabilities
- Deployment configurations (Kubernetes manifests, Docker files)
- CI/CD pipeline security

### Out of Scope

- Vulnerabilities in third-party services not controlled by this project
- Social engineering attacks
- Physical security
- Denial of service attacks requiring significant infrastructure
- Issues already disclosed in public CVE databases (unless unpatched)

## Disclosure Policy

- We follow coordinated vulnerability disclosure practices
- Public disclosure occurs after a patch is released plus 7 days
- Reporters may be credited in security advisories with their permission

## Security Documentation

For detailed security architecture, controls, and implementation guidance, see:

- [SECURITY_POLICY.md](SECURITY_POLICY.md) - Comprehensive security policy and controls
- [THREAT_MODEL.md](THREAT_MODEL.md) - STRIDE-based threat analysis
- [SECURITY_IMPLEMENTATION.md](SECURITY_IMPLEMENTATION.md) - Implementation details
- [SECURITY_GUARDRAILS.md](SECURITY_GUARDRAILS.md) - Runtime guardrails documentation
