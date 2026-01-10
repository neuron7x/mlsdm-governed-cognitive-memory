# ============================================================================
# MLSDM CI Workflow Policy Rules (OPA/Rego)
# ============================================================================
#
# These policies enforce security and governance standards for GitHub Actions
# workflows. They are checked in CI using conftest.
#
# STRIDE Categories Addressed:
# - Elevation of Privilege: Restrict workflow permissions
# - Tampering: Require pinned action versions
# - Denial of Service: Require job timeouts
# - Information Disclosure: Block hardcoded secrets
#
# Usage:
#   conftest test .github/workflows/*.yml -p policies/ci/
#
# ============================================================================

package ci.workflows

policy := data.policy.security_baseline.controls.ci_workflow_policy

# Ensure policy data is present (fail-closed)
deny[msg] {
    not data.policy.security_baseline.controls.ci_workflow_policy
    msg := "Missing policy data key data.policy.security_baseline.controls.ci_workflow_policy (from policy/security-baseline.yaml controls.ci_workflow_policy)."
}

# ============================================================================
# DENY RULES - Will fail CI if violated
# ============================================================================

# STRIDE: Elevation of Privilege
# Block workflows with overly permissive permissions
deny[msg] {
    prohibited := policy.prohibited_permissions[_]
    input.permissions == prohibited
    msg := sprintf("Workflow has prohibited permissions: %s.", [prohibited])
}

# STRIDE: Elevation of Privilege
# Block jobs with overly permissive permissions
deny[msg] {
    job := input.jobs[job_name]
    prohibited := policy.prohibited_permissions[_]
    job.permissions == prohibited
    msg := sprintf("Job '%s' has prohibited permissions: %s.", [job_name, prohibited])
}

# STRIDE: Tampering
# Block unpinned third-party actions (not from github/* or actions/*)
deny[msg] {
    job := input.jobs[job_name]
    step := job.steps[_]
    uses := step.uses
    uses != null

    # Check if it's a third-party action (not github/* or actions/*)
    not is_first_party_action(uses)

    # Check if it's missing a version pin entirely
    not str_contains(uses, "@")

    msg := sprintf("Job '%s' uses unpinned action '%s'. Pin to a specific SHA for security.", [job_name, uses])
}

# STRIDE: Tampering
# Block third-party actions with mutable references
deny[msg] {
    job := input.jobs[job_name]
    step := job.steps[_]
    uses := step.uses
    uses != null

    # Check if it's a third-party action (not github/* or actions/*)
    not is_first_party_action(uses)

    # Parse the ref part and check for exact match with prohibited refs
    ref_part := get_action_ref(uses)
    ref_part != ""
    policy_ref_token := sprintf("@%s", [ref_part])
    policy_ref_token == policy.prohibited_mutable_refs[_]

    msg := sprintf("Job '%s' uses prohibited mutable reference in '%s'. Pin to a SHA for security.", [job_name, uses])
}

# STRIDE: Information Disclosure
# Block workflows that might expose secrets in logs
deny[msg] {
    job := input.jobs[job_name]
    step := job.steps[_]
    run_cmd := step.run
    run_cmd != null

    # Check for potential secret exposure patterns
    str_contains(run_cmd, "echo ${{")
    str_contains(run_cmd, "secrets.")

    msg := sprintf("Job '%s' may be exposing secrets in logs via echo. Use masked outputs instead.", [job_name])
}

# ============================================================================
# WARN RULES - Will warn but not fail CI
# ============================================================================

# STRIDE: Denial of Service
# Warn if jobs don't have timeouts
warn[msg] {
    job := input.jobs[job_name]
    not job["timeout-minutes"]
    msg := sprintf("Job '%s' has no timeout-minutes. Consider adding a timeout to prevent runaway jobs.", [job_name])
}

# STRIDE: Tampering
# Warn about mutable action references (branches instead of SHAs)
# NOTE: This warn rule only applies to third-party actions to keep signal clean
warn[msg] {
    job := input.jobs[job_name]
    step := job.steps[_]
    uses := step.uses
    uses != null

    # Only warn for third-party actions
    not is_first_party_action(uses)

    # Parse the ref part and check for exact match with prohibited refs
    ref_part := get_action_ref(uses)
    ref_part != ""
    policy_ref_token := sprintf("@%s", [ref_part])
    policy_ref_token == policy.prohibited_mutable_refs[_]

    msg := sprintf("Job '%s' uses mutable reference in '%s'. Consider pinning to a SHA.", [job_name, uses])
}

# ============================================================================
# HELPER RULES
# ============================================================================

# Check if a string contains a pattern
str_contains(str, pattern) {
    indexof(str, pattern) >= 0
}

is_first_party_action(uses) {
    owner := split(uses, "/")[0]
    owner == policy.first_party_action_owners[_]
}

# Extract the ref part from an action uses string (e.g., "owner/repo@ref" -> "ref")
# Returns empty string if no "@" is present.
# Note: GitHub Actions refs cannot contain "@" so there's always exactly one "@" in valid uses.
# For malformed strings with multiple "@", we take the last segment, which is a safe default.
get_action_ref(uses) = ref {
    str_contains(uses, "@")
    parts := split(uses, "@")
    count(parts) >= 2
    ref := parts[count(parts) - 1]
}

get_action_ref(uses) = "" {
    not str_contains(uses, "@")
}
