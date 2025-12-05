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

import rego.v1

# ============================================================================
# DENY RULES - Will fail CI if violated
# ============================================================================

# STRIDE: Elevation of Privilege
# Block workflows with overly permissive write-all permissions
deny contains msg if {
    input.permissions == "write-all"
    msg := "Workflow has 'permissions: write-all' which grants excessive access. Use specific permissions instead."
}

# STRIDE: Elevation of Privilege
# Block jobs with write-all permissions
deny contains msg if {
    some job_name
    job := input.jobs[job_name]
    job.permissions == "write-all"
    msg := sprintf("Job '%s' has 'permissions: write-all' which grants excessive access. Use specific permissions.", [job_name])
}

# STRIDE: Tampering
# Block unpinned third-party actions (not from github/* or actions/*)
deny contains msg if {
    some job_name
    job := input.jobs[job_name]
    some step
    step := job.steps[_]
    uses := step.uses
    uses != null
    
    # Check if it's a third-party action (not github/* or actions/*)
    not startswith(uses, "actions/")
    not startswith(uses, "github/")
    
    # Check if it uses a mutable reference (main, master, or missing SHA)
    not contains(uses, "@")
    
    msg := sprintf("Job '%s' uses unpinned action '%s'. Pin to a specific SHA for security.", [job_name, uses])
}

# STRIDE: Information Disclosure
# Block workflows that might expose secrets in logs
deny contains msg if {
    some job_name
    job := input.jobs[job_name]
    some step
    step := job.steps[_]
    run_cmd := step.run
    run_cmd != null
    
    # Check for potential secret exposure patterns
    contains(run_cmd, "echo ${{")
    contains(run_cmd, "secrets.")
    
    msg := sprintf("Job '%s' may be exposing secrets in logs via echo. Use masked outputs instead.", [job_name])
}

# ============================================================================
# WARN RULES - Will warn but not fail CI
# ============================================================================

# STRIDE: Denial of Service
# Warn if jobs don't have timeouts
warn contains msg if {
    some job_name
    job := input.jobs[job_name]
    not job["timeout-minutes"]
    msg := sprintf("Job '%s' has no timeout-minutes. Consider adding a timeout to prevent runaway jobs.", [job_name])
}

# STRIDE: Tampering
# Warn about mutable action references (branches instead of SHAs)
warn contains msg if {
    some job_name
    job := input.jobs[job_name]
    some step
    step := job.steps[_]
    uses := step.uses
    uses != null
    
    # Check for mutable references like @main or @master
    contains(uses, "@main")
    
    msg := sprintf("Job '%s' uses mutable reference in '%s'. Consider pinning to a SHA.", [job_name, uses])
}

warn contains msg if {
    some job_name
    job := input.jobs[job_name]
    some step
    step := job.steps[_]
    uses := step.uses
    uses != null
    
    # Check for mutable references like @main or @master
    contains(uses, "@master")
    
    msg := sprintf("Job '%s' uses mutable reference in '%s'. Consider pinning to a SHA.", [job_name, uses])
}

# ============================================================================
# HELPER RULES
# ============================================================================

# Check if a string contains a pattern
contains(str, pattern) if {
    indexof(str, pattern) >= 0
}
