#!/usr/bin/env bash
# ==============================================================================
# MLSDM Kubernetes Manifest Validation Script
# ==============================================================================
# Copyright (c) 2024 MLSDM Project
# SPDX-License-Identifier: MIT
#
# Validates all Kubernetes manifests for syntax and
# common issues before deployment.
#
# Usage:
#   ./deploy/scripts/validate-manifests.sh
#   ./deploy/scripts/validate-manifests.sh --strict
#
# Requirements:
#   - yq or Python with PyYAML (for YAML validation)
#   - kubectl (optional, for kustomize validation)
#   - kubeconform (optional, for schema validation)
# ==============================================================================

# Strict mode: exit on error, error in pipes, undefined vars, fail on glob miss
set -eEfuo pipefail
shopt -s nullglob
IFS=$'\n\t'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$DEPLOY_DIR/k8s"
MONITORING_DIR="$DEPLOY_DIR/monitoring"
GRAFANA_DIR="$DEPLOY_DIR/grafana"

# Parse arguments
STRICT_MODE=false
for arg in "$@"; do
    case $arg in
        --strict)
            STRICT_MODE=true
            shift
            ;;
    esac
done

# Track validation status
ERRORS=0
WARNINGS=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# Check if required tools are available
check_tools() {
    log_info "Checking available tools..."
    
    HAS_YQ=false
    HAS_PYTHON=false
    HAS_KUBECTL=false
    HAS_KUBECONFORM=false
    
    if command -v yq &> /dev/null; then
        log_success "yq found"
        HAS_YQ=true
    fi
    
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml" 2>/dev/null; then
            log_success "python3 with PyYAML found"
            HAS_PYTHON=true
        fi
    fi
    
    if command -v kubectl &> /dev/null; then
        log_success "kubectl found"
        HAS_KUBECTL=true
    fi
    
    if command -v kubeconform &> /dev/null; then
        log_success "kubeconform found"
        HAS_KUBECONFORM=true
    fi
    
    if [ "$HAS_YQ" = false ] && [ "$HAS_PYTHON" = false ]; then
        log_error "Neither yq nor python3+PyYAML found. Cannot validate YAML."
        exit 1
    fi
}

# Validate YAML syntax
validate_yaml_syntax() {
    local file=$1
    local filename
    filename=$(basename "$file")
    
    if [ "$HAS_YQ" = true ]; then
        if yq eval '.' "$file" > /dev/null 2>&1; then
            log_success "$filename - YAML syntax valid"
            return 0
        else
            log_error "$filename - YAML syntax error"
            yq eval '.' "$file" 2>&1 | head -5
            return 1
        fi
    elif [ "$HAS_PYTHON" = true ]; then
        if python3 -c "import yaml; list(yaml.safe_load_all(open('$file')))" 2>/dev/null; then
            log_success "$filename - YAML syntax valid"
            return 0
        else
            log_error "$filename - YAML syntax error"
            return 1
        fi
    fi
    return 0
}

# Validate JSON syntax
validate_json_syntax() {
    local file=$1
    local filename
    filename=$(basename "$file")
    
    if python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
        log_success "$filename - JSON syntax valid"
        return 0
    else
        log_error "$filename - JSON syntax error"
        return 1
    fi
}

# Validate with yq for stricter YAML linting
validate_yaml_strict() {
    local file=$1
    local filename
    filename=$(basename "$file")
    
    if [ "$HAS_YQ" = true ]; then
        if yq eval '.' "$file" > /dev/null 2>&1; then
            log_success "$filename - yq validation passed"
            return 0
        else
            log_error "$filename - yq validation failed"
            return 1
        fi
    fi
    return 0
}

# Validate Kubernetes schema with kubeconform
validate_k8s_schema() {
    local file=$1
    local filename
    filename=$(basename "$file")
    
    if [ "$HAS_KUBECONFORM" = true ]; then
        if kubeconform -strict -summary "$file" 2>&1; then
            log_success "$filename - schema validation passed"
            return 0
        else
            log_error "$filename - schema validation failed"
            return 1
        fi
    fi
    return 0
}

# Check for common issues
check_common_issues() {
    local file=$1
    local filename
    filename=$(basename "$file")
    
    # Check for hardcoded secrets (look for non-placeholder, non-variable values)
    # Only warn if we see actual values that aren't placeholders or env vars
    if grep -qE "(password|secret|api-key|apikey):" "$file"; then
        # Check for values that look like actual secrets (not placeholders or env refs)
        if grep -qE "(password|secret|api-key|apikey):\s+['\"]?[a-zA-Z0-9+/=]{8,}['\"]?\s*$" "$file"; then
            if ! grep -qE "(CHANGE|change|placeholder|example|TODO)" "$file"; then
                log_warn "$filename - May contain hardcoded secrets"
            fi
        fi
    fi
    
    # Check for missing resource limits
    if grep -q "kind: Deployment" "$file" || grep -q "kind: StatefulSet" "$file"; then
        if ! grep -q "resources:" "$file"; then
            log_warn "$filename - Missing resource limits/requests"
        fi
    fi
    
    # Check for missing health probes
    if grep -q "kind: Deployment" "$file"; then
        if ! grep -q "livenessProbe:" "$file" && ! grep -q "readinessProbe:" "$file"; then
            log_warn "$filename - Missing health probes"
        fi
    fi
    
    # Check for 'latest' tag
    if grep -qE "image:.*:latest" "$file"; then
        log_warn "$filename - Uses 'latest' tag (not recommended for production)"
    fi
}

# Validate individual manifest
validate_manifest() {
    local file=$1
    local filename
    filename=$(basename "$file")
    
    echo ""
    log_info "Validating: $filename"
    
    # Skip non-YAML files
    if [[ ! "$file" =~ \.(yaml|yml)$ ]]; then
        log_info "Skipping non-YAML file: $filename"
        return 0
    fi
    
    # Validate YAML syntax
    validate_yaml_syntax "$file" || true
    
    # Strict YAML validation
    if [ "$STRICT_MODE" = true ]; then
        validate_yaml_strict "$file" || true
        validate_k8s_schema "$file" || true
    fi
    
    # Check for common issues
    check_common_issues "$file"
}

# Validate kustomization
validate_kustomization() {
    log_info "Validating kustomization..."
    
    if [ -f "$K8S_DIR/kustomization.yaml" ]; then
        if kubectl kustomize "$K8S_DIR" > /dev/null 2>&1; then
            log_success "Kustomization build successful"
        else
            log_error "Kustomization build failed"
            kubectl kustomize "$K8S_DIR" 2>&1 | head -10
        fi
    else
        log_warn "No kustomization.yaml found"
    fi
}

# Main validation
main() {
    echo "========================================="
    echo "MLSDM Kubernetes Manifest Validation"
    echo "========================================="
    echo ""
    
    check_tools
    
    echo ""
    echo "-----------------------------------------"
    echo "Validating Kubernetes Manifests"
    echo "-----------------------------------------"
    
    # Validate all YAML files in k8s directory
    for file in "$K8S_DIR"/*.yaml "$K8S_DIR"/*.yml; do
        if [ -f "$file" ]; then
            validate_manifest "$file"
        fi
    done
    
    echo ""
    echo "-----------------------------------------"
    echo "Validating Monitoring Manifests"
    echo "-----------------------------------------"
    
    for file in "$MONITORING_DIR"/*.yaml "$MONITORING_DIR"/*.yml; do
        if [ -f "$file" ]; then
            validate_manifest "$file"
        fi
    done
    
    # Validate JSON files
    echo ""
    echo "-----------------------------------------"
    echo "Validating JSON Files"
    echo "-----------------------------------------"
    
    for file in "$MONITORING_DIR"/*.json "$GRAFANA_DIR"/*.json; do
        if [ -f "$file" ]; then
            echo ""
            log_info "Validating: $(basename "$file")"
            validate_json_syntax "$file" || true
        fi
    done
    
    echo ""
    echo "-----------------------------------------"
    echo "Validating Kustomization"
    echo "-----------------------------------------"
    
    if [ "$HAS_KUBECTL" = true ]; then
        validate_kustomization
    else
        log_warn "kubectl not found - skipping kustomization validation"
    fi
    
    # Summary
    echo ""
    echo "========================================="
    echo "Validation Summary"
    echo "========================================="
    echo "Errors:   $ERRORS"
    echo "Warnings: $WARNINGS"
    
    if [ $ERRORS -gt 0 ]; then
        echo ""
        log_error "Validation failed with $ERRORS error(s)"
        exit 1
    elif [ $WARNINGS -gt 0 ]; then
        echo ""
        log_warn "Validation passed with $WARNINGS warning(s)"
        exit 0
    else
        echo ""
        log_success "All validations passed!"
        exit 0
    fi
}

main "$@"
