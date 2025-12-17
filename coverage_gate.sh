#!/bin/bash
# ============================================================================
# coverage_gate.sh - Coverage Gate Script for MLSDM
# ============================================================================
# SPDX-License-Identifier: MIT
#
# This script runs tests with coverage measurement and enforces a minimum
# coverage threshold. It is designed to work in CI and locally.
#
# Usage:
#   ./coverage_gate.sh               # Uses default threshold (65%)
#   COVERAGE_MIN=80 ./coverage_gate.sh  # Uses 80% threshold
#
# Exit codes:
#   0 - Tests passed and coverage meets threshold
#   1 - Tests failed or coverage below threshold
#
# Environment Variables:
#   COVERAGE_MIN - Minimum coverage percentage required (default: 65)
#   PYTEST_ARGS  - Additional arguments to pass to pytest
# ============================================================================

set -euo pipefail

# Default coverage threshold (can be overridden via environment variable)
# Set to match CI gate threshold for consistency
# CI workflow uses --cov-fail-under=65 in ci-neuro-cognitive-engine.yml
COVERAGE_MIN="${COVERAGE_MIN:-65}"

# Additional pytest arguments (can be extended via environment variable)
PYTEST_ARGS="${PYTEST_ARGS:-}"

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

echo "========================================================================"
echo "MLSDM Coverage Gate"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  COVERAGE_MIN: ${COVERAGE_MIN}%"
echo "  PYTEST_ARGS:  ${PYTEST_ARGS:-<none>}"
echo ""

# Change to repository root (script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Run pytest with coverage
echo "========================================================================"
echo "Running tests with coverage..."
echo "========================================================================"
echo ""

# Build pytest command
# We run unit and state tests as the primary gate, but allow PYTEST_ARGS to customize
#
# NOTE: We use `|| true` after the pipeline because with `set -e` and `pipefail`,
# a non-zero exit from pytest would terminate the script before we can capture
# PIPESTATUS. We explicitly check the exit code below to fail appropriately.
#
# State tests are included because they are unit-level tests for state modules,
# just organized in a separate directory for clarity.
#
# shellcheck disable=SC2086
python -m pytest tests/unit/ tests/state/ \
    --cov=src/mlsdm \
    --cov-report=term-missing \
    --cov-report=xml:coverage.xml \
    --cov-fail-under=0 \
    -m "not slow and not benchmark and not comprehensive" \
    -q \
    --tb=short \
    $PYTEST_ARGS 2>&1 | tee /tmp/coverage_output.txt || true

# Capture exit code from pytest (first command in pipeline)
PYTEST_EXIT_CODE="${PIPESTATUS[0]}"

if [ "$PYTEST_EXIT_CODE" -ne 0 ]; then
    echo ""
    printf "${RED}ERROR: Tests failed with exit code %d${NC}\n" "$PYTEST_EXIT_CODE"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Extracting coverage percentage..."
echo "========================================================================"

# Extract coverage percentage from the output
# The coverage report outputs a line like: "TOTAL    8901   2631   2296    224  68.25%"
COVERAGE_PERCENT=$(grep "^TOTAL" /tmp/coverage_output.txt | awk '{print $NF}' | sed 's/%//')

if [ -z "$COVERAGE_PERCENT" ]; then
    echo ""
    printf "${RED}ERROR: Could not extract coverage percentage from output${NC}\n"
    echo "Expected a line like: TOTAL    8901   2631   2296    224  68.25%"
    exit 1
fi

echo ""
echo "Coverage: ${COVERAGE_PERCENT}%"
echo "Threshold: ${COVERAGE_MIN}%"
echo ""

# Compare coverage to threshold (using awk for floating point comparison)
PASS=$(echo "$COVERAGE_PERCENT $COVERAGE_MIN" | awk '{if ($1 >= $2) print "1"; else print "0"}')

if [ "$PASS" = "1" ]; then
    echo "========================================================================"
    printf "${GREEN}✓ COVERAGE GATE PASSED${NC}\n"
    echo "========================================================================"
    echo ""
    echo "Coverage ${COVERAGE_PERCENT}% meets minimum threshold of ${COVERAGE_MIN}%"
    exit 0
else
    echo "========================================================================"
    printf "${RED}✗ COVERAGE GATE FAILED${NC}\n"
    echo "========================================================================"
    echo ""
    printf "${RED}Coverage ${COVERAGE_PERCENT}%% is below minimum threshold of ${COVERAGE_MIN}%%${NC}\n"
    echo ""
    printf "${YELLOW}To view uncovered lines, run:${NC}\n"
    echo "  python -m pytest tests/unit/ --cov=src/mlsdm --cov-report=term-missing"
    echo ""
    printf "${YELLOW}To adjust the threshold, set COVERAGE_MIN:${NC}\n"
    echo "  COVERAGE_MIN=${COVERAGE_PERCENT%.*} ./coverage_gate.sh"
    exit 1
fi
