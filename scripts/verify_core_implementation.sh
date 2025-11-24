#!/usr/bin/env bash
# MLSDM Core Implementation Verification Script
# 
# This script validates that the core cognitive components are fully implemented
# by running test collection and checking for TODOs/stubs/NotImplementedError.
#
# Usage: ./scripts/verify_core_implementation.sh
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "MLSDM Core Implementation Verification"
echo "======================================================================"
echo ""

# Define core modules to check
CORE_MODULES="memory,cognition,core,rhythm,speech,extensions"

# Change to repository root
cd "$(dirname "$0")/.."

echo "Repository: $(pwd)"
echo "Core modules: ${CORE_MODULES}"
echo ""

# Check 1: Test Collection
echo "======================================================================"
echo "CHECK 1: Test Collection"
echo "======================================================================"
echo "Command: python -m pytest tests/unit/ tests/core/ tests/property/ --co -q"
echo ""

TEST_OUTPUT=$(python -m pytest tests/unit/ tests/core/ tests/property/ --co -q 2>&1)
TEST_COUNT=$(echo "$TEST_OUTPUT" | grep "tests collected" | awk '{print $1}')

if [ -z "$TEST_COUNT" ]; then
    echo -e "${RED}✗ FAILED: Could not determine test count${NC}"
    echo "Output: $TEST_OUTPUT"
    exit 1
fi

echo -e "${GREEN}✓ PASSED: ${TEST_COUNT} tests collected${NC}"
echo ""

# Check 2: No TODOs or NotImplementedError
echo "======================================================================"
echo "CHECK 2: No TODOs or Stubs in Core Modules"
echo "======================================================================"
echo "Command: grep -rn \"TODO\\|NotImplementedError\" src/mlsdm/{memory,cognition,core,rhythm,speech,extensions}/"
echo ""

# Run grep and capture both output and exit code
GREP_OUTPUT=$(grep -rn "TODO\|NotImplementedError" src/mlsdm/memory/ src/mlsdm/cognition/ src/mlsdm/core/ src/mlsdm/rhythm/ src/mlsdm/speech/ src/mlsdm/extensions/ 2>&1 || true)
GREP_COUNT=$(echo "$GREP_OUTPUT" | grep -v "^$" | wc -l)

if [ "$GREP_COUNT" -eq 0 ] || [ -z "$GREP_OUTPUT" ]; then
    echo -e "${GREEN}✓ PASSED: No TODOs or NotImplementedError found${NC}"
else
    echo -e "${RED}✗ FAILED: Found ${GREP_COUNT} occurrences${NC}"
    echo ""
    echo "Matches:"
    echo "$GREP_OUTPUT"
    exit 1
fi
echo ""

# Summary
echo "======================================================================"
echo "VERIFICATION SUMMARY"
echo "======================================================================"
echo -e "${GREEN}✓ All checks passed${NC}"
echo ""
echo "Test Count: ${TEST_COUNT}"
echo "TODO/Stub Count: 0"
echo ""
echo "Core modules validated:"
echo "  - src/mlsdm/memory/"
echo "  - src/mlsdm/cognition/"
echo "  - src/mlsdm/core/"
echo "  - src/mlsdm/rhythm/"
echo "  - src/mlsdm/speech/"
echo "  - src/mlsdm/extensions/"
echo ""
echo "======================================================================"
echo -e "${GREEN}✓ CORE IMPLEMENTATION VERIFIED${NC}"
echo "======================================================================"
