#!/usr/bin/env bash
# Dev smoke tests - Quick sanity check for local development
# This script runs the same tests that CI runs, ensuring consistency

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MLSDM Dev Smoke Tests ===${NC}"
echo ""

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Run this script from the repository root.${NC}"
    exit 1
fi

# Run the canonical test command
echo -e "${YELLOW}Running tests with canonical command:${NC}"
echo "PYTHONPATH=src pytest -q --ignore=tests/load"
echo ""

PYTHONPATH=src pytest -q --ignore=tests/load

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All smoke tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Smoke tests failed${NC}"
    exit 1
fi
