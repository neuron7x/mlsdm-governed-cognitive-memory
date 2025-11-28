#!/usr/bin/env bash
# ==============================================================================
# MLSDM Environment Initialization Script
# ==============================================================================
# This script initializes the MLSDM environment by loading configuration from
# mlsdm_config.sh in the project root.
#
# Usage:
#   source bin/mlsdm-env.sh
#   # or
#   ./bin/mlsdm-env.sh  (if you need to execute it directly)
#
# Features:
#   - Context-independent: works from any directory (uses BASH_SOURCE)
#   - Defensive programming: validates config file thoroughly
#   - Cognitive feedback: clear error messages with instructions
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ==============================================================================
# 1. CONTEXT RESOLUTION
# Визначаємо шлях до скрипта, щоб знайти конфіг незалежно від місця запуску
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="mlsdm_config.sh"
CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"

# ==============================================================================
# 2. DEFENSIVE LOADING & FEEDBACK
# ==============================================================================
if [ -f "$CONFIG_PATH" ] && [ -r "$CONFIG_PATH" ]; then
    
    # Попередження, якщо файл порожній, але не блокуємо роботу жорстко, якщо це не критично
    if [ ! -s "$CONFIG_PATH" ]; then
        echo "⚠️  [MLSDM] WARNING: Config file is empty: $CONFIG_PATH"
    fi

    # shellcheck source=/dev/null
    source "$CONFIG_PATH"
    # echo "✅ [MLSDM] Loaded config: $CONFIG_PATH" # Uncomment for verbose mode

else
    echo "🛑 [MLSDM] CRITICAL ERROR: Cannot load configuration."
    echo "   Expected path: $CONFIG_PATH"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "   [Reason]: File not found."
        echo "   [Fix]: Run 'cp mlsdm_config.example.sh mlsdm_config.sh' in the project root."
    elif [ ! -r "$CONFIG_PATH" ]; then
        echo "   [Reason]: Permission denied."
        echo "   [Fix]: Run 'chmod +r $CONFIG_PATH'."
    fi
    
    exit 1
fi
