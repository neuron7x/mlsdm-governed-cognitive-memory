#!/usr/bin/env bash
# ==============================================================================
# MLSDM Configuration File Example
# ==============================================================================
# Copy this file to 'mlsdm_config.sh' in the project root and customize
# the values for your environment.
#
# Usage:
#   cp mlsdm_config.example.sh mlsdm_config.sh
#   # Edit mlsdm_config.sh with your settings
#   source bin/mlsdm-env.sh
# ==============================================================================

# ==============================================================================
# ENVIRONMENT SETTINGS
# ==============================================================================

# Environment mode: development, staging, production
export MLSDM_ENV="${MLSDM_ENV:-development}"

# Log level: DEBUG, INFO, WARNING, ERROR
export MLSDM_LOG_LEVEL="${MLSDM_LOG_LEVEL:-INFO}"

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Configuration file path
export MLSDM_CONFIG_PATH="${MLSDM_CONFIG_PATH:-config/default_config.yaml}"

# Data directory for persistent storage
export MLSDM_DATA_DIR="${MLSDM_DATA_DIR:-./data}"

# ==============================================================================
# API CONFIGURATION
# ==============================================================================

# API host and port
export MLSDM_API_HOST="${MLSDM_API_HOST:-0.0.0.0}"
export MLSDM_API_PORT="${MLSDM_API_PORT:-8000}"

# API key for authentication (change in production!)
export MLSDM_API_KEY="${MLSDM_API_KEY:-your-secret-key-here}"

# ==============================================================================
# MEMORY CONFIGURATION
# ==============================================================================

# Vector dimension (must match your embedding model)
export MLSDM_DIMENSION="${MLSDM_DIMENSION:-384}"

# Memory capacity (number of vectors)
export MLSDM_CAPACITY="${MLSDM_CAPACITY:-20000}"

# ==============================================================================
# COGNITIVE RHYTHM CONFIGURATION
# ==============================================================================

# Wake phase duration (active processing steps)
export MLSDM_WAKE_DURATION="${MLSDM_WAKE_DURATION:-8}"

# Sleep phase duration (consolidation steps)
export MLSDM_SLEEP_DURATION="${MLSDM_SLEEP_DURATION:-3}"

# ==============================================================================
# MORAL FILTER CONFIGURATION
# ==============================================================================

# Initial moral threshold (0.0-1.0, higher = stricter)
export MLSDM_MORAL_THRESHOLD="${MLSDM_MORAL_THRESHOLD:-0.5}"

# Threshold bounds
export MLSDM_MORAL_MIN_THRESHOLD="${MLSDM_MORAL_MIN_THRESHOLD:-0.3}"
export MLSDM_MORAL_MAX_THRESHOLD="${MLSDM_MORAL_MAX_THRESHOLD:-0.9}"

# ==============================================================================
# SECURITY SETTINGS
# ==============================================================================

# Enable secure mode (blocks certain operations)
export MLSDM_SECURE_MODE="${MLSDM_SECURE_MODE:-0}"

# Rate limiting (0 = disabled, for testing only)
export MLSDM_DISABLE_RATE_LIMIT="${MLSDM_DISABLE_RATE_LIMIT:-0}"

# ==============================================================================
# NOTES
# ==============================================================================
# - Never commit mlsdm_config.sh with real credentials to version control
# - Add mlsdm_config.sh to .gitignore
# - Use secrets management in production (AWS Secrets Manager, HashiCorp Vault)
# - See DEPLOYMENT_GUIDE.md for production best practices
