#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

DEFAULT_THAPO_SKA_CONF_DIR="$(realpath "$SCRIPT_DIR/../etc")"
export THAPO_SKA_CONF_DIR="${THAPO_SKA_CONF_DIR:-$DEFAULT_THAPO_SKA_CONF_DIR}"
export PYTHONPATH="$(realpath "$SCRIPT_DIR/../share/thapo_ska_py")"
"$SCRIPT_DIR/app-cli" "$@"
