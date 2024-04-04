#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

export THAPO_SKA_CONF_DIR="ska_backend/distribution/etc/local"
export THAPO_SKA_SECRET_FILE="ska_backend/.secret"
source .venv/bin/activate
cargo run -p ska_backend --bin app-cli -- "$@"
