#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

if [ ! $# -eq 0 ]
  then
    echo "Error: Usage: bash serve.sh"
    exit 1
fi

export THAPO_SKA_ENV_FILE=".env.local"
cd "$SCRIPT_DIR/../devops/scripts"
python docker-compose.py up local -s thapo_ska_gateway