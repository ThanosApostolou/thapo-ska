#!/bin/sh
set -e

# export $(cat "${THAPO_SKA_SECRET_FILE}" | xargs)
set -o allexport
source "${THAPO_SKA_SECRET_FILE}"
set +o allexport
echo "commands $@"
/opt/keycloak/bin/kc.sh "$@"