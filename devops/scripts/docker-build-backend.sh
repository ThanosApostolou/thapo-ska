#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

if [ ! $# -eq 1 ]
  then
    echo "Error: Expected arguments: Profile(local,dev)"
    exit 1
fi

THAPO_SKA_ENV_PROFILE="$1"
if [ ! "$THAPO_SKA_ENV_PROFILE" = "local" ]  && [ ! "$THAPO_SKA_ENV_PROFILE" = "dev" ]  ; then
    echo "wrong profile"
    exit 1
fi
cd "$SCRIPT_DIR/../../backend"
version="$(echo "$(cargo pkgid)" | rev | cut -d'#' -f1 | rev)"

cd "$SCRIPT_DIR/../.."
sudo docker build -f devops/docker/images/Dockerfile.backend --no-cache --progress=plain --build-arg="ARG_THAPO_SKA_ENV_FILE=.env.$THAPO_SKA_ENV_PROFILE" -t "thanosapostolou/thapo_ska_backend:${version}-${THAPO_SKA_ENV_PROFILE}" .
# --no-cache