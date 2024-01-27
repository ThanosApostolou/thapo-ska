#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

if [ ! $# -eq 1 ]
  then
    echo "Error: Usage: bash helm-build.sh profile(local/dev/prod)"
    exit 1
fi

profile="$1"
if ! [ "$profile" = "local" ] && ! [ "$profile" = "dev" ] && ! [ "$profile" = "prod" ]; then
    echo "profile should be local/dev/prod"
    exit 1
fi

function ctr_import() {
    image="$1"
    sudo docker save "$image" | sudo ctr -a /var/run/k3s/containerd/containerd.sock -n k8s.io images import -
}


cd "$SCRIPT_DIR/../../ska_backend"
backend_version="$(echo "$(cargo pkgid)" | rev | cut -d'#' -f1 | rev)"
ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_backend:$backend_version-$profile"


cd "$SCRIPT_DIR/../../frontend"
frontend_version="$(echo "$(cargo pkgid)" | rev | cut -d'#' -f1 | rev)"
ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_frontend:$frontend_version-$profile"

ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_gateway:$frontend_version-$profile"
ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_iam:$frontend_version-$profile"
