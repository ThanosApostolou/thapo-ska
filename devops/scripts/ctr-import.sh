#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"


function ctr_import() {
    image="$1"
    echo "image: $image"
    sudo docker save "$image" | sudo ctr -a /var/run/k3s/containerd/containerd.sock -n k8s.io images import -
}


cd "$SCRIPT_DIR/../../ska_backend"
backend_version="$(echo "$(cargo pkgid)" | rev | cut -d'#' -f1 | rev)"
ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_backend:$backend_version-local_kube"


cd "$SCRIPT_DIR/../../ska_frontend"
frontend_version="$(echo "$(cargo pkgid)" | rev | cut -d'#' -f1 | rev)"
ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_frontend:$frontend_version-local_kube"

ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_gateway:$frontend_version-local_kube"
ctr_import "registry.thapo-dev.org:5000/thapo/thapo_ska_iam:$frontend_version-local_kube"
