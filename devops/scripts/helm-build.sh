#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

if [ ! $# -eq 2 ]
  then
    echo "Error: Usage: bash helm-build.sh profile(local/dev/prod) action(install/uninstall)"
    exit 1
fi

profile="$1"
if ! [ "$profile" = "local" ] && ! [ "$profile" = "dev" ] && ! [ "$profile" = "prod" ]; then
    echo "profile should be local/dev/prod"
    exit 1
fi

action="$2"
if ! [ "$action" = "install" ] && ! [ "$action" = "uninstall" ]; then
    echo "action should be install/uninstall"
    exit 1
fi

if [ "$action" = "install" ]; then
    sudo helm uninstall -n "thapo-ska-$profile" "thapo-ska-$profile" --ignore-not-found --wait
    cd "$SCRIPT_DIR/../kubernetes/thapo-ska"
    sudo helm install -f "values-$profile.yaml" --create-namespace -n "thapo-ska-$profile" "thapo-ska-$profile" .
elif [ "$action" = "uninstall" ]; then
    sudo helm uninstall -n "thapo-ska-$profile" "thapo-ska-$profile" --ignore-not-found --wait
    sudo kubectl delete ns "thapo-ska-$profile" --wait --ignore-not-found
fi