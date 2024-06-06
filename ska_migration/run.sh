#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

source ".secret"
export DATABASE_URL="postgres://$THAPO_SKA_DB_USER:$THAPO_SKA_DB_PASSWORD@localhost:5432/thapo_ska_db_local"
export THAPO_SKA_DATA_DIR="$SCRIPT_DIR/../ska_backend/distribution/share"

# cargo run -- -s thapo_ska_schema $@

if [ ! $# -eq 1 ]
  then
    echo "Error: Expected arguments: action(up,down,generate)"
    exit 1
fi

action="$1"
if [ ! "$action" = "up" ]  && [ ! "$action" = "down" ] && [ ! "$action" = "generate" ]  ; then
    echo "wrong action $action"
    exit 1
fi


generate_entities() {
    sea-orm-cli generate entity -s thapo_ska_schema --with-serde both -o ../ska_backend/src/domain/entities
}

migrate_up() {
    sea-orm-cli migrate -d .  -s thapo_ska_schema up
}

migrate_down() {
    sea-orm-cli migrate -d .  -s thapo_ska_schema down
}


if [ "$action" = "up" ]; then
    migrate_up
elif [ "$action" = "down" ]; then
    migrate_down
elif [ "$action" = "generate" ]; then
    generate_entities
fi