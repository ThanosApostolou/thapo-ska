#!/bin/bash
#!/bin/bash
set -eu
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

source ".secret"
export DATABASE_URL="postgres://$THAPO_SKA_DB_USER:$THAPO_SKA_DB_PASSWORD@localhost:5432/thapo_ska_db_local"

# cargo run -- -s thapo_ska_schema $@

generate_entities() {
    sea-orm-cli generate entity -s thapo_ska_schema --with-serde both -o ../ska_backend/src/domain/entities
}

generate_entities