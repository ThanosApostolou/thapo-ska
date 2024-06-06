llm_build:
    #!/usr/bin/env bash
    set -eu
    poetry build --format wheel

backend_run_server:
    #!/usr/bin/env bash
    set -eu
    export THAPO_SKA_CONF_DIR="ska_backend/distribution/etc/local"
    export THAPO_SKA_DATA_DIR="ska_backend/distribution/share"
    export THAPO_SKA_SECRET_FILE="ska_backend/.secret"
    source .venv/bin/activate
    cargo watch -p ska_backend -w "./ska_backend/src" -x 'run --bin app-server'

backend_run_cli:
    #!/usr/bin/env bash
    set -eu
    export THAPO_SKA_CONF_DIR="ska_backend/distribution/etc/local"
    export THAPO_SKA_DATA_DIR="ska_backend/distribution/share"
    export THAPO_SKA_SECRET_FILE="ska_backend/.secret"
    source .venv/bin/activate
    cargo run --bin app-cli -- "$@"

backend_build:
    #!/usr/bin/env bash
    set -eu
    cargo build -p ska_backend --release --bin app-server --bin app-cli

backend_pack:
    #!/usr/bin/env bash
    set -eu
    rm -rf dist/ska_backend
    mkdir -p dist/ska_backend/bin
    mkdir -p dist/ska_backend/etc
    mkdir -p dist/ska_backend/share/thapo_ska_py
    rsync -av ska_backend/distribution/bin/. dist/ska_backend/bin
    rsync -av ska_backend/distribution/etc/${THAPO_SKA_ENV_PROFILE}/. dist/ska_backend/etc/
    rsync -av target/release/app-server dist/ska_backend/bin/app-server
    rsync -av target/release/app-cli dist/ska_backend/bin/app-cli

    pip install --compile ./dist/thapo_ska-0.1.0-py3-none-any.whl -t dist/ska_backend/share/thapo_ska_py --extra-index-url https://download.pytorch.org/whl/cpu

frontend_run:
    #!/usr/bin/env bash
    set -eu
    cd ska_frontend
    npm run dev

iam_gateway_run:
    #!/usr/bin/env bash
    set -eu
    export THAPO_SKA_ENV_FILE=".env.local"
    cd "devops/scripts"
    python docker-compose.py up local -s thapo_ska_iam
    python docker-compose.py up local -s thapo_ska_gateway