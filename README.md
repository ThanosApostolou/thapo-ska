# Specific Knowledge Assistant
specific knowledge assistant

## FRONTEND

### Preparation
```
rustup install nightly
rustup default nightly
rustup target add wasm32-unknown-unknown
cargo install trunk cargo-watch wasm-pack sea-orm-cli just
npm install -g tailwindcss@3.4.0 daisyui@4.4.24
```

### Run
```
trunk serve --open
```

### Test
```
wasm-pack test --node
```

## RAG

### prod env
1. copy data to /mnt/data/container-data/local-path-provisioner/thapo-ska/prod/data
2. run
```
sudo chown -R 14000:14000 /mnt/data/container-data/local-path-provisioner/thapo-ska/prod/data
```
3. rag
```
sudo kubectl -n thapo-ska-prod exec --stdin --tty deployment.apps/deployment-skabackend -- /bin/bash
```

inside shell
```
./bin/app-cli.sh model download
./bin/app-cli.sh model insert
./bin/app-cli.sh model rag-prepare --emb-name all-MiniLM-L6-v2
```