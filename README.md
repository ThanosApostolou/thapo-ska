# Specific Knowledge Assistant
specific knowledge assistant

## FRONTEND

### Preparation
```
rustup install nightly
rustup default nightly
rustup target add wasm32-unknown-unknown
cargo install trunk cargo-watch wasm-pack sea-orm-cli
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