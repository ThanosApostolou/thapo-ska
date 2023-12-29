FROM docker.io/rust:1.74.1 as build
ARG ARG_THAPO_SKA_ENV_PROFILE
WORKDIR /app/backend
COPY ../Cargo.toml /app
COPY ../Cargo.lock /app
COPY ./Cargo.toml /app/backend
COPY ./src /app/backend/src
RUN cargo build --release --features=feature-server --bin app-server
RUN cargo build --release --features=feature-cli --bin app-cli

FROM docker.io/rust:1.74.1
ARG ARG_THAPO_SKA_ENV_PROFILE
ENV THAPO_SKA_ENV_PROFILE=ARG_THAPO_SKA_ENV_PROFILE
WORKDIR /app/backend
COPY --from=build /app/target/release/app-server /app/app-server
COPY --from=build /app/target/release/app-cli /app/app-cli
CMD THAPO_SKA_ENV_PROFILE=${ARG_THAPO_SKA_ENV_PROFILE} /app-server