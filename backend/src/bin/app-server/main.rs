use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::Router;
use backend::{modules::global_state::GlobalState, server};

#[tokio::main]
async fn main() {
    let dotenv_file = std::env::var("THAPO_SKA_ENV_FILE").unwrap_or_else(|_| ".env".to_string());
    dotenv::from_filename(&dotenv_file)
        .expect(&*format!("could not load file {}", dotenv_file.clone()));

    let global_state = GlobalState::initialize_default();
    let global_state = Arc::new(global_state);
    // initialize tracing
    tracing_subscriber::fmt::init();

    let server = server::create_server(global_state.clone());

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    // let host = Ipv4Addr::from(global_state.env_config.server_host);
    // let addr = SocketAddr::from((
    //     global_state.env_config.server_host,
    //     global_state.env_config.server_port,
    // ));
    let listen_addr = global_state.env_config.server_host.clone()
        + ":"
        + &global_state.env_config.server_port.to_string();
    tracing::info!("listening on listen_addr={}", listen_addr);

    let listener: tokio::net::TcpListener =
        tokio::net::TcpListener::bind(listen_addr).await.unwrap();
    axum::serve(listener, server).await.unwrap();
}
