use ska_backend::prelude::*;

use ska_backend::cli::start_cli;
use ska_backend::modules::global_state::GlobalState;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let secret_file =
        std::env::var("THAPO_SKA_SECRET_FILE").unwrap_or_else(|_| ".secret".to_string());
    dotenv::from_filename(&secret_file)
        .unwrap_or_else(|_| panic!("could not load file {}", secret_file.clone()));

    let global_state = GlobalState::initialize_cli().await.unwrap();
    let global_state = Arc::new(global_state);

    start_cli(&global_state).await;
}
