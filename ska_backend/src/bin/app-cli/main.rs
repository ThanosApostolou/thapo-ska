use ska_backend::modules::global_state::GlobalState;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let dotenv_file = std::env::var("THAPO_SKA_ENV_FILE").unwrap_or_else(|_| ".env".to_string());
    dotenv::from_filename(&dotenv_file)
        .unwrap_or_else(|_| panic!("could not load file {}", dotenv_file.clone()));
    let secret_file =
        std::env::var("THAPO_SKA_SECRET_FILE").unwrap_or_else(|_| ".secret".to_string());
    dotenv::from_filename(&secret_file)
        .unwrap_or_else(|_| panic!("could not load file {}", secret_file.clone()));
    // initialize tracing
    tracing_subscriber::fmt::init();

    let global_state = GlobalState::initialize_default().await.unwrap();
    let global_state = Arc::new(global_state);

    println!("hello from cli")
}
