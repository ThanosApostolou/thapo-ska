use std::sync::Arc;

use reqwest::Client;

use super::EnvConfig;

#[derive(Clone, Debug)]
pub struct GlobalState {
    pub env_config: EnvConfig,
    pub api_client: Arc<Client>,
}

impl GlobalState {
    pub fn initialize_default() -> GlobalState {
        let env_config = EnvConfig::from_env();
        let api_client = Arc::new(Client::builder().build().unwrap_or_default());
        GlobalState {
            env_config,
            api_client,
        }
    }
}
