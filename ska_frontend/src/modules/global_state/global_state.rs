use std::sync::Arc;

use crate::modules::auth::auth_service;
use leptos::{expect_context, ReadSignal};

use reqwest::Client;

use super::EnvConfig;

#[derive(Clone, Debug)]
pub struct GlobalState {
    pub env_config: EnvConfig,
    pub api_client: Arc<Client>,
    pub oidc_provider_metadata: auth_service::MyProviderMetadata,
    pub oidc_client: auth_service::MyOidcClient,
}

impl GlobalState {
    pub async fn initialize_default() -> GlobalState {
        let env_config = EnvConfig::from_env();
        let api_client = Arc::new(Client::builder().build().unwrap_or_default());

        let (oidc_provider_metadata, oidc_client) =
            auth_service::create_oidc_client(&env_config).await.unwrap();
        GlobalState {
            env_config,
            api_client,
            oidc_provider_metadata,
            oidc_client,
        }
    }

    pub fn expect_context() -> ReadSignal<GlobalState> {
        expect_context::<ReadSignal<GlobalState>>()
    }
}
