use sea_orm::DatabaseConnection;

use crate::modules::{
    auth::service_auth::{self, MyOidcClient, MyProviderMetadata},
    db::init_db_connection,
};

use super::{EnvConfig, SecretConfig};

#[derive(Clone)]
pub struct GlobalState {
    pub env_config: EnvConfig,
    pub secret_config: SecretConfig,
    pub db_connection: DatabaseConnection,
    pub provider_metadata: MyProviderMetadata,
    pub oidc_client: MyOidcClient,
}

impl GlobalState {
    pub async fn initialize_default() -> anyhow::Result<GlobalState> {
        let env_config = EnvConfig::from_env();
        let secret_config = SecretConfig::from_env();
        let db_connection = init_db_connection(&env_config, &secret_config)
            .await
            .unwrap();

        let (provider_metadata, oidc_client) =
            service_auth::create_oidc_client(&env_config, &secret_config).await?;
        Ok(GlobalState {
            env_config,
            secret_config,
            db_connection,
            provider_metadata,
            oidc_client,
        })
    }
}
