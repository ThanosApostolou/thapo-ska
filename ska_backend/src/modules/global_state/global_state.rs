use sea_orm::DatabaseConnection;

use crate::modules::db::init_db_connection;

use super::{EnvConfig, SecretConfig};

#[derive(Clone)]
pub struct GlobalState {
    pub env_config: EnvConfig,
    pub secret_config: SecretConfig,
    pub db_connection: DatabaseConnection,
}

impl GlobalState {
    pub async fn initialize_default() -> GlobalState {
        let env_config = EnvConfig::from_env();
        let secret_config = SecretConfig::from_env();
        let db_connection = init_db_connection(&env_config, &secret_config)
            .await
            .unwrap();
        GlobalState {
            env_config,
            secret_config,
            db_connection,
        }
    }
}
