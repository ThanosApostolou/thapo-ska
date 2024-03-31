use std::collections::HashMap;

use sea_orm::DatabaseConnection;
use tracing_subscriber::prelude::*;

use crate::modules::{
    auth::{
        auth_models::AuthTypes,
        service_auth::{self, MyAuthClient},
    },
    db::init_db_connection,
    web::routes::Routes,
};

use super::{EnvConfig, SecretConfig};

#[derive(Clone)]
pub struct GlobalState {
    pub env_config: EnvConfig,
    pub secret_config: SecretConfig,
    pub db_connection: DatabaseConnection,
    pub auth_client: MyAuthClient,
    pub routes: Routes,
    pub routes_map: HashMap<String, AuthTypes>,
}

impl GlobalState {
    pub async fn initialize_default() -> anyhow::Result<GlobalState> {
        let env_config = EnvConfig::from_env();

        // initialize tracing
        let file_appender = tracing_appender::rolling::daily(
            env_config.ska_user_conf_dir.clone() + "/logs",
            "ska_backend.log",
        );
        let (file_writter, _guard) = tracing_appender::non_blocking(file_appender);
        let log_layer_stdout = tracing_subscriber::fmt::layer().pretty();
        let log_layer_file = tracing_subscriber::fmt::layer()
            .with_writer(file_writter)
            .with_ansi(false);
        let subscriber = tracing_subscriber::Registry::default()
            .with(log_layer_stdout)
            .with(log_layer_file);
        tracing::subscriber::set_global_default(subscriber)
            .expect("Unable to set global subscriber");

        tracing::info!("EnvConfig: {:?}", env_config);

        let secret_config = SecretConfig::from_env();
        let db_connection = init_db_connection(&env_config, &secret_config)
            .await
            .unwrap();

        let auth_client = service_auth::create_oidc_client(&env_config, &secret_config).await?;
        let routes = Routes::new();
        let routes_map = routes.routes_auth_types_map(env_config.server_path.clone());

        Ok(GlobalState {
            env_config,
            secret_config,
            db_connection,
            auth_client,
            routes,
            routes_map,
        })
    }

    pub async fn initialize_cli() -> anyhow::Result<GlobalState> {
        let env_config = EnvConfig::from_env();

        // initialize tracing
        let file_appender = tracing_appender::rolling::daily(
            env_config.ska_user_conf_dir.clone() + "/logs",
            "ska_cli.log",
        );
        let (file_writter, _guard) = tracing_appender::non_blocking(file_appender);
        let log_layer_stdout = tracing_subscriber::fmt::layer().pretty();
        let log_layer_file = tracing_subscriber::fmt::layer()
            .with_writer(file_writter)
            .with_ansi(false);
        let subscriber = tracing_subscriber::Registry::default()
            .with(log_layer_stdout)
            .with(log_layer_file);
        tracing::subscriber::set_global_default(subscriber)
            .expect("Unable to set global subscriber");

        let secret_config = SecretConfig::from_env();
        let db_connection = init_db_connection(&env_config, &secret_config)
            .await
            .unwrap();

        let auth_client = service_auth::create_oidc_client(&env_config, &secret_config).await?;
        let routes = Routes::new();
        let routes_map = routes.routes_auth_types_map(env_config.server_path.clone());

        Ok(GlobalState {
            env_config,
            secret_config,
            db_connection,
            auth_client,
            routes,
            routes_map,
        })
    }
}
