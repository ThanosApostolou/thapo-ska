use sea_orm::{ConnectOptions, DatabaseConnection, DbErr};
use ska_migration::{Migrator, MigratorTrait};

use crate::modules::global_state::{EnvConfig, SecretConfig};

pub async fn init_db_connection(
    env_config: &EnvConfig,
    secret_config: &SecretConfig,
) -> Result<DatabaseConnection, DbErr> {
    let database_url = format!(
        "postgres://{}:{}@{}:{}/{}",
        secret_config.db_user,
        secret_config.db_password,
        env_config.db_host,
        env_config.db_port,
        env_config.db_database
    );
    let mut connect_options = ConnectOptions::new(database_url);
    connect_options
        // .max_connections(100)
        //     .min_connections(5)
        //     .connect_timeout(Duration::from_secs(8))
        //     .acquire_timeout(Duration::from_secs(8))
        //     .idle_timeout(Duration::from_secs(8))
        //     .max_lifetime(Duration::from_secs(8))
        //     .sqlx_logging(true)
        //     .sqlx_logging_level(log::LevelFilter::Info)
        .set_schema_search_path(env_config.db_schema.clone()); // Setting default PostgreSQL schema
    let connection = sea_orm::Database::connect(connect_options).await?;
    Ok(connection)
}

pub async fn migrate_db(connection: &DatabaseConnection) -> Result<(), ska_migration::DbErr> {
    tracing::debug!("db::migrate_db start");
    Migrator::up(connection, None).await?;
    Migrator::status(connection).await?;
    Ok(())
}
