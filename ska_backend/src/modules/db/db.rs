use sea_orm::{
    AccessMode, ConnectOptions, DatabaseConnection, DatabaseTransaction, DbErr, IsolationLevel,
    TransactionTrait,
};
use ska_migration::{Migrator, MigratorTrait};

use crate::modules::{
    error::{ErrorCode, ErrorPacket, ErrorResponse},
    global_state::{EnvConfig, GlobalState, SecretConfig},
};

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

pub async fn transaction_begin_write(
    global_state: &GlobalState,
) -> Result<DatabaseTransaction, ErrorResponse> {
    let txn = global_state
        .db_connection
        .begin_with_config(
            Some(IsolationLevel::RepeatableRead),
            Some(AccessMode::ReadWrite),
        )
        .await
        .map_err(|e| {
            return ErrorResponse {
                error_code: ErrorCode::InternalServerError500,
                is_unexpected_error: true,
                packets: vec![ErrorPacket::new_backend(e.to_string().as_str())],
            };
        })?;
    return Ok(txn);
}

pub async fn transaction_begin_read(
    global_state: &GlobalState,
) -> Result<DatabaseTransaction, ErrorResponse> {
    let txn = global_state
        .db_connection
        .begin_with_config(
            Some(IsolationLevel::ReadCommitted),
            Some(AccessMode::ReadOnly),
        )
        .await
        .map_err(|e| {
            return ErrorResponse {
                error_code: ErrorCode::InternalServerError500,
                is_unexpected_error: true,
                packets: vec![ErrorPacket::new_backend(e.to_string().as_str())],
            };
        })?;
    return Ok(txn);
}

pub async fn transaction_commit(txn: DatabaseTransaction) -> Result<(), ErrorResponse> {
    txn.commit().await.map_err(|e| {
        return ErrorResponse {
            error_code: ErrorCode::InternalServerError500,
            is_unexpected_error: true,
            packets: vec![ErrorPacket::new_backend(e.to_string().as_str())],
        };
    })?;
    return Ok(());
}

pub async fn transaction_rollback(txn: DatabaseTransaction) -> Result<(), ErrorResponse> {
    txn.rollback().await.map_err(|e| {
        return ErrorResponse {
            error_code: ErrorCode::InternalServerError500,
            is_unexpected_error: true,
            packets: vec![ErrorPacket::new_backend(e.to_string().as_str())],
        };
    })?;
    return Ok(());
}
