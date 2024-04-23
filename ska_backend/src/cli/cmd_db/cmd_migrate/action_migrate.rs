use crate::modules::db;
use crate::modules::global_state::GlobalState;

pub async fn do_migrate(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_migrate start");

    db::migrate_db(&global_state.db_connection).await?;
    tracing::trace!("do_migrate end");
    Ok(())
}
