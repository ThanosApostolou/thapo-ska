use crate::modules::global_state::GlobalState;

use super::do_migrate;

pub async fn handle_migrate(global_state: &GlobalState) {
    tracing::debug!("handle_migrate start");
    let res = do_migrate(global_state).await;
    match res {
        Ok(_) => tracing::debug!("handle_migrate end"),
        Err(err) => {
            tracing::error!("handle_migrate end with error {}", err);
            std::process::exit(1);
        }
    }
}
