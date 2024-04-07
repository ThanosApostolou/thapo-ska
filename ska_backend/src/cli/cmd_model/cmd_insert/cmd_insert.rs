use crate::modules::global_state::GlobalState;

use super::do_insert;

pub fn handle_insert(global_state: &GlobalState) {
    tracing::debug!("handle_insert start");
    let res = do_insert(global_state);
    match res {
        Ok(_) => tracing::debug!("handle_insert end"),
        Err(err) => {
            tracing::error!("handle_insert end with error {}", err);
            std::process::exit(1);
        }
    }
}
