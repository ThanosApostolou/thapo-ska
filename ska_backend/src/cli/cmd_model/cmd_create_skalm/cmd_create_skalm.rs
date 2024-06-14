use crate::modules::global_state::GlobalState;

use super::do_create_skalm;

pub fn handle_create_skalm(global_state: &GlobalState) {
    tracing::debug!("handle_create_skalm start");
    let res = do_create_skalm(global_state);
    match res {
        Ok(_) => tracing::debug!("handle_create_skalm end"),
        Err(err) => {
            tracing::error!("handle_create_skalm end with error {}", err);
            std::process::exit(1);
        }
    }
}
