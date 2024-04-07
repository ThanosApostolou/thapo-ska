use crate::modules::global_state::GlobalState;

use super::do_download;

pub fn handle_download(global_state: &GlobalState) {
    tracing::debug!("handle_download start");
    let res = do_download(global_state);
    match res {
        Ok(_) => tracing::debug!("handle_download end"),
        Err(err) => {
            tracing::error!("handle_download end with error {}", err);
            std::process::exit(1);
        }
    }
}
