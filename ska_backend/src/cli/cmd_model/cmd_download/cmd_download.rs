use crate::modules::global_state::GlobalState;

use super::do_download;

// #[derive(Args)]
// pub struct CmdDownload {}

pub fn handle_download(global_state: &GlobalState) {
    tracing::debug!("cmd_download.handle_download start");
    do_download(global_state);
    tracing::debug!("cmd_download.handle_download end");
}
