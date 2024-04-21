use clap::Args;

use crate::modules::global_state::GlobalState;

use super::do_rag_prepare;

#[derive(Args, Debug)]
pub struct CmdRagPrepare {
    #[arg(short, long)]
    pub emb_name: String,
}

pub fn handle_rag_prepare(global_state: &GlobalState, embedding_model_name: &String) {
    tracing::debug!("handle_rag_prepare start");
    let res = do_rag_prepare(global_state, embedding_model_name);
    match res {
        Ok(_) => tracing::debug!("handle_rag_prepare end"),
        Err(err) => {
            tracing::error!("handle_rag_prepare end with error {}", err);
            std::process::exit(1);
        }
    }
}
