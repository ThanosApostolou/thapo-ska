use clap::Args;

use crate::modules::global_state::GlobalState;

use super::action_rag_invoke;

#[derive(Args, Debug)]
pub struct CmdRagInvokeArgs {
    #[arg(short, long)]
    pub emb_name: String,
    #[arg(short, long)]
    pub llm_name: String,
    #[arg(short, long)]
    pub question: String,
    #[arg(short, long)]
    pub prompt_template: Option<String>,
}

pub fn handle_rag_invoke(
    global_state: &GlobalState,
    emb_name: &String,
    llm_name: &String,
    question: &String,
    prompt_template: &Option<String>,
) {
    tracing::debug!("handle_rag_invoke start");
    let res = action_rag_invoke(global_state, emb_name, llm_name, question, prompt_template);
    match res {
        Ok(_) => tracing::debug!("handle_rag_invoke end"),
        Err(err) => {
            tracing::error!("handle_rag_invoke end with error {}", err);
            std::process::exit(1);
        }
    }
}
