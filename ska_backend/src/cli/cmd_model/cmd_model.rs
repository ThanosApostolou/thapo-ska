use clap::{Args, Subcommand};

use crate::{
    cli::cmd_model::{
        cmd_create_skalm::handle_create_skalm, cmd_download::handle_download,
        cmd_insert::handle_insert, cmd_rag_invoke::handle_rag_invoke,
        cmd_rag_prepare::handle_rag_prepare,
    },
    modules::global_state::GlobalState,
};

use super::{cmd_rag_invoke::CmdRagInvokeArgs, cmd_rag_prepare::CmdRagPrepareArgs};

#[derive(Args, Debug)]
pub struct CmdModelArgs {
    #[command(subcommand)]
    command: ModelSubcommands,
}

#[derive(Subcommand, Debug)]
pub enum ModelSubcommands {
    /// downloads models and LLMs for RAG
    Download,
    /// inserts the downloaded LLMs in the systems files location
    Insert,
    /// prepares the documents and vector store for RAG
    RagPrepare(CmdRagPrepareArgs),
    /// invokes an LLM with a question
    RagInvoke(CmdRagInvokeArgs),
    /// creates and trains SKA text generation model
    CreateSkalm,
}

pub async fn handle_model(global_state: &GlobalState, cmd_model: &CmdModelArgs) {
    tracing::debug!("handle_model start");
    match &cmd_model.command {
        ModelSubcommands::Download => handle_download(global_state),
        ModelSubcommands::Insert => handle_insert(global_state),
        ModelSubcommands::RagPrepare(cmd_rag_prepare_args) => {
            handle_rag_prepare(global_state, &cmd_rag_prepare_args.emb_name)
        }
        ModelSubcommands::RagInvoke(cmd_rag_invoke_args) => {
            handle_rag_invoke(
                global_state,
                &cmd_rag_invoke_args.emb_name,
                &cmd_rag_invoke_args.llm_name,
                &cmd_rag_invoke_args.question,
                &cmd_rag_invoke_args.prompt_template,
            )
            .await
        }
        ModelSubcommands::CreateSkalm => handle_create_skalm(global_state),
    }
    tracing::debug!("handle_model end");
}
