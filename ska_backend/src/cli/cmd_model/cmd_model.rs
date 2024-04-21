use clap::{Args, Subcommand};

use crate::{
    cli::cmd_model::{
        cmd_download::handle_download, cmd_download_data::handle_download_data,
        cmd_insert::handle_insert, cmd_rag_prepare::handle_rag_prepare,
    },
    modules::global_state::GlobalState,
};

use super::cmd_rag_prepare::CmdRagPrepare;

#[derive(Args, Debug)]
pub struct CmdModel {
    #[command(subcommand)]
    command: ModelSubcommands,
}

#[derive(Subcommand, Debug)]
pub enum ModelSubcommands {
    /// does testing things
    Download,
    DownloadData,
    Insert,
    RagPrepare(CmdRagPrepare),
}

pub fn handle_model(global_state: &GlobalState, cmd_model: &CmdModel) {
    tracing::debug!("handle_model start");
    match &cmd_model.command {
        ModelSubcommands::Download => handle_download(global_state),
        ModelSubcommands::DownloadData => handle_download_data(global_state),
        ModelSubcommands::Insert => handle_insert(global_state),
        ModelSubcommands::RagPrepare(cmd_rag_prepare_args) => {
            handle_rag_prepare(global_state, &cmd_rag_prepare_args.emb_name)
        }
    }
    tracing::debug!("handle_model end");
}
