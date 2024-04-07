use clap::{Args, Subcommand};

use crate::{
    cli::cmd_model::{cmd_download::handle_download, cmd_insert::handle_insert},
    modules::global_state::GlobalState,
};

#[derive(Args, Debug)]
pub struct CmdModel {
    #[command(subcommand)]
    command: ModelSubcommands,
}

#[derive(Subcommand, Debug)]
pub enum ModelSubcommands {
    /// does testing things
    Download,
    Insert,
}

pub fn handle_model(global_state: &GlobalState, cmd_model: &CmdModel) {
    tracing::debug!("handle_model start");
    match &cmd_model.command {
        ModelSubcommands::Download => handle_download(global_state),
        ModelSubcommands::Insert => handle_insert(global_state),
    }
    tracing::debug!("handle_model end");
}
