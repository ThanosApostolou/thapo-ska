use clap::{Args, Subcommand};

use crate::{cli::cmd_model::cmd_download::handle_download, modules::global_state::GlobalState};

#[derive(Args, Debug)]
pub struct CmdModel {
    #[command(subcommand)]
    command: ModelSubcommands,
}

#[derive(Subcommand, Debug)]
pub enum ModelSubcommands {
    /// does testing things
    Download,
}

pub fn handle_model(global_state: &GlobalState, cmd_model: &CmdModel) {
    tracing::debug!("cmd_model.handle_model start");
    match &cmd_model.command {
        ModelSubcommands::Download => handle_download(global_state),
    }
    tracing::debug!("cmd_model.handle_model end");
}
