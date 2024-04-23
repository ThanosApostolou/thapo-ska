use clap::{Args, Subcommand};

use crate::{cli::cmd_db::cmd_migrate::handle_migrate, modules::global_state::GlobalState};

#[derive(Args, Debug)]
pub struct CmdDbArgs {
    #[command(subcommand)]
    command: DbSubcommands,
}

#[derive(Subcommand, Debug)]
pub enum DbSubcommands {
    /// migrates db
    Migrate,
}

pub async fn handle_db(global_state: &GlobalState, cmd_model: &CmdDbArgs) {
    tracing::debug!("handle_model start");
    match &cmd_model.command {
        DbSubcommands::Migrate => handle_migrate(global_state).await,
    }
    tracing::debug!("handle_model end");
}
