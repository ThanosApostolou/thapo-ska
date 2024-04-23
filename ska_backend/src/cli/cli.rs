use clap::{Parser, Subcommand};

use crate::{cli::cmd_db::handle_db, modules::global_state::GlobalState};

use super::{
    cmd_db::CmdDbArgs,
    cmd_model::{handle_model, CmdModelArgs},
};

#[derive(Parser, Debug)]
#[command(name = "app-cli", version, about, long_about = None)]
pub struct Cli {
    /// Name of the person to greet
    //     #[arg(short, long)]
    //     name: String,

    //     /// Number of times to greet
    //     #[arg(short, long, default_value_t = 1)]
    //     count: u8,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Model(CmdModelArgs),
    Db(CmdDbArgs),
}

pub async fn start_cli(global_state: &GlobalState) {
    tracing::debug!("cli.start_cli start");
    let cli = Cli::parse();
    match &cli.command {
        Commands::Model(cmd_model) => handle_model(global_state, cmd_model),
        Commands::Db(cmd_db) => handle_db(global_state, cmd_db).await,
    }
    tracing::debug!("cli.start_cli end");
}
