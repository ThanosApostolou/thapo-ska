use clap::{Parser, Subcommand};

use crate::modules::global_state::GlobalState;

use super::cmd_model::{handle_model, CmdModel};

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
    Model(CmdModel),
}

pub fn start_cli(global_state: &GlobalState) {
    tracing::debug!("cli.start_cli start");
    let cli = Cli::parse();
    match &cli.command {
        Commands::Model(cmd_model) => handle_model(global_state, cmd_model),
    }
    tracing::debug!("cli.start_cli end");
}
