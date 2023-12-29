use std::{env, sync::Arc};

use leptos::*;

use log::Level;

use frontend::{modules::global_state::GlobalState, gui::App};

fn main() {
    let global_state = GlobalState::initialize_default();
    let global_state = Arc::new(global_state);
    console_log::init_with_level(Level::Debug).unwrap();
    leptos::mount_to_body(move || view! { <App global_state=global_state.clone() /> })
}
