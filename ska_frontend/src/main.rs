use std::sync::Arc;

use leptos::*;

use log::Level;

use ska_frontend::{gui::App, modules::global_state::GlobalState};

fn main() {
    let global_state = GlobalState::initialize_default();
    let global_state = Arc::new(global_state);
    console_log::init_with_level(Level::Debug).unwrap();
    leptos::mount_to_body(move || view! { <App /> })
}
