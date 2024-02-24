use leptos::*;

use log::Level;

use ska_frontend::gui::App;

fn main() {
    console_log::init_with_level(Level::Debug).unwrap();
    leptos::mount_to_body(move || view! { <App /> })
}
