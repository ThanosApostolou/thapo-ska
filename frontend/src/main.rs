use leptos::*;

use log::Level;

use frontend::App;

fn main() {
    console_log::init_with_level(Level::Debug).unwrap();
    leptos::mount_to_body(|| view! { <App/> })
}
