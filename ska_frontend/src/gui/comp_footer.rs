use std::sync::Arc;

use leptos::*;

use crate::modules::global_state::GlobalState;

#[component]
pub fn CompFooter() -> impl IntoView {
    let global_state = expect_context::<ReadSignal<GlobalState>>();
    view! {
        <footer class="footer p-2 bg-neutral text-neutral-content">
            profile: {move || global_state.get().env_config.env_profile.clone()}
        </footer>
    }
}
