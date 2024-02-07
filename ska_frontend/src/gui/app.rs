use crate::{gui::PageRoot, modules::global_state::GlobalState};
use leptos::*;
use leptos_router::Router;

#[component]
pub fn App() -> impl IntoView {
    let init_action = create_action(move |()| async move { initialize().await });
    init_action.dispatch(());

    view! {
        <Router>
            <Show when=move || init_action.value().with(|value| value.to_owned().is_none())
            >
                <p>loading...</p>
            </Show>
            <Show when=move || init_action.value().with(|value| value.to_owned().is_some())
            >
                <PageRoot global_state=init_action.value().get().to_owned().unwrap() />
            </Show>
        </Router>
    }
}

async fn initialize() -> GlobalState {
    log::info!("initialize called");
    let global_state = GlobalState::initialize_default().await;
    return global_state;
}
