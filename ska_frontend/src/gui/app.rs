use crate::{
    gui::PageRoot,
    modules::{
        auth::service_auth,
        global_state::{GlobalState, GlobalStore},
        storage::auth_storage_service,
    },
};
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
                <PageRoot global_state=init_action.value().get_untracked().to_owned().unwrap().0 global_store=init_action.value().get_untracked().to_owned().unwrap().1 />
            </Show>
        </Router>
    }
}

async fn initialize() -> (GlobalState, GlobalStore) {
    log::info!("initialize called");

    let global_state = GlobalState::initialize_default().await;
    let global_store = GlobalStore::initialize_default();
    let global_store = create_rw_signal(global_store);

    let (storage_refresh_token, storage_set_refresh_token, _) =
        auth_storage_service::use_storage_refresh_token();
    let result = service_auth::initial_check_login(
        &global_store.get_untracked(),
        storage_refresh_token,
        storage_set_refresh_token,
        &global_state.oidc_client,
        &global_state.api_client,
        global_state.env_config.backend_url.clone(),
    )
    .await;
    if let Err(error) = result {
        log::error!("error: {}", error)
    }

    (global_state, global_store.get_untracked())
}
