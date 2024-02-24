use leptos::*;

use crate::{
    gui::shared::PATH_HOME,
    modules::{
        auth::auth_service,
        global_state::{GlobalState, GlobalStore},
        storage::auth_storage_service,
    },
};

#[component]
pub fn CompHeader() -> impl IntoView {
    let global_state = GlobalState::expect_context();
    let global_store = GlobalStore::expect_context();
    let (_, session_set_pkce_verifier, _) = auth_storage_service::use_session_pkce_verifier();

    let login_action =
        create_action(
            move |()| async move { login(global_state, session_set_pkce_verifier).await },
        );
    let logout_action = create_action(move |()| async move { logout(global_state).await });

    view! {
        <header class="navbar bg-neutral shadow-lg">
            <div class="flex flex-row flex-1">
                <label for="my-drawer" class="btn drawer-button">
                    <img src="assets/icons/bors-3.svg" width="24" />
                </label>
                <a href={PATH_HOME} class="btn btn-ghost text-neutral-content text-xl">Specific Knowledge Assistant</a>
                <span class="flex-1"></span>
                <details class="dropdown dropdown-end">
                    <summary class="m-1 btn">
                        <Show when=move || global_store.get().refresh_token.get().is_some()><p>user</p></Show>
                        <img src="assets/icons/user-circle.svg" width="24" />
                        <img src="assets/icons/chevron-down.svg" width="16" />
                    </summary>
                    <ul class="p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-32">
                        <li>
                            <button class="btn btn-ghost" on:click=move |_| {
                                login_action.dispatch(())
                            }>
                                <img src="assets/icons/arrow-right-start-on-rectangle.svg" width="24" />login
                            </button>
                        </li>
                        <li>
                            <button class="btn btn-ghost" on:click=move |_| {
                                logout_action.dispatch(())
                            }>
                                <img src="assets/icons/arrow-right-start-on-rectangle.svg" width="24" />logout
                            </button>
                        </li>
                    </ul>
                </details>
            </div>
        </header>
    }
}

async fn login(
    global_state: ReadSignal<GlobalState>,
    session_set_pkce_verifier: WriteSignal<String>,
) {
    auth_service::login(&global_state.get().oidc_client, session_set_pkce_verifier).await;
}

async fn logout(global_state: ReadSignal<GlobalState>) -> anyhow::Result<()> {
    auth_service::logout(
        &global_state.get_untracked().env_config,
        &global_state.get_untracked().oidc_provider_metadata,
    )
    .await?;
    Ok(())
}
