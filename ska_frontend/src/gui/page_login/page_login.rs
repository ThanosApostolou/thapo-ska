use leptos::*;
use leptos_router::{use_query, ParamsError, Redirect};

use crate::{
    gui::page_login::LoginQuery,
    modules::{
        auth::service_auth,
        global_state::{GlobalState, GlobalStore},
        storage::auth_storage_service,
    },
};

#[component]
pub fn PageLogin() -> impl IntoView {
    let global_state_signal = expect_context::<ReadSignal<GlobalState>>();
    let global_store_signal = expect_context::<RwSignal<GlobalStore>>();
    let (session_pkce_verifier, session_set_pkce_verifier, _) =
        auth_storage_service::use_session_pkce_verifier();
    let (_, storage_set_refresh_token, _) = auth_storage_service::use_storage_refresh_token();
    let base_href = Signal::derive(move || global_state_signal.get().env_config.base_href.clone());
    let login_query_memo = use_query::<LoginQuery>();
    let check_login_complete = create_rw_signal(false);

    let check_login_action = create_action(move |()| async move {
        check_login(
            global_state_signal,
            global_store_signal,
            login_query_memo.get(),
            session_pkce_verifier,
            check_login_complete,
            session_set_pkce_verifier,
            storage_set_refresh_token,
        )
        .await
    });
    check_login_action.dispatch(());

    view! {
        <Show
            when=move || { check_login_complete.get() }
        >
        <Redirect path=base_href.with(|base_href| {base_href.clone() + "home"})  />
      </Show>
    }
}

async fn check_login(
    global_state: ReadSignal<GlobalState>,
    global_store: RwSignal<GlobalStore>,
    query_res: Result<LoginQuery, ParamsError>,
    session_pkce_verifier: Signal<String>,
    check_login_complete: RwSignal<bool>,
    session_set_pkce_verifier: WriteSignal<String>,
    storage_set_refresh_token: WriteSignal<String>,
) -> Result<(), anyhow::Error> {
    let query = query_res?;
    let global_state = &global_state.get_untracked();
    let global_store = &global_store.get_untracked();
    service_auth::after_login(
        global_store,
        &global_state.oidc_client,
        session_pkce_verifier,
        session_set_pkce_verifier,
        storage_set_refresh_token,
        query.iss,
        query.state,
        query.code,
        &global_state.api_client,
        global_state.env_config.backend_url.clone(),
    )
    .await?;
    check_login_complete.set(true);
    Ok(())
    // log::info!("query={}",)
}
