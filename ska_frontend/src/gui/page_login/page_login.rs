use std::io;

use anyhow::anyhow;
use leptos::{server_fn::codec::IntoRes, *};
use leptos_router::{use_query, ParamsError, Redirect};
use openidconnect::*;

use crate::{
    gui::page_login::LoginQuery,
    modules::{
        auth::auth_service,
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
    let (storage_refresh_token, storage_set_refresh_token, storage_remove_refresh_token) =
        auth_storage_service::use_storage_refresh_token();
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
            when=move || { check_login_complete.get() == true }
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
    auth_service::after_login(
        global_store,
        &global_state.get().oidc_client,
        session_pkce_verifier,
        session_set_pkce_verifier,
        storage_set_refresh_token,
        query.iss,
        query.state,
        query.code,
    )
    .await?;
    check_login_complete.set(true);
    Ok(())
    // log::info!("query={}",)
}

fn test() -> anyhow::Result<()> {
    let x = Err(anyhow!("test"))?;
    Ok(())
}
