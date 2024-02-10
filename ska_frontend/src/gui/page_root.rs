use crate::{
    gui::DrawerComp,
    modules::{
        auth::{auth_service, AuthQuery},
        global_state::{GlobalState, GlobalStore},
        storage::auth_storage_service,
    },
};
use leptos::*;
use leptos_router::{use_query, ParamsError};
use oauth2::PkceCodeVerifier;

#[component]
pub fn PageRoot(global_state: GlobalState) -> impl IntoView {
    let (global_state_signal, _) = create_signal(global_state);
    let global_store_signal: RwSignal<GlobalStore> =
        create_rw_signal(GlobalStore::initialize_default());

    let query_memo = use_query::<AuthQuery>();
    provide_context::<ReadSignal<GlobalState>>(global_state_signal);
    provide_context::<RwSignal<GlobalStore>>(global_store_signal);

    let (session_pkce_verifier, session_set_pkce_verifier, session_remove_pkce_verifier) =
        auth_storage_service::use_session_pkce_verifier();

    let check_login_action = create_action(move |()| async move {
        check_login(global_state_signal, query_memo.get(), session_pkce_verifier).await
    });
    check_login_action.dispatch(());

    view! {
        <DrawerComp />
    }
}

async fn check_login(
    global_state: ReadSignal<GlobalState>,
    query_res: Result<AuthQuery, ParamsError>,
    session_pkce_verifier: Signal<String>,
) -> anyhow::Result<()> {
    let query = query_res?;
    if let Some(iss) = query.iss.clone() {
        if let Some(state) = query.state.clone() {
            if let Some(code) = query.code.clone() {
                let pkce_verifier = session_pkce_verifier.get();
                log::info!("pkce_verifier_String={}", pkce_verifier);

                let token_response = auth_service::get_token_response(
                    &global_state.get().oidc_client,
                    PkceCodeVerifier::new(pkce_verifier),
                    code.clone(),
                )
                .await;
            }
        }
    }
    Ok(())
    // log::info!("query={}",)
}
