use crate::{
    gui::DrawerComp,
    modules::{
        auth::auth_service,
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

    provide_context::<ReadSignal<GlobalState>>(global_state_signal);
    provide_context::<RwSignal<GlobalStore>>(global_store_signal);



    view! {
        <DrawerComp />
    }
}
