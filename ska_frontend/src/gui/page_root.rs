use std::sync::Arc;

use crate::{
    gui::DrawerComp,
    modules::{
        auth,
        global_state::{EnvConfig, GlobalState, GlobalStore},
    },
};
use leptos::*;
use leptos_router::Router;
use serde_json::de::Read;

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
