use std::sync::Arc;

use crate::{
    gui::DrawerComp,
    modules::global_state::{GlobalState, GlobalStore},
};
use leptos::*;
use leptos_router::Router;

#[component]
pub fn App(global_state: Arc<GlobalState>) -> impl IntoView {
    provide_context::<Arc<GlobalState>>(global_state.clone());
    provide_context::<RwSignal<GlobalStore>>(create_rw_signal(GlobalStore::initialize_default()));

    log::info!(
        "runtime2 THAPO_SKA_PROFILE={}",
        global_state.clone().env_config.env_profile.clone()
    );

    view! {
        <Router>
            <DrawerComp />
        </Router>
    }
}
