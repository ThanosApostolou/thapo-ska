use std::sync::Arc;

use crate::{
    gui::{DrawerComp, HeaderComp},
    modules::global_state::GlobalState,
};
use leptos::*;
use leptos_router::Router;

#[component]
pub fn App(global_state: Arc<GlobalState>) -> impl IntoView {
    provide_context::<RwSignal<Arc<GlobalState>>>(create_rw_signal(global_state.clone()));

    log::info!(
        "runtime2 THAPO_SKA_PROFILE={}",
        global_state.clone().env_config.env_profile.clone()
    );

    view! {
        <Router>
            <HeaderComp/>
            profile: {move || global_state.clone().env_config.env_profile.clone()}
            <DrawerComp />
        </Router>
    }
}
