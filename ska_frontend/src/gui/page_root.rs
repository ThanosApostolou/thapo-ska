use crate::{
    gui::DrawerComp,
    modules::global_state::{GlobalState, GlobalStore},
};
use leptos::*;

#[component]
pub fn PageRoot(global_state: GlobalState, global_store: GlobalStore) -> impl IntoView {
    let (global_state_signal, _) = create_signal(global_state);
    let global_store_signal: RwSignal<GlobalStore> = create_rw_signal(global_store);

    provide_context::<ReadSignal<GlobalState>>(global_state_signal);
    provide_context::<RwSignal<GlobalStore>>(global_store_signal);

    view! {
        <DrawerComp />
    }
}
