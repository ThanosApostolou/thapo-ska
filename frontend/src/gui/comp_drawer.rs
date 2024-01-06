use std::sync::Arc;

use leptos::*;
use leptos_router::{Redirect, Route, Routes};
use log::info;

use crate::{
    gui::{page_assistant::PageAssistant, page_home::PageHome},
    modules::global_state::GlobalState,
};

#[component]
pub fn DrawerComp() -> impl IntoView {
    let global_state = expect_context::<RwSignal<Arc<GlobalState>>>();
    let (checked, setChecked) = create_signal(false);

    view! {
        <div class="drawer">
            <input id="my-drawer" type="checkbox" class="drawer-toggle" prop:checked={checked} />
            <div class="drawer-side">
                <label for="my-drawer" aria-label="close sidebar" class="drawer-overlay"></label>
                <ul class="menu p-4 w-80 min-h-full bg-base-200 text-base-content">
                    <li><a href="home" on:click=move |_| setChecked(false)>Sidebar Item 1</a></li>
                    <li><a href="assistant" on:click=move |_| {setChecked(false)}>Sidebar Item 2</a></li>
                </ul>
            </div>
            <div class="drawer-content">
                <Routes base={global_state.with(|global_state| global_state.env_config.base_href.clone())}>
                    // <AppRoute />
                    <Route path="/home" view=PageHome />
                    <Route path="/assistant" view=PageAssistant />
                    <Route path="" view=move || { view! { <Redirect path="home" /> }} />
                </Routes>

            </div>
        </div>
    }
}

// #[component(transparent)]
// fn AppRoute() -> impl IntoView {
//     view! {
//         <Route path="/home" view=PageHome />
//         <Route path="/assistant" view=PageAssistant />
//         <Route path="" view=move || { view! { <Redirect path="home" /> }} />
//     }
// }
