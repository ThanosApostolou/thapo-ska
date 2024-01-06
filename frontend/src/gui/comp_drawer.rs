use std::sync::Arc;

use leptos::*;
use leptos_router::{use_location, Redirect, Route, Routes};
use log::info;

use crate::{
    gui::{page_assistant::PageAssistant, page_home::PageHome, CompFooter, CompHeader},
    modules::global_state::GlobalState,
};

#[component]
pub fn DrawerComp() -> impl IntoView {
    let global_state = expect_context::<RwSignal<Arc<GlobalState>>>();
    let (checked, setChecked) = create_signal(false);
    let location = use_location();
    let isLocationHome = Signal::derive(move || {
        location
            .pathname
            .get()
            .starts_with((global_state.get().env_config.base_href.clone() + "home").as_str())
    });
    let isAssistantHome = Signal::derive(move || {
        location
            .pathname
            .get()
            .starts_with((global_state.get().env_config.base_href.clone() + "assistant").as_str())
    });

    view! {
        <div class="drawer w-full h-full">
            <input id="my-drawer" type="checkbox" class="drawer-toggle" prop:checked={checked} />
            <div class="drawer-side">
                <label for="my-drawer" aria-label="close sidebar" class="drawer-overlay"></label>
                <ul class="menu p-4 w-80 min-h-full bg-base-200 text-base-content">
                    {location.pathname}
                    <li><a href="home" on:click=move |_| setChecked(false) class=("active", move || isLocationHome())>Home</a></li>
                    <li><a href="assistant" on:click=move |_| {setChecked(false)}>Assistant</a></li>
                </ul>
            </div>
            <div class="drawer-content w-full h-full flex flex-col">
                <div class="flex-none">
                    <CompHeader />
                </div>
                <div class="flex-1">
                    <Routes base={global_state.with(|global_state| global_state.env_config.base_href.clone())}>
                        // <AppRoute />
                        <Route path="/home" view=PageHome />
                        <Route path="/assistant" view=PageAssistant />
                        <Route path="" view=move || { view! { <Redirect path="home" /> }} />
                    </Routes>
                </div>
                <div class="flex-none">
                    <CompFooter />
                </div>

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
