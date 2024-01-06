use leptos::*;

#[component]
pub fn CompHeader() -> impl IntoView {
    view! {
        <header class="navbar bg-base-300 shadow-lg">
            <label for="my-drawer" class="btn drawer-button">
                <img src="public/icons/bors-3.svg" width="24" />
            </label>
            <a class="btn btn-ghost text-xl">Specific Knowledge Assistant</a>
        </header>
    }
}
