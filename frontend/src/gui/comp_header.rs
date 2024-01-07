use leptos::*;

#[component]
pub fn CompHeader() -> impl IntoView {
    view! {
        <header class="navbar bg-neutral shadow-lg">
            <label for="my-drawer" class="btn drawer-button">
                <img src="assets/icons/bors-3.svg" width="24" />
            </label>
            <a class="btn btn-ghost text-neutral-content text-xl">Specific Knowledge Assistant</a>
        </header>
    }
}
