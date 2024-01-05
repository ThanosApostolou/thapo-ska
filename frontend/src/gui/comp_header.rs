use leptos::*;

#[component]
pub fn HeaderComp() -> impl IntoView {
    view! {
        <div class="navbar bg-base-300 shadow-lg">
            <label for="my-drawer" class="btn drawer-button">
                <img src="public/icons/bors-3.svg" width="32" />
            </label>
            <a class="btn btn-ghost text-xl">Specific Knowledge Assistant</a>
        </div>
    }
}
