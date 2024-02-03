use leptos::*;

use crate::gui::shared::{PATH_ACCOUNT, PATH_HOME};

#[component]
pub fn CompHeader() -> impl IntoView {
    view! {
        <header class="navbar bg-neutral shadow-lg">
            <div class="flex flex-row flex-1">
                <label for="my-drawer" class="btn drawer-button">
                    <img src="assets/icons/bors-3.svg" width="24" />
                </label>
                <a href={PATH_HOME} class="btn btn-ghost text-neutral-content text-xl">Specific Knowledge Assistant</a>
                <span class="flex-1"></span>
                <details class="dropdown dropdown-end">
                    <summary class="m-1 btn">
                        <img src="assets/icons/user-circle.svg" width="24" />
                        <img src="assets/icons/chevron-down.svg" width="16" />
                    </summary>
                    <ul class="p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-32">
                        <li>
                            <a href={PATH_ACCOUNT} class="btn btn-ghost">
                                <img src="assets/icons/arrow-right-start-on-rectangle.svg" width="24" />login
                            </a>
                        </li>
                    </ul>
                </details>
            </div>
        </header>
    }
}
