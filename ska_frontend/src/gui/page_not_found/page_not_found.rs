use leptos::*;

#[component]
pub fn PageNotFound() -> impl IntoView {
    view! {
        <div class="ska-page-container">
            <div class="ska-page-column-flex">
                <div>
                    "Ooops requested page not found"
                </div>
            </div>
        </div>
    }
}
