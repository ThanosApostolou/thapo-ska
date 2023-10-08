use std::error::Error;

use leptos::{leptos_dom::logging::console_log, *};
use wasm_bindgen::convert::IntoWasmAbi;
use web_sys::MouseEvent;

#[component]
pub fn App() -> impl IntoView {
    let (count, set_count) = create_signal(0);
    let action1 = create_action(move |input: &(MouseEvent,)| {
        let input = input.clone();
        let mouse_event = input.0;
        async move { on_click(count, set_count, &mouse_event).await.unwrap() }
    });

    view! {
        <button
            on:click=move |mouse_event| {
                action1.dispatch((mouse_event.clone(),));
                spawn_local(async move {
                    on_click(count, set_count, &mouse_event).await.unwrap();
                });
            //    use_future()
                //  on_click(count, set_count, mouse_event).unwrap();
            }
        >
            "Click me2: "
            {move || count.get()}
        </button>
    }
}

async fn on_click(
    count: ReadSignal<i32>,
    set_count: WriteSignal<i32>,
    mouse_event: &MouseEvent,
) -> Result<(), Box<dyn Error>> {
    log::info!("mousevent: {}", mouse_event.type_());

    let new_count = count.get() + 1;
    set_count(new_count);
    let body = reqwest::get("https://www.rust-lang.org")
        .await?
        .text()
        .await?;
    log::info!("body: {}", body);
    Ok(())
}
