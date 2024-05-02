use leptos::*;

use crate::{
    gui::page_assistant::{
        dtos::DtoAssistantOptions,
        models::{ChatPacketSignals, ChatPacketType},
        CompChat,
    },
    modules::{
        error::DtoErrorPacket,
        global_state::{GlobalState, GlobalStore},
    },
};

use super::service_assistant;

#[component]
pub fn PageAssistant() -> impl IntoView {
    let chat_packets = create_rw_signal::<Vec<ChatPacketSignals>>(vec![ChatPacketSignals {
        timestamp: create_rw_signal(1),
        value: create_rw_signal("Please ask me anything related to this field".to_string()),
        packet_type: create_rw_signal(ChatPacketType::ANSWER),
    }]);

    // signals
    let global_state = GlobalState::expect_context();
    let global_store = GlobalStore::expect_context();
    let (assistant_options, set_assistant_options) =
        create_signal::<Option<DtoAssistantOptions>>(None);
    let (errors, set_errors) = create_signal::<Vec<DtoErrorPacket>>(vec![]);

    // actions
    // let fetch_assistant_options_action = create_action(move |()| async move {
    //     fetch_assistant_options(
    //         &global_state.get().clone(),
    //         &global_store.get(),
    //         set_assistant_options,
    //         set_errors,
    //     )
    //     .await
    // });
    // fetch_assistant_options_action.dispatch(());

    // view
    view! {
        // <Show when=move || fetch_assistant_options_action.value().get().is_none()
        // >
        //     <p>loading...</p>
        // </Show>
        // <Show when=move || fetch_assistant_options_action.value().get().is_some()
        // >
        <div>d</div>
        <Show when=move || errors.with(|value| !value.clone().is_empty())
        >
            <div>
                {
                    errors.get().iter()
                    .map(|error_packet| view! {
                        <p>{&error_packet.message}</p>
                    }).collect::<Vec<_>>()
                }
            </div>
        </Show>
        <Show when=move || errors.with(|value| value.clone().is_empty())
        >
            <div class="flex flex-row flex-auto min-h-0">
                <div class="ska-page-column bg-base-300 max-w-64 break-words">
                    <p>test aaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbccccc</p>
                    <p>test</p>
                    <p>
                        <button class="btn">test</button>
                    </p>
                    <div class="flex flex-col">
                    </div>
                </div>

                <div class="ska-page-column-flex flex">
                    <CompChat chat_packets=chat_packets />
                </div>
            </div>

        </Show>


        //     // view! {
        //     //     {if errors.get().is_empty() {
        //     //         view! {
        //     //             <div>
        //     //             {
        //     //                 errors.get().iter()
        //     //                 .map(|error_packet| view! {
        //     //                     <p>{&error_packet.message}</p>
        //     //                 }).collect::<Vec<_>>()
        //     //             }
        //     //             </div>
        //     //         }
        //     //     } else {
        //     //         view! {
        //     //         }
        //     //     }}
        //     // }
        //     // <Show when=move || errors.with(|value| !value.to_owned().is_empty())
        //     // >
        //     //     {errors.get().iter()
        //     //         .map(|error_packet| view! {
        //     //             <p>{&error_packet.message}</p>
        //     //         }).collect::<Vec<_>>()

        //     //     }
        //     // </Show>
        //     // <Show when=move || errors.with(|value| value.to_owned().is_empty())
        //     // >
        //     //     <div class="flex flex-row flex-auto min-h-0">
        //     //         <div class="ska-page-column bg-base-300 max-w-64 break-words">
        //     //             <p>test aaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbccccc</p>
        //     //             <p>test</p>
        //     //             <p>
        //     //                 <button class="btn">test</button>
        //     //             </p>
        //     //             <div class="flex flex-col">
        //     //             </div>
        //     //         </div>

        //     //         <div class="ska-page-column-flex flex">
        //     //             <CompChat chat_packets=chat_packets />
        //     //         </div>
        //     //     </div>

        //     // </Show>
        // </Show>

    }
}

async fn fetch_assistant_options(
    global_state: &GlobalState,
    global_store: &GlobalStore,
    set_assistant_options: WriteSignal<Option<DtoAssistantOptions>>,
    set_errors: WriteSignal<Vec<DtoErrorPacket>>,
) {
    // let assistant_options_result = service_assistant::fetch_assistant_options(
    //     global_store,
    //     &global_state.api_client,
    //     &global_state.env_config.backend_url,
    // )
    // .await;
    // match assistant_options_result {
    //     Ok(assistant_options) => {
    //         set_assistant_options(Some(assistant_options));
    //         set_errors(vec![]);
    //     }
    //     Err(error_response) => {
    //         set_assistant_options(None);
    //         set_errors(error_response.packets.clone());
    //     }
    // }
}
