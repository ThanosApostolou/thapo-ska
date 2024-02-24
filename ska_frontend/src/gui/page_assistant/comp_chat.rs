use leptos::{web_sys::SubmitEvent, *};
use log::info;

use crate::{
    gui::page_assistant::{
        dtos::AskAssistantQuestionRequest, models::ChatPacketType, service_assistant,
    },
    modules::global_state::{GlobalState, GlobalStore},
};

use super::models::ChatPacketSignals;

#[component]
pub fn CompChat(chat_packets: RwSignal<Vec<ChatPacketSignals>>) -> impl IntoView {
    let global_state = GlobalState::expect_context();
    let global_store = GlobalStore::expect_context();
    let (_question, _set_question) = create_signal("".to_string());
    let (question, set_question) = create_signal("".to_string());

    view! {
        <div class="flex flex-col flex-auto items-stretch">
            <div class="ska-page-column-flex bg-error">
                <For each=chat_packets
                    key=|chat_packet| chat_packet.timestamp
                    let:chat_packet
                >
                    <Show when=move || chat_packet.packet_type.with(|packet_type| matches!(packet_type, ChatPacketType::ANSWER))
                    >
                        <div class="chat chat-start">
                            <div class="chat-bubble">{chat_packet.value}</div>
                        </div>
                    </Show>
                    <Show when=move || chat_packet.packet_type.with(|packet_type| matches!(packet_type, ChatPacketType::QUESTION))
                    >
                        <div class="chat chat-end">
                            <div class="chat-bubble">{chat_packet.value}</div>
                        </div>
                    </Show>
                </For>
            </div>

            <div class="ska-page-column">
                <form class="form-control" on:submit=move |ev| on_submit(ev, global_state, global_store, question, chat_packets)>
                    <label class="label w-full">
                        // <span class="label-text mr-2">Ask</span>
                        <input type="text" placeholder="Ask your question" class="input input-bordered input-primary w-full mr-1" prop:value=question
                            on:input=move |ev| set_question(event_target_value(&ev))
                        />
                        <button type="submit" class="btn btn-outline btn-primary">
                            <img src="assets/icons/paper-airplane.svg" height="32" width="32" />
                        </button>
                    </label>
                </form>
            </div>
        </div>
    }
}

fn on_submit(
    ev: SubmitEvent,
    global_state: ReadSignal<GlobalState>,
    global_store: RwSignal<GlobalStore>,
    question: ReadSignal<String>,
    chat_packets: RwSignal<Vec<ChatPacketSignals>>,
) {
    ev.prevent_default();
    spawn_local(async move {
        info!("question: {}", question.get_untracked());
        chat_packets.update(|chat_packets| {
            chat_packets.push(ChatPacketSignals {
                timestamp: create_rw_signal(2),
                value: create_rw_signal(question.get()),
                packet_type: create_rw_signal(ChatPacketType::QUESTION),
            })
        });

        let request = AskAssistantQuestionRequest {
            question: question.get_untracked(),
        };
        let api_client = global_state.get_untracked().api_client.clone();
        let backend_url = global_state.get_untracked().env_config.backend_url.clone();
        let result = service_assistant::ask_assistant_question(
            global_store,
            api_client,
            &backend_url,
            &request,
        )
        .await;
        match result {
            Ok(response) => chat_packets.update(|chat_packets| {
                let new_timestamp = match chat_packets.last() {
                    Some(packet) => packet.timestamp.get_untracked() + 1,
                    None => 1,
                };
                chat_packets.push(ChatPacketSignals {
                    timestamp: create_rw_signal(new_timestamp),
                    value: create_rw_signal(response.answer.clone()),
                    packet_type: create_rw_signal(ChatPacketType::ANSWER),
                })
            }),
            Err(error) => {
                log::error!("comp_chat error: {}", error.ser().unwrap_or_default())
            }
        }
    });
}
