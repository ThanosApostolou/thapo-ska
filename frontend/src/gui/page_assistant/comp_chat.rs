use leptos::{web_sys::SubmitEvent, *};
use log::info;

use crate::gui::page_assistant::models::ChatPacketType;

use super::models::ChatPacketSignals;

#[component]
pub fn CompChat(chat_packets: RwSignal<Vec<ChatPacketSignals>>) -> impl IntoView {
    let (question, set_question) = create_signal("".to_string());

    view! {
        <div class="flex flex-col flex-auto items-stretch">
            <div class="ska-page-column-flex bg-error">
                <For each=chat_packets
                    key=|chat_packet| chat_packet.timestamp.clone()
                    let:chat_packet
                >
                    <div class="chat chat-end">
                        <div class="chat-bubble">{chat_packet.value}</div>
                    </div>
                </For>
            </div>

            <div class="ska-page-column">
                <form class="form-control" on:submit=move |ev| on_submit(ev, question, chat_packets)>
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
    question: ReadSignal<String>,
    chat_packets: RwSignal<Vec<ChatPacketSignals>>,
) {
    ev.prevent_default();
    info!("question: {}", question.get());
    chat_packets.update(|chat_packets| {
        chat_packets.push(ChatPacketSignals {
            timestamp: create_rw_signal(2),
            value: create_rw_signal(question.get()),
            packet_type: create_rw_signal(ChatPacketType::QUESTION),
        })
    })
}
