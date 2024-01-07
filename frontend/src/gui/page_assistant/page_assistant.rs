use leptos::*;

use crate::gui::page_assistant::{
    models::{ChatPacketSignals, ChatPacketType},
    CompChat,
};

#[component]
pub fn PageAssistant() -> impl IntoView {
    let chat_packets = create_rw_signal::<Vec<ChatPacketSignals>>(vec![ChatPacketSignals {
        timestamp: create_rw_signal(1),
        value: create_rw_signal("some question".to_string()),
        packet_type: create_rw_signal(ChatPacketType::QUESTION),
    }]);

    view! {
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
    }
}
