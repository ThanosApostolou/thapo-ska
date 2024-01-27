use leptos::RwSignal;
use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumString, IntoStaticStr};

#[derive(Clone, Serialize, Deserialize)]
pub struct ChatPacketSignals {
    pub timestamp: RwSignal<u32>,
    pub value: RwSignal<String>,
    pub packet_type: RwSignal<ChatPacketType>,
}

#[derive(Clone, EnumString, Display, AsRefStr, IntoStaticStr, Serialize, Deserialize)]
pub enum ChatPacketType {
    QUESTION,
    ANSWER,
}
