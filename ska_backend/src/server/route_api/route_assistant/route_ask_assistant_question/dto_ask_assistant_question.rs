use serde::{Deserialize, Serialize};

use crate::domain::{nn_model::DocumentDto, user::user_enums::ChatPacketType};

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionRequest {
    pub chat_id: i64,
    pub question: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionResponse {
    pub question: DtoChatPacket,
    pub answer: DtoChatPacket,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoChatPacket {
    pub created_at: i64,
    pub message_body: String,
    pub packet_type: ChatPacketType,
    pub context: Vec<DocumentDto>,
}
