use serde::{Deserialize, Serialize};

use crate::domain::user::dto_chat_details::DtoChatPacket;

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
