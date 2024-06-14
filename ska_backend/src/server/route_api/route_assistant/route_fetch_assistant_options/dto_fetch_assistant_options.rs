use serde::{Deserialize, Serialize};

use crate::domain::user::dto_chat_details::DtoChatDetails;

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoAssistantOptions {
    pub llms: Vec<DtoLlmData>,
    pub user_chats: Vec<DtoChatDetails>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoLlmData {
    pub name: String,
    pub default_prompt: String,
}
