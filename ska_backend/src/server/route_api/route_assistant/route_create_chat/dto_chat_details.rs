use sea_orm::prelude::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoChatDetails {
    pub chat_id: Option<u32>,
    pub user_id: u32,
    pub llm_model: String,
    pub prompt_template: Option<String>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub default_prompt: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoCreateUpdateChatResponse {
    pub chat_id: u32,
}
