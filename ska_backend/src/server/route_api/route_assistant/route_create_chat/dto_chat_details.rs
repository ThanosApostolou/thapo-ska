use sea_orm::prelude::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoChatDetails {
    pub chat_id: Option<i64>,
    pub user_id: i64,
    pub chat_name: String,
    pub llm_model: String,
    pub prompt_template: Option<String>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub default_prompt: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoCreateUpdateChatResponse {
    pub chat_id: i64,
}
