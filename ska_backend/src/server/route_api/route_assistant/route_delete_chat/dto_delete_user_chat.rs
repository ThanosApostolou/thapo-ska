use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoDeleteUserChatRequest {
    pub chat_id: i64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoDeleteUserChatResponse {
    pub chat_id: i64,
}
