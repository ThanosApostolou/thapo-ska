use serde::{Deserialize, Serialize};

use crate::domain::user::dto_chat_details::{DtoChatDetails, DtoChatPacket};

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoFetchChatMessagesRequest {
    pub chat_id: i64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoFetchChatMessagesResponse {
    pub user_chat: DtoChatDetails,
    pub chat_packets: Vec<DtoChatPacket>,
}
