use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::domain::{
    entities::{chat_message, user_chat},
    nn_model::{service_nn_model, DocumentDto},
};

use super::user_enums::ChatPacketType;

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoChatDetails {
    pub chat_id: Option<i64>,
    pub user_id: i64,
    pub chat_name: String,
    pub llm_model: String,
    pub prompt_template: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub default_prompt: String,
}

impl DtoChatDetails {
    pub fn from_user_chat(chat: &user_chat::Model) -> DtoChatDetails {
        let llm_models = service_nn_model::get_nn_models_list();
        let llm_model = llm_models
            .into_iter()
            .find(|llm_model| llm_model.name == chat.llm_model);
        let default_prompt = match llm_model {
            Some(llm_model) => llm_model.default_prompt,
            None => "".to_string(),
        };
        DtoChatDetails {
            chat_id: Some(chat.chat_id),
            user_id: chat.user_id_fk,
            chat_name: chat.chat_name.clone(),
            llm_model: chat.llm_model.clone(),
            prompt_template: chat.prompt.clone(),
            temperature: chat.temperature,
            top_p: chat.top_p,
            default_prompt: default_prompt,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoCreateUpdateChatResponse {
    pub chat_id: i64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoChatPacket {
    pub created_at: i64,
    pub message_body: String,
    pub packet_type: ChatPacketType,
    pub context: Vec<DocumentDto>,
}

impl DtoChatPacket {
    pub fn from_chat_message(message: &chat_message::Model) -> anyhow::Result<DtoChatPacket> {
        let packet_type = ChatPacketType::from_str(message.message_type.as_str())
            .unwrap_or(ChatPacketType::Answer);

        let context: Vec<DocumentDto> = serde_json::from_value(message.context.clone())?;
        return Ok(DtoChatPacket {
            created_at: message.created_at.and_utc().timestamp(),
            message_body: message.message_body.clone(),
            packet_type: packet_type,
            context: context,
        });
    }
}
