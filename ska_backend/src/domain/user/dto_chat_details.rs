use serde::{Deserialize, Serialize};

use crate::domain::{entities::user_chat, nn_model::service_nn_model};

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
    pub fn fromUserChat(chat: user_chat::Model) -> DtoChatDetails {
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
            chat_name: chat.chat_name,
            llm_model: chat.llm_model,
            prompt_template: chat.prompt,
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
