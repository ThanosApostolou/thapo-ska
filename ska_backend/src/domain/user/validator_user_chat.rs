use crate::{
    domain::{
        entities::user_chat,
        nn_model::{service_nn_model, NnModelData},
        repos::repo_user_chat,
    },
    modules::{
        auth::auth_models::UserDetails,
        error::ErrorPacket,
        global_state::{self, GlobalState},
    },
};

use super::dto_chat_details::DtoChatDetails;

pub struct ValidDataCreateUpdateUserChat {
    pub chat_id: Option<i64>,
    pub user_id: i64,
    pub chat_name: String,
    pub prompt: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub llm_model: NnModelData,
    pub existing_user_chat: Option<user_chat::Model>,
}

pub async fn validate_create_update_user_chat(
    global_state: &GlobalState,
    user_details: &UserDetails,
    dto_chat_details: DtoChatDetails,
    is_update: bool,
) -> Result<ValidDataCreateUpdateUserChat, Vec<ErrorPacket>> {
    let mut errors: Vec<ErrorPacket> = vec![];

    let result_user_id = syntax_user_id(Some(dto_chat_details.user_id));
    if let Err(error) = &result_user_id {
        errors.push(error.clone());
    }

    let result_chat_id = syntax_chat_id(dto_chat_details.chat_id);
    if let Err(error) = &result_chat_id {
        errors.push(error.clone());
    }

    let result_chat_name = syntax_chat_name(Some(&dto_chat_details.chat_name));
    if let Err(error) = &result_chat_name {
        errors.push(error.clone());
    }

    let result_prompt = syntax_prompt(&dto_chat_details.prompt_template);
    if let Err(error) = &result_prompt {
        errors.push(error.clone());
    }

    let result_temperature = syntax_temperature(dto_chat_details.temperature);
    if let Err(error) = &result_temperature {
        errors.push(error.clone());
    }

    let result_top_p = syntax_top_p(dto_chat_details.top_p);
    if let Err(error) = &result_top_p {
        errors.push(error.clone());
    }

    let result_br1 = br1_edited_chat_model_details(&dto_chat_details);
    if let Err(error) = result_br1 {
        errors.push(error);
    };

    let result_br2 = br2_llm_model(&dto_chat_details);
    if let Err(error) = &result_br2 {
        errors.push(error.clone());
    };

    if errors.len() > 0 {
        return Err(errors);
    }

    if is_update {
        let result_chat_id = mandatory_chat_id(dto_chat_details.chat_id);
        if let Err(error) = &result_chat_id {
            errors.push(error.clone());
        };

        if let Ok(chat_id) = result_chat_id {
            let result_br3_existing_chat = br3_existing_chat(global_state, chat_id).await;
            if let Err(error) = &result_br3_existing_chat {
                errors.push(error.clone());
            };
            if let Ok(chat) = result_br3_existing_chat {
                let result_br4_user_can_update_chat = br4_user_can_update_chat(&chat, user_details);
                if let Err(error) = &result_br4_user_can_update_chat {
                    errors.push(error.clone());
                };

                if let (true, Ok(llm), Ok(())) = (
                    errors.is_empty(),
                    result_br2,
                    result_br4_user_can_update_chat,
                ) {
                    return Ok(ValidDataCreateUpdateUserChat {
                        chat_id: Some(chat.chat_id),
                        user_id: user_details.user_id,
                        chat_name: dto_chat_details.chat_name.clone(),
                        prompt: dto_chat_details.prompt_template,
                        temperature: dto_chat_details.temperature,
                        top_p: dto_chat_details.top_p,
                        llm_model: llm,
                        existing_user_chat: Some(chat),
                    });
                } else {
                    return Err(errors);
                }
            }
        }
    } else {
        if let Ok(llm) = result_br2 {
            return Ok(ValidDataCreateUpdateUserChat {
                chat_id: None,
                user_id: user_details.user_id,
                chat_name: dto_chat_details.chat_name.clone(),
                prompt: dto_chat_details.prompt_template,
                temperature: dto_chat_details.temperature,
                top_p: dto_chat_details.top_p,
                llm_model: llm,
                existing_user_chat: None,
            });
        }
    }

    return Err(errors);
}

pub fn syntax_user_id(user_id: Option<i64>) -> Result<(), ErrorPacket> {
    if let Some(user_id) = user_id {
        if user_id < 0 {
            return Err(ErrorPacket {
                message: "user id must be greater or equal than 0".to_string(),
                backend_message: "user id must be greater or equal than 0".to_string(),
            });
        }
    }
    return Ok(());
}

pub fn syntax_chat_id(chat_id: Option<i64>) -> Result<(), ErrorPacket> {
    if let Some(chat_id) = chat_id {
        if chat_id < 0 {
            return Err(ErrorPacket {
                message: "chat id must be greater or equal than 0".to_string(),
                backend_message: "chat id must be greater or equal than 0".to_string(),
            });
        }
    }
    return Ok(());
}
pub fn syntax_chat_name(chat_name: Option<&String>) -> Result<(), ErrorPacket> {
    if let Some(chat_name) = chat_name {
        if chat_name.len() < 1 || chat_name.len() > 200 {
            let message = "chat name must be between 1 and 200 characters".to_string();
            return Err(ErrorPacket {
                message: message.clone(),
                backend_message: message,
            });
        }
    }
    return Ok(());
}

pub fn syntax_prompt(prompt: &Option<String>) -> Result<(), ErrorPacket> {
    if let Some(prompt) = prompt {
        let len = prompt.trim().len();
        if len < 2 || len > 512 {
            return Err(ErrorPacket {
                message: "prompt size must be between 2 and 512".to_string(),
                backend_message: "prompt size must be between 2 and 512".to_string(),
            });
        }
    }
    return Ok(());
}

pub fn syntax_temperature(temperature: Option<f64>) -> Result<(), ErrorPacket> {
    if let Some(temperature) = temperature {
        if temperature < 0.0 || temperature > 1.0 {
            return Err(ErrorPacket {
                message: "temperature must be between 0 and 1".to_string(),
                backend_message: "temperature must be between 0 and 1".to_string(),
            });
        }
    }
    return Ok(());
}

pub fn syntax_top_p(top_p: Option<f64>) -> Result<(), ErrorPacket> {
    if let Some(top_p) = top_p {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(ErrorPacket {
                message: "temperature must be between 0 and 1".to_string(),
                backend_message: "temperature must be between 0 and 1".to_string(),
            });
        }
    }
    return Ok(());
}

pub fn br1_edited_chat_model_details(dto_chat_details: &DtoChatDetails) -> Result<(), ErrorPacket> {
    let all_null = dto_chat_details.prompt_template.is_none()
        && dto_chat_details.temperature.is_none()
        && dto_chat_details.top_p.is_none();
    let some_null = dto_chat_details.prompt_template.is_none()
        || dto_chat_details.temperature.is_none()
        || dto_chat_details.top_p.is_none();

    if some_null && !all_null {
        return Err(ErrorPacket::new_backend(
            "chat llm details should all be null or none null",
        ));
    }
    return Ok(());
}

pub fn br2_llm_model(dto_chat_details: &DtoChatDetails) -> Result<NnModelData, ErrorPacket> {
    let llm_models = service_nn_model::get_nn_models_list();
    let llm = llm_models
        .into_iter()
        .find(|llm| llm.name == dto_chat_details.llm_model);
    match llm {
        Some(llm) => Ok(llm),
        None => {
            let skalm = NnModelData::get_skalm_data();
            if skalm.name == dto_chat_details.llm_model {
                Ok(skalm)
            } else {
                let message = format!("llm with name {} doesn't exist", dto_chat_details.llm_model);
                Err(ErrorPacket {
                    message: message.clone(),
                    backend_message: message,
                })
            }
        }
    }
}

pub async fn br3_existing_chat(
    global_state: &GlobalState,
    chat_id: i64,
) -> Result<user_chat::Model, ErrorPacket> {
    let chat = repo_user_chat::find_by_chat_id(&global_state.db_connection, chat_id)
        .await
        .map_err(|e| ErrorPacket::new_backend(format!("{}", e).as_str()))?;

    let error_message = format!("could not find chat with id {}", chat_id);
    let chat = chat.ok_or(ErrorPacket {
        backend_message: error_message.clone(),
        message: error_message,
    })?;

    Ok(chat)
}

pub fn br4_user_can_update_chat(
    chat: &user_chat::Model,
    user_details: &UserDetails,
) -> Result<(), ErrorPacket> {
    let mut can_update = false;
    if user_details.user_id == chat.user_id_fk {
        can_update = true;
    }

    match can_update {
        true => Ok(()),
        false => Err(ErrorPacket::new_backend(
            format!(
                "user_id: {} does not own chat_id {}",
                user_details.user_id, chat.chat_id
            )
            .as_str(),
        )),
    }
}

pub fn mandatory_user_id(user_id: Option<i64>) -> Result<i64, ErrorPacket> {
    match user_id {
        Some(user_id) => Ok(user_id),
        None => Err(ErrorPacket::new_backend("user_id is mandatory")),
    }
}

pub fn mandatory_chat_id(chat_id: Option<i64>) -> Result<i64, ErrorPacket> {
    match chat_id {
        Some(chat_id) => Ok(chat_id),
        None => Err(ErrorPacket::new_backend("chat_id is mandatory")),
    }
}
