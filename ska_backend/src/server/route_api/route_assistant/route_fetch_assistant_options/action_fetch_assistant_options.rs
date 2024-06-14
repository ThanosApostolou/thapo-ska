use crate::{
    domain::{
        nn_model::{service_nn_model, NnModelData, NnModelType},
        repos::repo_user_chat,
        user::dto_chat_details::DtoChatDetails,
    },
    modules::{
        auth::auth_models::UserDetails,
        error::{ErrorCode, ErrorResponse},
        global_state::GlobalState,
    },
};

use super::{DtoAssistantOptions, DtoLlmData};

pub async fn do_fetch_assistant_options(
    global_state: &GlobalState,
    user_details: UserDetails,
) -> Result<DtoAssistantOptions, ErrorResponse> {
    let mut models: Vec<NnModelData> = vec![NnModelData::get_skalm_data()];
    models.extend(
        service_nn_model::get_nn_models_list()
            .into_iter()
            .filter(|nn_model| NnModelType::ModelLlm == nn_model.model_type)
            .collect::<Vec<NnModelData>>(),
    );

    let dto_llms: Vec<DtoLlmData> = models
        .into_iter()
        .map(|llm| DtoLlmData {
            name: llm.name,
            default_prompt: llm.default_prompt,
        })
        .collect();

    let user_chats =
        repo_user_chat::find_by_user_id(&global_state.db_connection, user_details.user_id)
            .await
            .map_err(|_| ErrorResponse {
                error_code: ErrorCode::UnprocessableEntity422,
                is_unexpected_error: true,
                packets: vec![],
            })?;

    let dto_user_chats = user_chats
        .into_iter()
        .map(|chat| DtoChatDetails::from_user_chat(chat))
        .collect();

    Ok(DtoAssistantOptions {
        llms: dto_llms,
        user_chats: dto_user_chats,
    })
}
