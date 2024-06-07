use chrono::{NaiveDateTime, Utc};
use sea_orm::{ActiveValue::NotSet, Set};

use crate::{
    domain::{
        entities::user_chat,
        nn_model::{service_nn_model, NnModelData, NnModelType},
    },
    modules::{
        auth::auth_models::UserDetails,
        error::{ErrorCode, ErrorPacket, ErrorResponse},
        global_state::GlobalState,
    },
};

use super::{DtoChatDetails, DtoCreateUpdateChatResponse};

pub async fn do_create_chat(
    global_state: &GlobalState,
    user_details: UserDetails,
    dto_chat_details: DtoChatDetails,
) -> Result<DtoCreateUpdateChatResponse, ErrorResponse> {
    let current_date = Utc::now();
    let current_date = NaiveDateTime::new(current_date.date_naive(), current_date.time());

    let user_chat_am = user_chat::ActiveModel {
        // user_id: NotSet,
        // sub: Set(user_authentication_details.sub.clone()),
        // email: Set(user_authentication_details.email.clone()),
        // last_login: Set(current_date),
        // created_at: Set(current_date),
        // updated_at: Set(current_date),
        chat_id: NotSet,
        user_id_fk: Set(user_details.user_id),
        prompt: Set(dto_chat_details.prompt_template.clone()),
        temperature: Set(dto_chat_details.temperature.clone()),
        top_p: Set(dto_chat_details.top_p.clone()),
        created_at: Set(current_date),
        updated_at: Set(current_date),
    };
    // match user_opt {
    //     Some(user) => {
    //         tracing::debug!("Some(user_opt)");
    //         let mut user_am = user.into_active_model();
    //         user_am.email = sea_orm::Set(user_authentication_details.email.clone());
    //         user_am.updated_at = sea_orm::Set(current_date);
    //         user_am.last_login = sea_orm::Set(current_date);
    //         let user = repo_users::update(&global_state.db_connection, user_am).await?;
    //         tracing::debug!("action_app_login::create_or_update_user end updated");
    //         Ok(user)
    //     }
    //     None => {
    //         tracing::debug!("None(user_opt)");
    //         let user_am = users::ActiveModel {
    //             user_id: NotSet,
    //             sub: Set(user_authentication_details.sub.clone()),
    //             email: Set(user_authentication_details.email.clone()),
    //             last_login: Set(current_date),
    //             created_at: Set(current_date),
    //             updated_at: Set(current_date),
    //         };
    //         let user = repo_users::insert(&global_state.db_connection, user_am).await?;
    //         tracing::debug!("action_app_login::create_or_update_user end created");
    //         Ok(user)
    //     }
    // }

    Ok(DtoCreateUpdateChatResponse { chat_id: 0 })
    // let emb_model: NnModelData = service_nn_model::get_nn_models_list()
    //     .into_iter()
    //     .filter(|nn_model| NnModelType::ModelEmbedding == nn_model.model_type)
    //     .collect::<Vec<NnModelData>>()[0]
    //     .clone();

    // let rag_invoke_result = service_nn_model::rag_invoke(
    //     global_state,
    //     &emb_model.name,
    //     &request.llm_model,
    //     &request.question,
    //     &request.prompt_template,
    // );
    // match rag_invoke_result {
    //     Ok(output) => Ok(DtoCreateUpdateChatResponse {
    //         answer: output.answer,
    //         sources: None,
    //     }),
    //     Err(error) => {
    //         let error_str = error.to_string();
    //         let error_packets: Vec<ErrorPacket> = if error_str.is_empty() {
    //             vec![]
    //         } else {
    //             vec![ErrorPacket {
    //                 message: error_str.clone(),
    //                 backend_message: error_str,
    //             }]
    //         };
    //         Err(ErrorResponse::new(
    //             ErrorCode::UnprocessableEntity422,
    //             true,
    //             error_packets,
    //         ))
    //     }
    // }
}
