use axum::{extract::State, Extension, Json};
use hyper::StatusCode;

use crate::{
    domain::user::dto_chat_details::{DtoChatDetails, DtoCreateUpdateChatResponse},
    modules::{auth::auth_models::UserDetails, error::DtoErrorResponse, global_state::GlobalState},
};

use std::sync::Arc;

use super::do_create_chat;

pub async fn handle_create_chat(
    State(global_state): State<Arc<GlobalState>>,
    Extension(user_details): Extension<UserDetails>,
    Json(body): Json<DtoChatDetails>,
) -> Result<Json<DtoCreateUpdateChatResponse>, (StatusCode, Json<DtoErrorResponse>)> {
    let query_json = serde_json::to_string(&body).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(DtoErrorResponse {
                status_code: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                is_unexpected_error: false,
                packets: vec![],
            }),
        )
    })?;
    tracing::trace!("handle_ask_assistant_question start query={}", &query_json);

    let create_chat_result = do_create_chat(&global_state, user_details, body).await;
    match create_chat_result {
        Ok(create_chat) => Ok(Json(create_chat)),
        Err(error_response) => {
            tracing::error!("{}", error_response);
            Err((
                error_response.error_code.into_status_code(),
                Json(DtoErrorResponse::from_error_response(error_response)),
            ))
        }
    }
}
