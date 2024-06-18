use axum::{
    extract::{Query, State},
    Extension, Json,
};
use hyper::StatusCode;

use crate::modules::{
    auth::auth_models::UserDetails, error::DtoErrorResponse, global_state::GlobalState,
};

use std::sync::Arc;

use super::{do_delete_chat, DtoDeleteUserChatRequest, DtoDeleteUserChatResponse};

pub async fn handle_delete_chat(
    State(global_state): State<Arc<GlobalState>>,
    Extension(user_details): Extension<UserDetails>,
    Query(params): Query<DtoDeleteUserChatRequest>,
) -> Result<Json<DtoDeleteUserChatResponse>, (StatusCode, Json<DtoErrorResponse>)> {
    let query_json = serde_json::to_string(&params).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(DtoErrorResponse {
                status_code: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                is_unexpected_error: false,
                packets: vec![],
            }),
        )
    })?;
    tracing::trace!("handle_delete_chat start params={}", &query_json);

    let create_chat_result = do_delete_chat(&global_state, user_details, params).await;
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
