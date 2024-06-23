use axum::{
    extract::{Query, State},
    Extension, Json,
};
use hyper::StatusCode;

use crate::modules::{
    auth::auth_models::UserDetails, error::DtoErrorResponse, global_state::GlobalState,
};

use std::sync::Arc;

use super::{do_fetch_chat_messages, DtoFetchChatMessagesRequest, DtoFetchChatMessagesResponse};

pub async fn handle_fetch_chat_messages(
    State(global_state): State<Arc<GlobalState>>,
    Extension(user_details): Extension<UserDetails>,
    Query(request): Query<DtoFetchChatMessagesRequest>,
) -> Result<Json<DtoFetchChatMessagesResponse>, (StatusCode, Json<DtoErrorResponse>)> {
    tracing::info!("handle_fetch_chat_messages start");
    let result = do_fetch_chat_messages(&global_state, user_details, request).await;
    match result {
        Ok(dto) => {
            tracing::info!("handle_fetch_chat_messages end");
            return Ok(Json(dto));
        }
        Err(error_response) => {
            tracing::warn!("handle_fetch_chat_messages end error {}", &error_response);
            return Err((
                error_response.error_code.into_status_code(),
                Json(DtoErrorResponse::from_error_response(error_response)),
            ));
        }
    }
}
