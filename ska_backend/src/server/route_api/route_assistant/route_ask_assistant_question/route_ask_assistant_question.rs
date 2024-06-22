use std::sync::Arc;

use axum::{
    extract::{Query, State},
    Extension, Json,
};
use hyper::StatusCode;

use crate::modules::{
    auth::auth_models::UserDetails, error::DtoErrorResponse, global_state::GlobalState,
};

use super::{do_ask_assistant_question, AskAssistantQuestionRequest, AskAssistantQuestionResponse};

pub async fn handle_ask_assistant_question(
    State(global_state): State<Arc<GlobalState>>,
    Extension(user_details): Extension<UserDetails>,
    Query(query): Query<AskAssistantQuestionRequest>,
) -> Result<Json<AskAssistantQuestionResponse>, (StatusCode, Json<DtoErrorResponse>)> {
    tracing::trace!(
        "handle_ask_assistant_question start chat_id={}, question={}",
        &query.chat_id,
        &query.question
    );
    let ask_assistant_question_result =
        do_ask_assistant_question(&global_state, user_details, query).await;
    match ask_assistant_question_result {
        Ok(ask_assistant_question) => Ok(Json(ask_assistant_question)),
        Err(error_response) => {
            tracing::error!("{}", error_response);
            Err((
                error_response.error_code.into_status_code(),
                Json(DtoErrorResponse::from_error_response(error_response)),
            ))
        }
    }
}
