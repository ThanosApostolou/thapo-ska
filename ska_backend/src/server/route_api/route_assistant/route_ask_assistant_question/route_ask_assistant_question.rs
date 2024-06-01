use axum::{
    extract::{Query, State},
    Json,
};
use hyper::StatusCode;

use crate::modules::{error::DtoErrorResponse, global_state::GlobalState};

use std::sync::Arc;

use super::{do_ask_assistant_question, AskAssistantQuestionRequest, AskAssistantQuestionResponse};

pub async fn handle_ask_assistant_question(
    State(global_state): State<Arc<GlobalState>>,
    Query(query): Query<AskAssistantQuestionRequest>,
) -> Result<Json<AskAssistantQuestionResponse>, (StatusCode, Json<DtoErrorResponse>)> {
    tracing::trace!(
        "handle_ask_assistant_question start llm_model={}, question={}",
        &query.llm_model,
        &query.question
    );
    let ask_assistant_question_result = do_ask_assistant_question(&global_state, query).await;
    match ask_assistant_question_result {
        Ok(ask_assistant_question) => Ok(Json(AskAssistantQuestionResponse {
            answer: ask_assistant_question.answer,
            sources: ask_assistant_question.sources,
        })),
        Err(error_response) => {
            tracing::error!("{}", error_response);
            Err((
                error_response.error_code.into_status_code(),
                Json(DtoErrorResponse::from_error_response(error_response)),
            ))
        }
    }
}
