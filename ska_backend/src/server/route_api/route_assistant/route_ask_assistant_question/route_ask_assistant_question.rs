use axum::{
    extract::{Query, State},
    Json,
};
use hyper::StatusCode;

use crate::modules::global_state::GlobalState;

use std::sync::Arc;

use super::{AskAssistantQuestionRequest, AskAssistantQuestionResponse};

// basic handler that responds with a static string
pub async fn handle_ask_assistant_question(
    State(_): State<Arc<GlobalState>>,
    Query(query): Query<AskAssistantQuestionRequest>,
) -> (StatusCode, Json<AskAssistantQuestionResponse>) {
    (
        StatusCode::OK,
        Json(AskAssistantQuestionResponse {
            answer: query.question.clone() + "? - some random answer",
        }),
    )
}
