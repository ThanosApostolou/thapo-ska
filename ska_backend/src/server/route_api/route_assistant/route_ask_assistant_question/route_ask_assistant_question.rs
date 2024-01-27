use axum::{
    extract::{Query, State},
    routing::{get, MethodRouter},
    Json,
};
use hyper::StatusCode;

use crate::modules::global_state::GlobalState;

use std::sync::Arc;

use super::{AskAssistantQuestionRequest, AskAssistantQuestionResponse};

pub const PATH_ASK_ASSISTANT_QUESTION: &'static str = "/ask_assistant_question";

pub fn build_route_ask_assistant_question() -> MethodRouter<Arc<GlobalState>> {
    return get(handle_ask_assistant_question);
}

// basic handler that responds with a static string
pub async fn handle_ask_assistant_question(
    State(global_state): State<Arc<GlobalState>>,
    Query(query): Query<AskAssistantQuestionRequest>,
) -> (StatusCode, Json<AskAssistantQuestionResponse>) {
    (
        StatusCode::OK,
        Json(AskAssistantQuestionResponse {
            answer: query.question.clone() + "? - some random answer",
        }),
    )
}
