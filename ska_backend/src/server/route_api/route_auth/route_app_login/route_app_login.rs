use axum::{
    extract::{Query, State},
    routing::{post, MethodRouter},
    Json,
};
use hyper::StatusCode;

use crate::modules::global_state::GlobalState;

use std::sync::Arc;

use super::{AskAssistantQuestionRequest, AskAssistantQuestionResponse};

pub const PATH_APP_LOGIN: &'static str = "/app_login";

pub fn build_route_app_login() -> MethodRouter<Arc<GlobalState>> {
    return post(handle_app_login);
}

// basic handler that responds with a static string
pub async fn handle_app_login(
    State(_): State<Arc<GlobalState>>,
) -> (StatusCode, Json<AskAssistantQuestionResponse>) {
    (
        StatusCode::OK,
        Json(AskAssistantQuestionResponse {
            answer: "? - some random answer".to_string(),
        }),
    )
}
