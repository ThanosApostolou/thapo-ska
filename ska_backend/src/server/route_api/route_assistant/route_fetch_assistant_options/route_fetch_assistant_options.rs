use axum::{extract::State, Extension, Json};
use hyper::StatusCode;

use crate::modules::{
    auth::auth_models::UserDetails, error::DtoErrorResponse, global_state::GlobalState,
};

use std::sync::Arc;

use super::{do_fetch_assistant_options, DtoAssistantOptions};

pub async fn handle_fetch_assistant_options(
    State(global_state): State<Arc<GlobalState>>,
    Extension(user_details): Extension<UserDetails>,
) -> Result<Json<DtoAssistantOptions>, (StatusCode, Json<DtoErrorResponse>)> {
    let ask_assistant_question_result = do_fetch_assistant_options(&global_state, user_details).await;
    match ask_assistant_question_result {
        Ok(ask_assistant_question) => Ok(Json(ask_assistant_question)),
        Err(error_response) => Err((
            error_response.error_code.into_status_code(),
            Json(DtoErrorResponse::from_error_response(error_response)),
        )),
    }
}
