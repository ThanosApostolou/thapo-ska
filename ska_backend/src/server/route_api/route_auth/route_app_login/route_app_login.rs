use axum::{extract::State, Json};
use hyper::{HeaderMap, StatusCode};

use crate::{
    modules::{
        auth::{service_auth, DtoUserDetails},
        error::{ErrorCode, ErrorResponse},
        global_state::GlobalState,
    },
    server::route_api::route_auth::route_app_login::do_app_login,
};

use std::sync::Arc;

// basic handler that responds with a static string
pub async fn handle_app_login(
    State(global_state): State<Arc<GlobalState>>,
    headers: HeaderMap,
) -> Result<Json<DtoUserDetails>, (StatusCode, Json<ErrorResponse>)> {
    tracing::info!("handle_app_login start");
    let response_result = service_auth::authenticate_user(global_state.clone(), &headers).await;
    match response_result {
        Ok(_) => {
            let result = do_app_login(global_state);
            match result {
                Ok(dto_user_details) => {
                    tracing::info!("handle_app_login end");
                    Ok(Json(dto_user_details))
                }
                Err(error_response) => {
                    tracing::warn!("handle_app_login end error");
                    Err((
                        error_response.status_code.into_status_code(),
                        Json(error_response),
                    ))
                }
            }
        }
        Err(e) => {
            tracing::error!("error: {}", e);
            tracing::warn!("handle_app_login end unauthorized");
            Err((
                ErrorCode::Unauthorized401.into_status_code(),
                Json(ErrorResponse {
                    status_code: ErrorCode::Unauthorized401,
                    is_unexpected_error: true,
                    packets: vec![],
                }),
            ))
        }
    }
}
