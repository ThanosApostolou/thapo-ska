use axum::{
    extract::State,
    http::{self, HeaderValue},
    routing::{post, MethodRouter},
    Json,
};
use hyper::{HeaderMap, StatusCode};

use crate::{
    modules::{
        auth::{service_auth, DtoUserDetails},
        error::{ErrorCode, ErrorResponse},
        global_state::GlobalState,
        web::utils_web::ControllerResponse,
    },
    server::route_api::route_auth::route_app_login::do_app_login,
};

use std::sync::Arc;

pub const PATH_APP_LOGIN: &'static str = "/app_login";

pub fn build_route_app_login() -> MethodRouter<Arc<GlobalState>> {
    return post(handle_app_login);
}

// basic handler that responds with a static string
pub async fn handle_app_login(
    State(global_state): State<Arc<GlobalState>>,
    headers: HeaderMap,
) -> (http::StatusCode, Json<ControllerResponse<DtoUserDetails>>) {
    tracing::info!("handle_app_login start");
    let response_result = service_auth::authenticate(global_state.clone(), &headers).await;
    match response_result {
        Ok(_) => {
            let result = do_app_login(global_state);
            match result {
                Ok(dto_user_details) => {
                    tracing::info!("handle_app_login end");
                    return (
                        StatusCode::OK,
                        Json(ControllerResponse::Ok(dto_user_details)),
                    );
                }
                Err(error_response) => {
                    tracing::warn!("handle_app_login end error");
                    return (
                        error_response.status_code.into_status_code(),
                        Json(ControllerResponse::Err(error_response)),
                    );
                }
            }
        }
        Err(e) => {
            tracing::error!("error: {}", e);
            tracing::warn!("handle_app_login end unauthorized");
            return (
                ErrorCode::Unauthorized401.into_status_code(),
                Json(ControllerResponse::Err(ErrorResponse {
                    status_code: ErrorCode::Unauthorized401,
                    is_unexpected_error: true,
                    packets: vec![],
                })),
            );
        }
    }
}
