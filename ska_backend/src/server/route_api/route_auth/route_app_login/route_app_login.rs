use axum::{
    extract::State,
    http::{self, HeaderValue},
    routing::{post, MethodRouter},
    Json,
};
use hyper::{HeaderMap, StatusCode};

use crate::{
    modules::{auth::DtoUserDetails, error::ErrorResponse, global_state::GlobalState},
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
) -> (
    http::StatusCode,
    Json<Result<DtoUserDetails, ErrorResponse>>,
) {
    tracing::info!("handle_app_login start");
    let empty = HeaderValue::from_str("empty").unwrap();
    let header_authorization = headers
        .get("Authorization")
        .unwrap_or(&empty)
        .to_str()
        .unwrap();

    tracing::info!("headers {}", header_authorization);
    let result = do_app_login(global_state);
    match result {
        Ok(dto_user_details) => {
            tracing::info!("handle_app_login end");
            return (StatusCode::OK, Json(Ok(dto_user_details)));
        }
        Err(error_response) => {
            tracing::warn!("handle_app_login end error");
            return (
                error_response.status_code.into_status_code(),
                Json(Err(error_response)),
            );
        }
    }
}
