use axum::{
    extract::State,
    http::HeaderValue,
    routing::{post, MethodRouter},
    Json,
};
use hyper::{HeaderMap, StatusCode};

use crate::modules::{auth::DtoUserDetails, global_state::GlobalState};

use std::sync::Arc;

pub const PATH_APP_LOGIN: &'static str = "/app_login";

pub fn build_route_app_login() -> MethodRouter<Arc<GlobalState>> {
    return post(handle_app_login);
}

// basic handler that responds with a static string
pub async fn handle_app_login(
    State(_): State<Arc<GlobalState>>,
    headers: HeaderMap,
) -> (StatusCode, Json<DtoUserDetails>) {
    tracing::trace!("handle_app_login start");
    let empty = HeaderValue::from_str("empty").unwrap();
    let header_authorization = headers
        .get("Authorization")
        .unwrap_or(&empty)
        .to_str()
        .unwrap();

    tracing::info!("headers {}", header_authorization);
    tracing::trace!("handle_app_login end");
    (
        StatusCode::OK,
        Json(DtoUserDetails {
            id: "1".to_string(),
            sub: "a".to_string(),
            name: "name".to_string(),
        }),
    )
}
