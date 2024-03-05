use axum::{extract::State, Extension, Json};
use hyper::{HeaderMap, StatusCode};

use crate::{
    modules::{
        auth::{auth_models::UserAuthenticationDetails, DtoUserDetails},
        error::ErrorResponse,
        global_state::GlobalState,
    },
    server::route_api::route_auth::route_app_login::do_app_login,
};

use std::sync::Arc;

// basic handler that responds with a static string
pub async fn handle_app_login(
    State(global_state): State<Arc<GlobalState>>,
    headers: HeaderMap,
    Extension(user_authentication_details): Extension<UserAuthenticationDetails>,
) -> Result<Json<DtoUserDetails>, (StatusCode, Json<ErrorResponse>)> {
    tracing::info!("handle_app_login start");
    let result = do_app_login(&global_state, user_authentication_details).await;
    match result {
        Ok(dto_user_details) => {
            tracing::info!("handle_app_login end");
            Ok(Json(dto_user_details))
        }
        Err(error_response) => {
            tracing::warn!("handle_app_login end error");
            Err((
                error_response.error_code.into_status_code(),
                Json(error_response),
            ))
        }
    }

    // match auth_user {
    //     AuthUser::Authenticated(user_authentication_details) => {
    //         let result = do_app_login(global_state, user_authentication_details);
    //         match result {
    //             Ok(dto_user_details) => {
    //                 tracing::info!("handle_app_login end");
    //                 Ok(Json(dto_user_details))
    //             }
    //             Err(error_response) => {
    //                 tracing::warn!("handle_app_login end error");
    //                 Err((
    //                     error_response.status_code.into_status_code(),
    //                     Json(error_response),
    //                 ))
    //             }
    //         }
    //     }
    //     _ => {
    //         tracing::warn!("handle_app_login end unauthorized");
    //         Err((
    //             ErrorCode::Unauthorized401.into_status_code(),
    //             Json(ErrorResponse {
    //                 status_code: ErrorCode::Unauthorized401,
    //                 is_unexpected_error: true,
    //                 packets: vec![],
    //             }),
    //         ))
    //     }
    // }
}
