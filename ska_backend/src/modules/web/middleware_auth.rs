use axum::{
    body::Body,
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use hyper::{HeaderMap, StatusCode};
use std::sync::Arc;

use crate::modules::{
    auth::{auth_models::AuthUser, service_auth},
    global_state::GlobalState,
};

pub async fn middleware_auth(
    State(global_state): State<Arc<GlobalState>>,
    // run the `HeaderMap` extractor
    headers: HeaderMap,
    // you can also add more extractors here but the last
    // extractor must implement `FromRequest` which
    // `Request` does
    mut request: Request,
    next: Next,
) -> Response {
    let path: &str = request.uri().path();
    tracing::info!("middleware_auth path: {}", path);
    let auth_type_res = global_state.routes_map.get(request.uri().path()).ok_or(
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .body(Body::empty())
            .unwrap(),
    );

    match auth_type_res {
        Ok(auth_type) => {
            let auth_result =
                service_auth::perform_auth_user(&global_state, &headers, auth_type).await;
            match auth_result {
                Ok(auth_user) => {
                    match auth_user {
                        AuthUser::None => {
                            tracing::debug!("middleware_auth auth_user AuthUser::None ok");
                        }
                        AuthUser::Authenticated(user_authentication_details) => {
                            tracing::debug!(
                                "middleware_auth auth_user AuthUser::Authenticated start"
                            );
                            request.extensions_mut().insert(user_authentication_details);
                        }
                        AuthUser::Authorized(user_details) => {
                            tracing::debug!("middleware_auth auth_user AuthUser::Authorized start");
                            request.extensions_mut().insert(user_details);
                        }
                    }
                    tracing::debug!("middleware_auth ok");
                    next.run(request).await
                }
                Err(e) => {
                    tracing::error!("middleware_auth error: {}", e);
                    Response::builder()
                        .status(e.error_code.into_status_code())
                        .body(Body::empty())
                        .unwrap()
                }
            }
        }
        Err(e) => e,
    }
}
