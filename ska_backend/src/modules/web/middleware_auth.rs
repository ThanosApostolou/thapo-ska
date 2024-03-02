use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use hyper::{HeaderMap, StatusCode};
use std::sync::Arc;

use crate::modules::global_state::GlobalState;

pub async fn middleware_auth(
    State(global_state): State<Arc<GlobalState>>,
    // run the `HeaderMap` extractor
    headers: HeaderMap,
    // you can also add more extractors here but the last
    // extractor must implement `FromRequest` which
    // `Request` does
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let path: &str = request.uri().path();
    tracing::info!("middleware_auth path: {}", path);
    let response = next.run(request).await;
    // match get_token(&headers) {
    //     Some(token) if token_is_valid(token) => {
    //         let response = next.run(request).await;
    //         Ok(response)
    //     }
    //     _ => Err(StatusCode::UNAUTHORIZED),
    // }
    Ok(response)
}
