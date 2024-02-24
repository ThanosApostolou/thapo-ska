use super::{
    route_assistant::{build_route_assistant, PATH_ASSISTANT},
    route_auth::{build_route_auth, PATH_AUTH},
};
use crate::modules::global_state::GlobalState;
use axum::Router;
use std::sync::Arc;

pub const PATH_API: &'static str = "/api";

pub fn build_route_api() -> Router<Arc<GlobalState>> {
    return Router::new()
        .nest(PATH_ASSISTANT, build_route_assistant())
        .nest(PATH_AUTH, build_route_auth());
}
