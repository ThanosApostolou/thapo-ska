use super::route_app_login::{build_route_app_login, PATH_APP_LOGIN};
use crate::modules::global_state::GlobalState;
use axum::Router;
use std::sync::Arc;

pub const PATH_AUTH: &'static str = "/auth";

pub fn build_route_auth() -> Router<Arc<GlobalState>> {
    return Router::new().route(PATH_APP_LOGIN, build_route_app_login());
}
