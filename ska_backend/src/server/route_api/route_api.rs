use super::{route_assistant::build_route_assistant, route_auth::build_route_auth};
use crate::modules::{global_state::GlobalState, web::routes::RouteApi};
use axum::Router;
use std::sync::Arc;

pub fn build_route_api(route_api: &RouteApi) -> Router<Arc<GlobalState>> {
    Router::new()
        .nest(
            route_api.route_assistant.path.self_path,
            build_route_assistant(&route_api.route_assistant),
        )
        .nest(
            route_api.route_auth.path.self_path,
            build_route_auth(&route_api.route_auth),
        )
}
