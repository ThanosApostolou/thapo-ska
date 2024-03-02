use crate::modules::{global_state::GlobalState, web::routes::RouteAuth};
use axum::Router;
use std::sync::Arc;

pub fn build_route_auth(route_auth: &RouteAuth) -> Router<Arc<GlobalState>> {
    let mut router = Router::new();
    for endpoint in &route_auth.endpoints {
        router = router.route(endpoint.path.self_path, endpoint.method_router.clone())
    }
    router
}
