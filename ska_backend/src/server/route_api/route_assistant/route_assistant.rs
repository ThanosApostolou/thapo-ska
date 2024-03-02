use crate::modules::{global_state::GlobalState, web::routes::RouteAssistant};
use axum::Router;
use std::sync::Arc;

pub fn build_route_assistant(route_assistant: &RouteAssistant) -> Router<Arc<GlobalState>> {
    let mut router = Router::new();
    for endpoint in &route_assistant.endpoints {
        router = router.route(endpoint.path.self_path, endpoint.method_router.clone())
    }
    router
    // return Router::new().route(
    //     PATH_ASK_ASSISTANT_QUESTION,
    //     build_route_ask_assistant_question(),
    // );
}
