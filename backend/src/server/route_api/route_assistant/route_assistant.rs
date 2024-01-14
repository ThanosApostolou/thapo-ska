use super::route_ask_assistant_question::{
    build_route_ask_assistant_question, PATH_ASK_ASSISTANT_QUESTION,
};
use crate::modules::global_state::GlobalState;
use axum::{Router};
use std::sync::Arc;

pub const PATH_ASSISTANT: &'static str = "/assistant";

pub fn build_route_assistant() -> Router<Arc<GlobalState>> {
    return Router::new().route(
        PATH_ASK_ASSISTANT_QUESTION,
        build_route_ask_assistant_question(),
    );
}
