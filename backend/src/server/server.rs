use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::modules::global_state::GlobalState;

use super::route_api::{build_route_api, PATH_API};

pub fn create_server(global_state: Arc<GlobalState>) -> Router {
    let server_router = Router::new()
        .nest(
            "/backend",
            Router::new()
                .route("/", get(root))
                .route("/users", post(create_user))
                .nest(PATH_API, build_route_api()),
        )
        .with_state(global_state);
    return server_router;
}

#[derive(Clone, Serialize, Deserialize)]
struct RootResponseDto {
    msg: String,
}

// basic handler that responds with a static string
async fn root(State(global_state): State<Arc<GlobalState>>) -> (StatusCode, Json<RootResponseDto>) {
    (
        StatusCode::OK,
        Json(RootResponseDto {
            msg: String::from(format!(
                "hello world {}",
                &global_state.env_config.env_profile
            )),
        }),
    )
}

async fn create_user(
    // this argument tells axum to parse the request body
    // as JSON into a `CreateUser` type
    Json(payload): Json<CreateUser>,
) -> (StatusCode, Json<User>) {
    // insert your application logic here
    let user = User {
        id: 1337,
        username: payload.username,
    };

    // this will be converted into a JSON response
    // with a status code of `201 Created`
    (StatusCode::CREATED, Json(user))
}

// the input to our `create_user` handler
#[derive(Deserialize)]
struct CreateUser {
    username: String,
}

// the output to our `create_user` handler
#[derive(Serialize)]
struct User {
    id: u64,
    username: String,
}
