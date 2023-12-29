use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, str::FromStr, sync::Arc};

use crate::modules::global_state::GlobalState;

pub fn create_server(global_state: Arc<GlobalState>) -> Router {
    // build our application with a routes
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/", get(root))
        // `POST /users` goes to `create_user`
        .route("/users", post(create_user))
        .with_state(global_state);
    return app;
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
