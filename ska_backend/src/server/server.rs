use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{str::FromStr, sync::Arc, time};
use tower_http::{catch_panic, cors, timeout, trace};

use crate::modules::{global_state::GlobalState, web::middleware_auth};

use super::route_api::build_route_api;

pub fn create_server(global_state: Arc<GlobalState>) -> Router {
    let tracing_level = tracing::Level::from_str(&global_state.env_config.rust_log).unwrap();

    Router::new()
        .nest(
            &global_state.env_config.server_path,
            Router::new()
                .route("/", get(root))
                .route("/users", post(create_user))
                .nest(
                    global_state.routes.route_api.path.self_path,
                    build_route_api(&global_state.routes.route_api),
                ),
        )
        .layer(
            tower::ServiceBuilder::new()
                .layer(
                    trace::TraceLayer::new_for_http()
                        .make_span_with(
                            trace::DefaultMakeSpan::new()
                                .include_headers(false)
                                .level(tracing_level),
                        )
                        .on_request(trace::DefaultOnRequest::new().level(tracing_level))
                        .on_response(
                            trace::DefaultOnResponse::new()
                                .level(tracing_level)
                                .latency_unit(tower_http::LatencyUnit::Micros),
                        ),
                )
                .layer(catch_panic::CatchPanicLayer::new())
                .layer(cors::CorsLayer::new())
                .layer(timeout::TimeoutLayer::new(time::Duration::from_secs(
                    global_state.env_config.request_timeout,
                )))
                .layer(axum::middleware::from_fn_with_state(
                    global_state.clone(),
                    middleware_auth::middleware_auth,
                )),
        )
        .with_state(global_state)
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
            msg: format!("hello world {}", &global_state.env_config.env_profile),
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
