use leptos::Params;
use leptos_router::Params;
use serde::{Deserialize, Serialize};
// ?state=mPdz3E-nLKwByWkw8JPlTw&session_state=52a2b960-4ef0-4e80-9b3b-f691008f7e22
// &iss=https%3A%2F%2Fthapo-ska-local.thapo-local.org%3A9443%2Fiam%2Frealms%2Fthapo_ska_local
// &code=9ea304ab-3dc9-4cba-a1ea-699c41966e6a.52a2b960-4ef0-4e80-9b3b-f691008f7e22.4c5d9f78-c4fb-48f8-9f41-9fe0ae501ef7
#[derive(Clone, PartialEq, Serialize, Deserialize, Params)]
pub struct LoginQuery {
    pub state: Option<String>,
    pub iss: Option<String>,
    pub code: Option<String>,
}
