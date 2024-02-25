use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DtoUserDetails {
    id: String,
    sub: String,
    name: String,
}
