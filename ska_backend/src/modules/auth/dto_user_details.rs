use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DtoUserDetails {
    pub id: String,
    pub sub: String,
    pub name: String,
}
