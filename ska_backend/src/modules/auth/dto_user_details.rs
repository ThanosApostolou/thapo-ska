use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::auth_models::AuthRoles;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DtoUserDetails {
    pub user_id: i64,
    pub sub: String,
    pub username: String,
    pub email: String,
    pub roles: HashSet<AuthRoles>,
}
