use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumString, IntoStaticStr};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DtoUserDetails {
    pub user_id: i64,
    pub sub: String,
    pub username: String,
    pub email: String,
    pub roles: HashSet<AuthRoles>,
}

#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    AsRefStr,
    IntoStaticStr,
    EnumString,
    Display,
    PartialEq,
    Eq,
    Hash,
)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum AuthRoles {
    #[serde(rename = "SKA_ADMIN")]
    SkaAdmin,
    #[serde(rename = "SKA_USER")]
    SkaUser,
    #[serde(rename = "SKA_GUEST")]
    SkaGuest,
}
