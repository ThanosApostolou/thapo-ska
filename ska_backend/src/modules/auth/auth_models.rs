use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, EnumString, IntoStaticStr};

#[derive(Clone, Debug, Serialize, Deserialize, AsRefStr, IntoStaticStr)]
pub enum AuthTypes {
    Public,
    Authentication,
    AuthorizationNoRoles,
    Authorization(HashSet<AuthRoles>),
}

#[derive(
    Clone, Debug, Serialize, Deserialize, AsRefStr, IntoStaticStr, EnumString, PartialEq, Eq, Hash,
)]
pub enum AuthRoles {
    #[strum(serialize = "SKA_ADMIN")]
    SkaAdmin,
    #[strum(serialize = "SKA_USER")]
    SkaUser,
    #[strum(serialize = "SKA_GUEST")]
    SkaGuest,
}

#[derive(Clone, Debug, Serialize, Deserialize, AsRefStr, IntoStaticStr, PartialEq)]
pub enum AuthUser {
    None,
    Authenticated(UserAuthenticationDetails),
    Authorized(UserDetails),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UserAuthenticationDetails {
    pub sub: String,
    pub username: String,
    pub email: String,
    pub roles: HashSet<AuthRoles>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UserDetails {
    pub user_authentication_details: UserAuthenticationDetails,
    pub user_id: u64,
    pub last_login: u64,
}
