use crate::modules::{
    auth::{auth_models::UserAuthenticationDetails, DtoUserDetails},
    error::ErrorResponse,
    global_state::GlobalState,
};
use std::sync::Arc;

// basic handler that responds with a static string
pub fn do_app_login(
    _: Arc<GlobalState>,
    user_authentication_details: UserAuthenticationDetails,
) -> Result<DtoUserDetails, ErrorResponse> {
    tracing::trace!("do_app_login start");

    let dto_user_details = DtoUserDetails {
        id: "1".to_string(),
        sub: user_authentication_details.sub,
        name: user_authentication_details.username,
    };
    tracing::trace!("do_app_login end");
    Ok(dto_user_details)
}
