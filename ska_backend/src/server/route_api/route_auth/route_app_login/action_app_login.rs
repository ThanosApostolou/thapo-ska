use crate::modules::{auth::DtoUserDetails, error::ErrorResponse, global_state::GlobalState};
use std::sync::Arc;

// basic handler that responds with a static string
pub fn do_app_login(_: Arc<GlobalState>) -> Result<DtoUserDetails, ErrorResponse> {
    tracing::trace!("do_app_login start");

    let dto_user_details = DtoUserDetails {
        id: "1".to_string(),
        sub: "a".to_string(),
        name: "name".to_string(),
    };
    tracing::trace!("do_app_login end");
    Ok(dto_user_details)
}
