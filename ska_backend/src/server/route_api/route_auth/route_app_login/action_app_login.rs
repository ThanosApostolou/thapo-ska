use chrono::{NaiveDateTime, Utc};
use sea_orm::{ActiveValue::NotSet, DatabaseTransaction, IntoActiveModel, Set};

use crate::{
    domain::{entities::users, repos::repo_users},
    modules::{
        auth::{auth_models::UserAuthenticationDetails, DtoUserDetails},
        db,
        error::ErrorResponse,
        global_state::GlobalState,
    },
};

// basic handler that responds with a static string
pub async fn do_app_login(
    global_state: &GlobalState,
    user_authentication_details: UserAuthenticationDetails,
) -> Result<DtoUserDetails, ErrorResponse> {
    tracing::debug!("action_app_login::do_app_login start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_write(global_state).await?;

    let user_opt = repo_users::find_by_sub(&txn, &user_authentication_details.sub)
        .await
        .map_err(|err| ErrorResponse::new_standard(err.to_string(), true, false))?;

    let user = create_or_update_user(global_state, &txn, user_opt, &user_authentication_details)
        .await
        .map_err(|err| ErrorResponse::new_standard(err.to_string(), true, false))?;

    let dto_user_details = DtoUserDetails {
        user_id: user.user_id,
        sub: user_authentication_details.sub,
        username: user_authentication_details.username,
        email: user_authentication_details.email,
        roles: user_authentication_details.roles.clone(),
    };
    db::transaction_commit(txn).await?;
    tracing::debug!("action_app_login::do_app_login end");
    Ok(dto_user_details)
}

async fn create_or_update_user(
    _global_state: &GlobalState,
    txn: &DatabaseTransaction,
    user_opt: Option<users::Model>,
    user_authentication_details: &UserAuthenticationDetails,
) -> anyhow::Result<users::Model> {
    tracing::debug!("action_app_login::create_or_update_user start");
    let current_date = Utc::now();
    let current_date = NaiveDateTime::new(current_date.date_naive(), current_date.time());
    match user_opt {
        Some(user) => {
            tracing::debug!("Some(user_opt)");
            let mut user_am = user.into_active_model();
            user_am.email = sea_orm::Set(user_authentication_details.email.clone());
            user_am.updated_at = sea_orm::Set(current_date);
            user_am.last_login = sea_orm::Set(current_date);
            let user = repo_users::update(txn, user_am).await?;
            tracing::debug!("action_app_login::create_or_update_user end updated");
            Ok(user)
        }
        None => {
            tracing::debug!("None(user_opt)");
            let user_am = users::ActiveModel {
                user_id: NotSet,
                sub: Set(user_authentication_details.sub.clone()),
                email: Set(user_authentication_details.email.clone()),
                last_login: Set(current_date),
                created_at: Set(current_date),
                updated_at: Set(current_date),
            };
            let user = repo_users::insert(txn, user_am).await?;
            tracing::debug!("action_app_login::create_or_update_user end created");
            Ok(user)
        }
    }
}
