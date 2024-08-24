use sea_orm::{ActiveValue::NotSet, DatabaseTransaction, Set};

use crate::{
    domain::{
        entities::user_chat,
        repos::{repo_chat_message, repo_user_chat},
        user::validator_user_chat::{self, ValidDataDeleteUserChat},
    },
    modules::{
        auth::auth_models::UserDetails,
        db,
        error::{ErrorCode, ErrorResponse},
        global_state::GlobalState,
    },
    server::route_api::route_assistant::DtoDeleteUserChatResponse,
};

use super::DtoDeleteUserChatRequest;

pub async fn do_delete_chat(
    global_state: &GlobalState,
    user_details: UserDetails,
    params: DtoDeleteUserChatRequest,
) -> Result<DtoDeleteUserChatResponse, ErrorResponse> {
    tracing::trace!("do_delete_chat start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_write(global_state).await?;
    match validate_delete_chat(global_state, &user_details, &txn, params).await {
        Ok(valid_data) => {
            let chat_id = delete_user_chat(global_state, &txn, &valid_data)
                .await
                .map_err(|_| {
                    return ErrorResponse {
                        error_code: ErrorCode::UnprocessableEntity422,
                        is_unexpected_error: true,
                        packets: vec![],
                    };
                })?;

            let dto_create_delete_chat_response = DtoDeleteUserChatResponse { chat_id: chat_id };
            db::transaction_commit(txn).await?;
            tracing::trace!("do_delete_chat end");
            return Ok(dto_create_delete_chat_response);
        }
        Err(error) => return Err(error),
    }
}

async fn validate_delete_chat(
    _global_state: &GlobalState,
    user_details: &UserDetails,
    txn: &DatabaseTransaction,
    dto_delete_user_chat: DtoDeleteUserChatRequest,
) -> Result<ValidDataDeleteUserChat, ErrorResponse> {
    let valid_data = validator_user_chat::validate_delete_user_chat(
        _global_state,
        user_details,
        txn,
        dto_delete_user_chat.chat_id,
    )
    .await
    .map_err(|packets| ErrorResponse {
        error_code: ErrorCode::UnprocessableEntity422,
        is_unexpected_error: false,
        packets,
    })?;

    Ok(valid_data)
}

async fn delete_user_chat(
    _global_state: &GlobalState,
    txn: &DatabaseTransaction,
    valid_data: &ValidDataDeleteUserChat,
) -> anyhow::Result<i64> {
    let user_chat_am = user_chat::ActiveModel {
        chat_id: Set(valid_data.existing_user_chat.chat_id),
        user_id_fk: NotSet,
        chat_name: NotSet,
        llm_model: NotSet,
        prompt: NotSet,
        temperature: NotSet,
        top_p: NotSet,
        created_at: NotSet,
        updated_at: NotSet,
    };
    let _delete_result =
        repo_chat_message::delete_all_of_chat(txn, valid_data.existing_user_chat.chat_id).await?;
    let _delete_result = repo_user_chat::delete(txn, user_chat_am).await?;

    return Ok(valid_data.existing_user_chat.chat_id);
}
