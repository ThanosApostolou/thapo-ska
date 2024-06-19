use chrono::{NaiveDateTime, Utc};
use sea_orm::{ActiveValue::NotSet, DatabaseTransaction, Set};

use crate::{
    domain::{
        entities::user_chat,
        repos::repo_user_chat,
        user::{
            dto_chat_details::{DtoChatDetails, DtoCreateUpdateChatResponse},
            validator_user_chat::{self, ValidDataCreateUpdateUserChat},
        },
    },
    modules::{
        auth::auth_models::UserDetails,
        db,
        error::{ErrorCode, ErrorResponse},
        global_state::GlobalState,
    },
};

pub async fn do_update_chat(
    global_state: &GlobalState,
    user_details: UserDetails,
    dto_chat_details: DtoChatDetails,
) -> Result<DtoCreateUpdateChatResponse, ErrorResponse> {
    tracing::trace!("do_update_chat start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_write(global_state).await?;
    match validate_update_chat(global_state, &user_details, &txn, dto_chat_details).await {
        Ok(valid_data) => {
            let chat = update_user_chat(global_state, &txn, &user_details, &valid_data)
                .await
                .map_err(|_| {
                    return ErrorResponse {
                        error_code: ErrorCode::UnprocessableEntity422,
                        is_unexpected_error: true,
                        packets: vec![],
                    };
                })?;

            let dto_create_update_chat_response = DtoCreateUpdateChatResponse {
                chat_id: chat.chat_id,
            };
            db::transaction_commit(txn).await?;
            tracing::trace!("do_update_chat end");
            return Ok(dto_create_update_chat_response);
        }
        Err(error) => return Err(error),
    }
}

async fn validate_update_chat(
    _global_state: &GlobalState,
    user_details: &UserDetails,
    txn: &DatabaseTransaction,
    dto_chat_details: DtoChatDetails,
) -> Result<ValidDataCreateUpdateUserChat, ErrorResponse> {
    let valid_data = validator_user_chat::validate_create_update_user_chat(
        _global_state,
        user_details,
        txn,
        dto_chat_details,
        true,
    )
    .await
    .map_err(|packets| ErrorResponse {
        error_code: ErrorCode::UnprocessableEntity422,
        is_unexpected_error: false,
        packets,
    })?;

    Ok(valid_data)
}

async fn update_user_chat(
    _global_state: &GlobalState,
    txn: &sea_orm::DatabaseTransaction,
    user_details: &UserDetails,
    valid_data: &ValidDataCreateUpdateUserChat,
) -> anyhow::Result<user_chat::Model> {
    let current_date = Utc::now();
    let current_date = NaiveDateTime::new(current_date.date_naive(), current_date.time());

    if let Some(existing_user_chat) = &valid_data.existing_user_chat {
        let user_chat_am = user_chat::ActiveModel {
            chat_id: Set(existing_user_chat.chat_id),
            user_id_fk: Set(user_details.user_id),
            chat_name: Set(valid_data.chat_name.clone()),
            llm_model: Set(valid_data.llm_model.name.clone()),
            prompt: Set(valid_data.prompt.clone()),
            temperature: Set(valid_data.temperature.clone()),
            top_p: Set(valid_data.top_p.clone()),
            created_at: NotSet,
            updated_at: Set(current_date),
        };

        let user_chat = repo_user_chat::update(txn, user_chat_am).await?;
        return Ok(user_chat);
    }
    return Err(anyhow::anyhow!("unexpected error"));
}
