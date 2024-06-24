use sea_orm::DatabaseTransaction;

use crate::{
    domain::{
        entities::{chat_message, user_chat},
        repos::repo_user_chat,
        user::dto_chat_details::{DtoChatDetails, DtoChatPacket},
    },
    modules::{
        auth::auth_models::UserDetails,
        db,
        error::{ErrorCode, ErrorPacket, ErrorResponse},
        global_state::GlobalState,
    },
};

use super::{DtoFetchChatMessagesRequest, DtoFetchChatMessagesResponse};

pub async fn do_fetch_chat_messages(
    global_state: &GlobalState,
    user_details: UserDetails,
    request: DtoFetchChatMessagesRequest,
) -> Result<DtoFetchChatMessagesResponse, ErrorResponse> {
    tracing::trace!("do_fetch_chat_messages start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_read(global_state).await?;

    let valid_data =
        validate_fetch_chat_messages(global_state, &user_details, &txn, &request).await?;

    let chat_packets: Vec<DtoChatPacket> = valid_data
        .chat_messages
        .iter()
        .map(|message| DtoChatPacket::from_chat_message(message))
        .collect::<anyhow::Result<Vec<DtoChatPacket>>>()
        .map_err(|e| ErrorResponse {
            error_code: ErrorCode::InternalServerError500,
            is_unexpected_error: false,
            packets: vec![ErrorPacket::from_error(e)],
        })?;

    let dto_assitant_options = DtoFetchChatMessagesResponse {
        user_chat: DtoChatDetails::from_user_chat(&valid_data.chat),
        chat_packets: chat_packets,
    };
    db::transaction_commit(txn).await?;
    tracing::trace!("do_fetch_chat_messages end");
    Ok(dto_assitant_options)
}

pub struct ValidDataFetchChatMessages {
    pub chat: user_chat::Model,
    pub chat_messages: Vec<chat_message::Model>,
}

async fn validate_fetch_chat_messages(
    _global_state: &GlobalState,
    user_details: &UserDetails,
    txn: &DatabaseTransaction,
    request: &DtoFetchChatMessagesRequest,
) -> Result<ValidDataFetchChatMessages, ErrorResponse> {
    let validation_result =
        perform_fetch_chat_messages(_global_state, user_details, txn, request).await;

    match validation_result {
        Ok(valid_data) => return Ok(valid_data),
        Err(error_packets) => {
            return Err(ErrorResponse {
                error_code: ErrorCode::UnprocessableEntity422,
                is_unexpected_error: false,
                packets: error_packets,
            });
        }
    }
}

async fn perform_fetch_chat_messages(
    _global_state: &GlobalState,
    user_details: &UserDetails,
    txn: &DatabaseTransaction,
    request: &DtoFetchChatMessagesRequest,
) -> Result<ValidDataFetchChatMessages, Vec<ErrorPacket>> {
    tracing::trace!("perform_fetch_chat_messages start");
    let mut error_packets: Vec<ErrorPacket> = vec![];

    let result_chat_messages =
        repo_user_chat::find_by_chat_id_with_messages(txn, request.chat_id).await;
    if let Err(e) = &result_chat_messages {
        error_packets.push(ErrorPacket {
            message: "".to_string(),
            backend_message: e.to_string(),
        });
    }

    if let Ok(chat_messages_option) = &result_chat_messages {
        if let Some((chat, _chat_messages)) = chat_messages_option {
            if chat.user_id_fk != user_details.user_id {
                let message = format!(
                    "user_id={} does not own chat_id={}",
                    user_details.user_id, request.chat_id
                );
                error_packets.push(ErrorPacket {
                    message: message.clone(),
                    backend_message: message,
                });
            }
        }
    }

    if let (true, Ok(Some((chat, chat_messages)))) =
        (error_packets.len() == 0, result_chat_messages)
    {
        let valid_data = ValidDataFetchChatMessages {
            chat,
            chat_messages,
        };
        tracing::trace!("perform_fetch_chat_messages end ok");
        return Ok(valid_data);
    }

    tracing::trace!("perform_fetch_chat_messages end error");
    return Err(error_packets);
}
