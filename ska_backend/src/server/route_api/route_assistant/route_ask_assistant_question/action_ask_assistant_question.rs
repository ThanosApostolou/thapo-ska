use chrono::{NaiveDateTime, Utc};
use sea_orm::ActiveValue::NotSet;
use sea_orm::{DatabaseTransaction, Set};

use crate::domain::entities::{chat_message, user_chat};
use crate::domain::repos::{repo_chat_message, repo_user_chat};
use crate::domain::user::dto_chat_details::DtoChatPacket;
use crate::domain::user::user_enums::ChatPacketType;
use crate::domain::user::validator_chat_message;
use crate::{
    domain::nn_model::{service_nn_model, NnModelData, NnModelType},
    modules::{
        auth::auth_models::UserDetails,
        db,
        error::{ErrorCode, ErrorPacket, ErrorResponse},
        global_state::GlobalState,
    },
};

use super::{AskAssistantQuestionRequest, AskAssistantQuestionResponse};

pub async fn do_ask_assistant_question(
    global_state: &GlobalState,
    user_details: UserDetails,
    request: AskAssistantQuestionRequest,
) -> Result<AskAssistantQuestionResponse, ErrorResponse> {
    tracing::trace!("do_ask_assistant_question start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_write(global_state).await?;

    let valid_data =
        validate_ask_assistant_question(global_state, &user_details, &txn, &request).await?;

    let response_dto = ask_assistant_question(global_state, &user_details, &txn, &valid_data)
        .await
        .map_err(|e| ErrorResponse {
            error_code: ErrorCode::UnprocessableEntity422,
            is_unexpected_error: true,
            packets: vec![ErrorPacket::new_backend(e.to_string().as_ref())],
        })?;

    db::transaction_commit(txn).await?;
    tracing::trace!("do_ask_assistant_question end");
    return Ok(response_dto);
}

async fn ask_assistant_question(
    global_state: &GlobalState,
    _user_details: &UserDetails,
    txn: &sea_orm::DatabaseTransaction,
    valid_data: &ValidDataAskAssistantQuestion,
) -> anyhow::Result<AskAssistantQuestionResponse> {
    let output = service_nn_model::rag_invoke(
        global_state,
        &valid_data.emb_model,
        &valid_data.llm_model,
        &valid_data.question,
        &valid_data.chat.prompt,
        valid_data.chat.temperature,
        valid_data.chat.top_p,
    )
    .await?;

    // question
    let current_date = Utc::now();
    let current_date = NaiveDateTime::new(current_date.date_naive(), current_date.time());

    let j = serde_json::Value::Object(serde_json::Map::new());
    let chat_message_am = chat_message::ActiveModel {
        chat_message_id: NotSet,
        chat_id_fk: Set(valid_data.chat.chat_id),
        message_type: Set(ChatPacketType::Question.to_string()),
        message_body: Set(valid_data.question.clone()),
        context: Set(j),
        created_at: Set(current_date),
    };
    let chat_message_question = repo_chat_message::insert(txn, chat_message_am).await?;

    let current_date = Utc::now();
    let current_date = NaiveDateTime::new(current_date.date_naive(), current_date.time());

    let chat_message_am = chat_message::ActiveModel {
        chat_message_id: NotSet,
        chat_id_fk: Set(valid_data.chat.chat_id),
        message_type: Set(ChatPacketType::Answer.to_string()),
        message_body: Set(output.answer.clone()),
        context: Set(serde_json::json!(output.context)),
        created_at: Set(current_date),
    };
    let chat_message_answer = repo_chat_message::insert(txn, chat_message_am).await?;

    let dto_question = DtoChatPacket {
        created_at: chat_message_question.created_at.and_utc().timestamp(),
        message_body: chat_message_question.message_body,
        packet_type: ChatPacketType::Question,
        context: vec![],
    };
    let dto_answer = DtoChatPacket {
        created_at: chat_message_answer.created_at.and_utc().timestamp(),
        message_body: chat_message_answer.message_body,
        packet_type: ChatPacketType::Answer,
        context: output.context,
    };
    let ask_assistant_question_response = AskAssistantQuestionResponse {
        question: dto_question,
        answer: dto_answer,
    };
    return Ok(ask_assistant_question_response);
}

pub struct ValidDataAskAssistantQuestion {
    pub chat: user_chat::Model,
    pub chat_messages: Vec<chat_message::Model>,
    pub llm_model: NnModelData,
    pub question: String,
    pub emb_model: NnModelData,
}

async fn validate_ask_assistant_question(
    _global_state: &GlobalState,
    user_details: &UserDetails,
    txn: &DatabaseTransaction,
    request: &AskAssistantQuestionRequest,
) -> Result<ValidDataAskAssistantQuestion, ErrorResponse> {
    let validation_result =
        perform_validation_ask_assistant_question(_global_state, user_details, txn, request).await;

    match validation_result {
        Ok(valid_data) => return Ok(valid_data),
        Err(error_packets) => {
            return Err(ErrorResponse {
                error_code: ErrorCode::UnprocessableEntity422,
                is_unexpected_error: false,
                packets: error_packets,
            })
        }
    }
}

async fn perform_validation_ask_assistant_question(
    _global_state: &GlobalState,
    user_details: &UserDetails,
    txn: &DatabaseTransaction,
    request: &AskAssistantQuestionRequest,
) -> Result<ValidDataAskAssistantQuestion, Vec<ErrorPacket>> {
    tracing::trace!("perform_validation_ask_assistant_question start");
    let mut error_packets: Vec<ErrorPacket> = vec![];

    let emb_model: Option<NnModelData> = service_nn_model::get_nn_models_list()
        .into_iter()
        .find(|nn_model| NnModelType::ModelEmbedding == nn_model.model_type)
        .clone();

    let result_chat_messages =
        repo_user_chat::find_by_chat_id_with_messages(txn, request.chat_id).await;
    if let Err(e) = &result_chat_messages {
        error_packets.push(ErrorPacket {
            message: "".to_string(),
            backend_message: e.to_string(),
        });
    }

    let resutl_syntax_message_body =
        validator_chat_message::syntax_message_body(&Some(&request.question));
    if let Err(error) = &resutl_syntax_message_body {
        error_packets.push(error.clone());
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

    let mut llm_model: Option<NnModelData> = None;
    if let Ok(Some((chat, _))) = &result_chat_messages {
        let skalm_data = NnModelData::get_skalm_data();
        if skalm_data.name == chat.llm_model {
            llm_model = Some(skalm_data);
        } else {
            llm_model = service_nn_model::get_nn_models_list()
                .into_iter()
                .find(|nn_model| {
                    NnModelType::ModelLlm == nn_model.model_type && nn_model.name == chat.llm_model
                });
        }
        if llm_model.is_none() {
            let message = format!("llm_model={} does not exist", chat.llm_model);
            error_packets.push(ErrorPacket {
                message: message.clone(),
                backend_message: message,
            });
        }
    }

    if let (true, Ok(Some((chat, chat_messages))), Some(llm_model), Some(emb_model)) = (
        error_packets.len() == 0,
        result_chat_messages,
        llm_model,
        emb_model,
    ) {
        let valid_data = ValidDataAskAssistantQuestion {
            chat,
            chat_messages,
            llm_model: llm_model,
            question: request.question.clone(),
            emb_model,
        };
        tracing::trace!("perform_validation_ask_assistant_question end ok");
        return Ok(valid_data);
    }

    tracing::trace!("perform_validation_ask_assistant_question end error");
    return Err(error_packets);
}
