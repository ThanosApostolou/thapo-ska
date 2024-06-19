use crate::{
    domain::nn_model::{service_nn_model, NnModelData, NnModelType},
    modules::{
        db,
        error::{ErrorCode, ErrorPacket, ErrorResponse},
        global_state::GlobalState,
    },
};

use super::{AskAssistantQuestionRequest, AskAssistantQuestionResponse};

pub async fn do_ask_assistant_question(
    global_state: &GlobalState,
    request: AskAssistantQuestionRequest,
) -> Result<AskAssistantQuestionResponse, ErrorResponse> {
    tracing::trace!("do_ask_assistant_question start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_read(global_state).await?;
    let emb_model: NnModelData = service_nn_model::get_nn_models_list()
        .into_iter()
        .filter(|nn_model| NnModelType::ModelEmbedding == nn_model.model_type)
        .collect::<Vec<NnModelData>>()[0]
        .clone();

    let rag_invoke_result = service_nn_model::rag_invoke(
        global_state,
        &emb_model.name,
        &request.llm_model,
        &request.question,
        &request.prompt_template,
    );
    match rag_invoke_result {
        Ok(output) => {
            let ask_assistant_question_response = AskAssistantQuestionResponse {
                answer: output.answer,
                sources: None,
            };
            db::transaction_commit(txn).await?;
            tracing::trace!("do_ask_assistant_question end");
            return Ok(ask_assistant_question_response);
        }
        Err(error) => {
            let error_str = error.to_string();
            let error_packets: Vec<ErrorPacket> = if error_str.is_empty() {
                vec![]
            } else {
                vec![ErrorPacket {
                    message: error_str.clone(),
                    backend_message: error_str,
                }]
            };
            tracing::warn!("do_ask_assistant_question end error");
            return Err(ErrorResponse::new(
                ErrorCode::UnprocessableEntity422,
                true,
                error_packets,
            ));
        }
    }
}
