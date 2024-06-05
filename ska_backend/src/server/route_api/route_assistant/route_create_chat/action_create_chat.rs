use crate::{
    domain::nn_model::{service_nn_model, NnModelData, NnModelType},
    modules::{
        error::{ErrorCode, ErrorPacket, ErrorResponse},
        global_state::GlobalState,
    },
};

use super::{DtoChatDetails, DtoCreateUpdateChatResponse};

pub async fn do_create_chat(
    global_state: &GlobalState,
    request: DtoChatDetails,
) -> Result<DtoCreateUpdateChatResponse, ErrorResponse> {
    Ok(DtoCreateUpdateChatResponse { chat_id: 0 })
    // let emb_model: NnModelData = service_nn_model::get_nn_models_list()
    //     .into_iter()
    //     .filter(|nn_model| NnModelType::ModelEmbedding == nn_model.model_type)
    //     .collect::<Vec<NnModelData>>()[0]
    //     .clone();

    // let rag_invoke_result = service_nn_model::rag_invoke(
    //     global_state,
    //     &emb_model.name,
    //     &request.llm_model,
    //     &request.question,
    //     &request.prompt_template,
    // );
    // match rag_invoke_result {
    //     Ok(output) => Ok(DtoCreateUpdateChatResponse {
    //         answer: output.answer,
    //         sources: None,
    //     }),
    //     Err(error) => {
    //         let error_str = error.to_string();
    //         let error_packets: Vec<ErrorPacket> = if error_str.is_empty() {
    //             vec![]
    //         } else {
    //             vec![ErrorPacket {
    //                 message: error_str.clone(),
    //                 backend_message: error_str,
    //             }]
    //         };
    //         Err(ErrorResponse::new(
    //             ErrorCode::UnprocessableEntity422,
    //             true,
    //             error_packets,
    //         ))
    //     }
    // }
}
