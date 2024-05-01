use std::sync::Arc;

use reqwest::Client;

use crate::modules::{error::DtoErrorResponse, global_state::GlobalStore, web::utils_web};

use super::dtos::{AskAssistantQuestionRequest, AskAssistantQuestionResponse, DtoAssistantOptions};

const PATH_API_ASSISTANT: &str = "/api/assistant";

pub async fn ask_assistant_question(
    global_store: &GlobalStore,
    api_client: &Client,
    backend_url: &String,
    request: &AskAssistantQuestionRequest,
) -> Result<AskAssistantQuestionResponse, DtoErrorResponse> {
    let ask_assistant_question_url =
        backend_url.clone() + PATH_API_ASSISTANT + "/ask_assistant_question";
    let request_builder = api_client.get(&ask_assistant_question_url).query(request);

    let response =
        utils_web::send_request::<AskAssistantQuestionResponse>(global_store, request_builder)
            .await?;

    Ok(response)
}

pub async fn fetch_assistant_options(
    global_store: &GlobalStore,
    api_client: &Client,
    backend_url: &String,
) -> Result<DtoAssistantOptions, DtoErrorResponse> {
    let ask_assistant_question_url: String =
        backend_url.clone() + PATH_API_ASSISTANT + "/fetch_assistant_options";
    let request_builder = api_client.get(&ask_assistant_question_url);

    let result =
        utils_web::send_request::<DtoAssistantOptions>(global_store, request_builder).await?;
    Ok(result)
}
