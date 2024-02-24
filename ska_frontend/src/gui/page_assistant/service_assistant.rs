use std::{borrow::Borrow, sync::Arc};

use leptos::{RwSignal, SignalGet};
use reqwest::Client;

use crate::modules::{global_state::GlobalStore, web::utils_web};

use super::dtos::{
    AskAssistantQuestionRequest, AskAssistantQuestionResponse, AskAssistantQuestionResponseError,
};

const PATH_API_ASSISTANT: &str = "/api/assistant";

pub async fn ask_assistant_question(
    global_store: RwSignal<GlobalStore>,
    api_client: Arc<Client>,
    backend_url: &String,
    request: &AskAssistantQuestionRequest,
) -> Result<AskAssistantQuestionResponse, AskAssistantQuestionResponseError> {
    let ask_assistant_question_url =
        backend_url.clone() + PATH_API_ASSISTANT + "/ask_assistant_question";
    let request_builder = api_client.get(&ask_assistant_question_url).query(request);
    let request_builder = utils_web::add_common_headers(&global_store(), request_builder);
    // let request_builder.header(headers)
    let http_response = request_builder.send().await.map_err(|error| {
        log::error!("{}", error.status().unwrap_or_default());
        AskAssistantQuestionResponseError {
            message: "unexpected error".to_string(),
        }
    })?;

    let response = http_response
        .json::<AskAssistantQuestionResponse>()
        .await
        .map_err(|error| {
            log::error!("{}", error);
            AskAssistantQuestionResponseError {
                message: "unexpected error".to_string(),
            }
        })?;

    Ok(response)
}
