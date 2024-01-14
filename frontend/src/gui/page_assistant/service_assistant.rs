use std::sync::Arc;

use reqwest::{Client, Error, Response};

use super::dtos::{
    AskAssistantQuestionRequest, AskAssistantQuestionResponse, AskAssistantQuestionResponseError,
};

const PATH_API_ASSISTANT: &'static str = "/api/assistant";

pub async fn ask_assistant_question(
    api_client: Arc<Client>,
    backend_url: &String,
    request: &AskAssistantQuestionRequest,
) -> Result<AskAssistantQuestionResponse, AskAssistantQuestionResponseError> {
    let ask_assistant_question_url =
        backend_url.clone() + PATH_API_ASSISTANT + "/ask_assistant_question";
    let http_response = api_client
        .get(&ask_assistant_question_url)
        .query(request)
        .send()
        .await
        .map_err(|error| {
            log::error!("{}", error.status().unwrap_or_default());
            return AskAssistantQuestionResponseError {
                message: "unexpected error".to_string(),
            };
        })?;

    let response = http_response
        .json::<AskAssistantQuestionResponse>()
        .await
        .map_err(|error| {
            log::error!("{}", error);
            return AskAssistantQuestionResponseError {
                message: "unexpected error".to_string(),
            };
        })?;

    return Ok(response);
}
