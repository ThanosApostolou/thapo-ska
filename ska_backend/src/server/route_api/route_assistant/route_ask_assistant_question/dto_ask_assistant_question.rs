use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionRequest {
    pub question: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionResponse {
    pub answer: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionResponseError {
    pub message: String,
}
