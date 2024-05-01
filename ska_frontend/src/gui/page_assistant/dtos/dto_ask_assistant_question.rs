use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionRequest {
    pub question: String,
    pub llm_model: String,
    pub prompt_template: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AskAssistantQuestionResponse {
    pub answer: String,
}
