use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoAssistantOptions {
    pub llms: Vec<DtoLlmData>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DtoLlmData {
    pub name: String,
    pub default_prompt: String,
}
