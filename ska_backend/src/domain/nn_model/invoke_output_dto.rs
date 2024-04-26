use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InvokeOutputDto {
    pub context: Vec<DocumentDto>,
    pub question: String,
    pub answer: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DocumentDto {
    pub page_content: String,
    pub metadata: HashMap<String, String>,
}
