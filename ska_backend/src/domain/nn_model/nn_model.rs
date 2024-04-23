use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumString, IntoStaticStr};

#[derive(Debug, Clone)]
pub enum LlmModelTypeEnum {
    CtransformersLlama,
    LlamaCpp,
    HuggingFace,
}

impl LlmModelTypeEnum {
    pub fn get_value(&self) -> &str {
        match self {
            LlmModelTypeEnum::CtransformersLlama => "ctransformers_llama",
            LlmModelTypeEnum::LlamaCpp => "llamacpp",
            LlmModelTypeEnum::HuggingFace => "huggingface",
        }
    }
}

#[derive(Debug, Clone)]
pub enum NnModelEnum {
    AllMiniLML6,
    Llama27BChatGGUF,
    TinyLlama1_1BChat,
}
impl NnModelEnum {
    pub fn get_data(&self) -> NnModelData {
        match self {
            NnModelEnum::AllMiniLML6 => NnModelData {
                name: "all-MiniLM-L6-v2".to_string(),
                repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                rel_path: "all-MiniLM-L6-v2".to_string(),
                model_path: "all-MiniLM-L6-v2".to_string(),
                revision: "e4ce9877abf3edfe10b0d82785e83bdcb973e22e".to_string(),
                allow_patterns: "*".to_string(),
                ignore_patterns: "".to_string(),
                model_type: NnModelType::ModelEmbedding,
                default_prompt: "".to_string(),
                llm_model_type: None,
            },
            NnModelEnum::Llama27BChatGGUF => NnModelData {
                name: "Llama-2-7B-Chat-GGUF".to_string(),
                repo_id: "TheBloke/Llama-2-7B-Chat-GGUF".to_string(),
                rel_path: "Llama-2-7B-Chat-GGUF".to_string(),
                model_path: "Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf".to_string(),
                revision: "191239b3e26b2882fb562ffccdd1cf0f65402adb".to_string(),
                allow_patterns: "llama-2-7b-chat.Q2_K.gguf".to_string(),
                ignore_patterns: "".to_string(),
                model_type: NnModelType::ModelLlm,
                default_prompt: "[INST]
<<SYS>>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
<</SYS>>
Question: {question}
Context: {context}
Answer: [/INST]".to_string(),
                llm_model_type: Some(LlmModelTypeEnum::LlamaCpp),
            },
            NnModelEnum::TinyLlama1_1BChat => NnModelData {
                name: "TinyLlama-1.1B-intermediate-step".to_string(),
                repo_id: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
                rel_path: "TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
                model_path: "TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
                revision: "036fa4651240b9a1487f709833b9e4b96b4c1574".to_string(),
                allow_patterns: "*".to_string(),
                ignore_patterns: "*.bin".to_string(),
                model_type: NnModelType::ModelLlm,
                default_prompt: "[INST]
<<SYS>>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
<</SYS>>
Question: {question}
Context: {context}
Answer: [/INST]".to_string(),
                llm_model_type: Some(LlmModelTypeEnum::HuggingFace),
            },
        }
    }

    pub fn get_data_list() -> Vec<NnModelData> {
        Vec::<NnModelData>::from([
            NnModelEnum::AllMiniLML6.get_data(),
            NnModelEnum::Llama27BChatGGUF.get_data(),
            NnModelEnum::TinyLlama1_1BChat.get_data(),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct NnModelData {
    pub name: String,
    pub repo_id: String,
    pub rel_path: String,
    pub model_path: String,
    pub revision: String,
    pub allow_patterns: String,
    pub ignore_patterns: String,
    pub model_type: NnModelType,
    pub default_prompt: String,
    pub llm_model_type: Option<LlmModelTypeEnum>,
}

impl NnModelData {
    pub fn into_tuple(&self) -> (String, String, String, String) {
        (
            self.repo_id.clone(),
            self.rel_path.clone(),
            self.revision.clone(),
            self.allow_patterns.clone(),
        )
    }
}

#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    AsRefStr,
    IntoStaticStr,
    EnumString,
    Display,
    PartialEq,
    Eq,
    Hash,
)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum NnModelType {
    #[serde(rename = "MODEL_EMBEDDING")]
    ModelEmbedding,
    #[serde(rename = "MODEL_LLM")]
    ModelLlm,
}
